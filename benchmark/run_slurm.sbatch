#!/bin/bash
#SBATCH --job-name=halu-bench
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=halu-bench-%j.out
# Full benchmark run on one node with 2 GPUs: answers for each model (1 GPU each, one after the other), then one judge
# load (2 GPUs) for all answers, then the report. Everything site-specific comes from the environment:
#
#   MODELS="name1=/path/to/model1 name2=/path/to/model2"   models to test (name = label in the report)
#   NO_THINKING="name2"                                      models without an enable_thinking chat-template switch
#   JUDGE=/path/to/gemma-4-31B-it                            judge model (2 GPUs)
#   PYTHON=/path/to/python                                   an environment with vLLM (>= 0.24 for Gemma 4) and transformers
#   CONTAINER=/path/to/image.sif                             optional: run python inside this Apptainer/Singularity image
#   OUT=runs                                                 output directory (default: runs)
#
#   sbatch --account=<account> --partition=<gpu partition> --export=ALL benchmark/run_slurm.sbatch
set -euo pipefail
: "${MODELS:?}" "${JUDGE:?}" "${PYTHON:?}"
OUT=${OUT:-runs}
HERE=$(cd "$(dirname "$0")" && pwd)
[[ -f "$HERE/generate.py" ]] || HERE=$SLURM_SUBMIT_DIR/benchmark
Q=$HERE/../data/questions.jsonl
py() {
  if [[ -n ${CONTAINER:-} ]]; then
    apptainer exec --nv -B "$PWD" -B "$(dirname "$JUDGE")" "$CONTAINER" env HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 "$PYTHON" "$@"
  else
    HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 "$PYTHON" "$@"
  fi
}
mkdir -p "$OUT"
for spec in $MODELS; do
  name=${spec%%=*}; model=${spec#*=}
  modes="default thinking"; [[ " ${NO_THINKING:-} " == *" $name "* ]] && modes="default"
  [[ -s $OUT/$name/answers.jsonl ]] && { echo "skip $name (answers exist)"; continue; }
  mkdir -p "$OUT/$name"
  CUDA_VISIBLE_DEVICES=0 py "$HERE/generate.py" --questions "$Q" --model "$model" --modes $modes --out "$OUT/$name/answers.jsonl"
done
# one judge load for all models: tag rows with the run name, judge, split back
py - "$OUT" <<'PY'
import json, sys, glob
out = sys.argv[1]
with open(f"{out}/all_answers.jsonl", "w") as f:
    for p in sorted(glob.glob(f"{out}/*/answers.jsonl")):
        for line in open(p):
            f.write(json.dumps({**json.loads(line), "run": p.split("/")[-2]}, ensure_ascii=False) + "\n")
PY
py "$HERE/judge.py" --answers "$OUT/all_answers.jsonl" --judge "$JUDGE" --tp 2 --out "$OUT/all_judged.jsonl"
py - "$OUT" <<'PY'
import json, sys, collections
out = sys.argv[1]
by = collections.defaultdict(list)
for line in open(f"{out}/all_judged.jsonl"):
    by[json.loads(line)["run"]].append(line)
for name, lines in by.items():
    open(f"{out}/{name}/judged.jsonl", "w").writelines(lines)
PY
py "$HERE/score.py" "$OUT"/*/judged.jsonl --names $(for s in $MODELS; do echo "${s%%=*}"; done | sort) --json "$OUT/report.json"
