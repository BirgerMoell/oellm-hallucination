#!/bin/bash
#SBATCH --job-name=halu-pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=halu-pairs-%j.out
# On-policy DPO data on one node with 2 GPUs: sample the policy (one shard per GPU), judge every sample (2 GPUs),
# check the chosen candidates and write the missing ones (2 GPUs), build the pairs. Each step is skipped when its output exists, so
# a resubmitted job continues. Everything site-specific comes from the environment:
#
#   POLICY=/path/to/model           the model to be trained
#   POOL=data/train/pool.jsonl      questions (train/build_pool.py)
#   JUDGE=/path/to/gemma-4-31B-it   judge and writer
#   PYTHON=/path/to/python          an environment with vLLM (>= 0.24 for Gemma 4) and transformers
#   CONTAINER=/path/to/image.sif    optional: run python inside this Apptainer/Singularity image
#   BINDS=/a,/b                     optional: container bind paths (default: the working dir, judge and policy dirs)
#   OUT=runs/<name>                 output directory;  K=8 samples per question
#   KNOWN_SHARE=0.35                minimum share of answer pairs (known + sharpen) among the hallucination pairs
#   AUDIT_JUDGE=/path/to/Qwen3.5-27B optional: audit the pairs (audit_pairs.py) with a second judge from another family
#
#   sbatch --account=<account> --partition=<gpu partition> --export=ALL train/run_slurm.sbatch
set -euo pipefail
: "${POLICY:?}" "${POOL:?}" "${JUDGE:?}" "${PYTHON:?}" "${OUT:?}"
K=${K:-8}
HERE=$(cd "$(dirname "$0")" && pwd)
[[ -f "$HERE/sample.py" ]] || HERE=$SLURM_SUBMIT_DIR/train
BENCH=$HERE/../benchmark
py() {
  if [[ -n ${CONTAINER:-} ]]; then
    $(command -v apptainer || command -v singularity) exec --nv \
      -B "${BINDS:-$PWD,$(dirname "$JUDGE"),$(dirname "$POLICY")}" "$CONTAINER" \
      env HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 "$PYTHON" "$@"
  else
    HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 "$PYTHON" "$@"
  fi
}
mkdir -p "$OUT"
if [[ ! -s $OUT/samples.jsonl ]]; then
  echo "== sample $(date)"
  for g in 0 1; do
    CUDA_VISIBLE_DEVICES=$g py "$HERE/sample.py" --pool "$POOL" --model "$POLICY" --k "$K" --shard "$g/2" \
      --out "$OUT/samples-$g.jsonl" &
  done
  wait
  cat "$OUT/samples-0.jsonl" "$OUT/samples-1.jsonl" > "$OUT/samples.jsonl"
fi
if [[ ! -s $OUT/judged.jsonl ]]; then
  echo "== judge $(date)"
  py "$BENCH/judge.py" --answers "$OUT/samples.jsonl" --judge "$JUDGE" --tp 2 --out "$OUT/judged.jsonl"
fi
if [[ ! -s $OUT/curated.jsonl ]]; then
  echo "== curate $(date)"
  py "$HERE/build_pairs.py" curate --judged "$OUT/judged.jsonl" --writer "$JUDGE" --tp 2 --out "$OUT/curated.jsonl"
fi
echo "== pairs $(date)"
py "$HERE/build_pairs.py" pairs --judged "$OUT/judged.jsonl" --curated "$OUT/curated.jsonl" --out-dir "$OUT/pairs" \
  --known-share "${KNOWN_SHARE:-0.35}"
if [[ -n ${AUDIT_JUDGE:-} ]]; then
  echo "== audit $(date)"
  py "$HERE/audit_pairs.py" checks --pairs "$OUT/pairs" --pool "$POOL"
  py "$HERE/audit_pairs.py" judge --pairs "$OUT/pairs" --pool "$POOL" --judge "$AUDIT_JUDGE" --tp 2 --per-cell "${AUDIT_PER_CELL:-30}"
fi
echo "== done $(date)"
