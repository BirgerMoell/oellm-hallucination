#!/usr/bin/env python3
"""Stage the DPO training mix: the natural-question hallucination pairs, CoCoNot contrast pairs and general pairs.

- hallucination pairs: build_pairs.py output, as they are;
- CoCoNot (allenai/coconot, config "pref", ODC-By): only the false-presupposition and underspecified contrast pairs,
  where the chosen answer correctly answers a question with a valid premise and the rejected one often wrongly
  denies it (the guard against "it does not exist" for real things). Chosen answers that mention being an AI or a
  knowledge cutoff are dropped, and with --checker the rest must pass the answer check of build_pairs.py;
- general pairs: a random sample of an existing staged preference set (prompt / chosen / rejected message lists),
  so the model's other behaviour does not drift.

  python stage_mix.py --pairs runs/x/pairs --coconot pref.parquet --general <staged dir> --general-n 3000 \
      --out runs/x/mix/staged [--checker <gemma dir> --tp 2]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

AI_OR_CUTOFF = re.compile(r"(as an AI|language model|knowledge cutoff|last update|as of 20\d\d|my training data)", re.I)
COCONOT_SUBCATEGORIES = ("false presuppositions", "underspecified")


def h(x: str) -> float:
    return int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def write(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, required=True)
    ap.add_argument("--coconot", type=Path, required=True)
    ap.add_argument("--general", type=Path, required=True)
    ap.add_argument("--general-n", type=int, default=3000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--checker")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261116)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"sources": {}}
    for split in ("train", "validation"):
        shutil.copy(a.pairs / f"natural-halu-{split}.jsonl.gz", a.out / f"natural-halu-{split}.jsonl.gz")
    manifest["sources"]["natural-halu"] = json.loads((a.pairs / "pairs-manifest.json").read_text())["splits"]

    import pyarrow.parquet as pq

    coco = [r for r in pq.read_table(a.coconot).to_pylist() if r["subcategory"] in COCONOT_SUBCATEGORIES
            and not AI_OR_CUTOFF.search(r["chosen"]) and r["chosen"].strip() and r["rejected"].strip()
            and r["chosen"].strip() != r["rejected"].strip()]
    dropped = Counter()
    if a.checker:
        import build_pairs as BP
        from vllm import LLM, SamplingParams

        llm = LLM(a.checker, tensor_parallel_size=a.tp, max_model_len=8192, seed=a.seed, gpu_memory_utilization=0.88)
        tok = llm.get_tokenizer()
        prompts = [tok.apply_chat_template([{"role": "user", "content": BP.CHECK["answer"].format(
            lang="English", question=r["prompt"], facts="Facts: no reference is given; judge with your own knowledge.",
            answer=r["chosen"][:4000])}], tokenize=False, add_generation_prompt=True) for r in coco]
        verdicts = [BP.parse_check(o.outputs[0].text) for o in
                    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=200))]
        kept = []
        for r, (ok, why) in zip(coco, verdicts):
            if ok:
                kept.append(r)
            else:
                dropped[why[:40]] += 1
        coco = kept
    rows = [{"id": f"coconot-{r['id']}", "prompt": [{"role": "user", "content": r["prompt"]}],
             "chosen": [{"role": "assistant", "content": r["chosen"].strip()}],
             "rejected": [{"role": "assistant", "content": r["rejected"].strip()}],
             "source": f"coconot-pref:{r['subcategory']}", "language": "en"} for r in coco]
    for split, part in (("train", [r for r in rows if h(r["id"]) >= 0.05]), ("validation", [r for r in rows if h(r["id"]) < 0.05])):
        write(a.out / f"coconot-{split}.jsonl.gz", part)
    manifest["sources"]["coconot"] = {"kept": len(rows), "checked": bool(a.checker),
                                      "check_failures": dict(dropped.most_common(10))}

    for split, n in (("train", a.general_n), ("validation", max(50, a.general_n // 20))):
        reservoir: list[str] = []  # uniform sample without loading the whole set
        seen = 0
        for f in sorted(a.general.glob(f"*-{split}.jsonl.gz")):
            with gzip.open(f, "rt", encoding="utf-8") as handle:
                for line in handle:
                    seen += 1
                    if len(reservoir) < n:
                        reservoir.append(line)
                    elif (j := rng.randrange(seen)) < n:
                        reservoir[j] = line
        picked = [json.loads(line) for line in reservoir]
        write(a.out / f"general-{split}.jsonl.gz", picked)
        manifest["sources"][f"general-{split}"] = {"from": str(a.general), "rows": len(picked),
                                                   "by_source": dict(Counter(r["source"].split(":")[0] for r in picked))}
    (a.out / "mix-manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
