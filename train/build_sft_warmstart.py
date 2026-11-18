#!/usr/bin/env python3
"""SFT warm-start data for round 2 (docs/SFT_WARMSTART_PLAN.md), and the other half of the pool for re-sampling.

The question pool is split in two halves by a hash of the question id. Half A's curated chosen answers (round 1's
curated.jsonl: checked by gemma, audited by Qwen3.5-27B) become SFT targets; half B is written out as a pool for
sampling the warm-started model, so the round-2 DPO pairs come from questions it was not fine-tuned on.

Rows (messages format, one conversation per row):
  abstain   half-A `unknown` and `made_up` curated answers (honest "I'm not sure" / "I don't know")
  answer    half-A `known` and `sharpen` curated answers, plus the CoCoNot false-premise chosen answers
  identity  the identity dataset's train split (optional, --identity)
  general   chosen sides of general preference pairs (--general, staged *-train.jsonl.gz), or chat rows from a
            parquet with a `messages` column (--general-parquet), so other behaviour does not drift
Answers are kept at least --answer-share of the hallucination rows: the abstention rows are cut to match.

  python build_sft_warmstart.py --curated runs/x/curated.jsonl --pool ../data/train/pool.jsonl \
      --coconot runs/x/mix/staged/coconot-train.jsonl.gz --identity identity/train.jsonl \
      --general runs/x/mix/staged --general-n 4000 --out runs/x/warmstart
Writes sft-train.jsonl.gz, sft-validation.jsonl.gz, pool-b.jsonl and manifest.json in --out.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def h(x: str) -> float:
    return int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def half(qid: str, salt: str) -> str:
    return "A" if h(f"{salt}:{qid}") < 0.5 else "B"


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--coconot", type=Path, help="staged CoCoNot pairs (stage_mix.py output); chosen answers are used")
    ap.add_argument("--identity", type=Path, help="identity dataset train.jsonl (messages)")
    ap.add_argument("--identity-n", type=int, default=1000)
    ap.add_argument("--identity-languages", nargs="+", default=["en", "de", "fr", "es", "it", "nl", "sv", "pl", "pt", "fi"])
    ap.add_argument("--general", type=Path, help="staged dir with general *-train.jsonl.gz pairs (chosen sides)")
    ap.add_argument("--general-parquet", type=Path, nargs="+", help="parquet files with a messages column instead")
    ap.add_argument("--general-n", type=int, default=4000)
    ap.add_argument("--max-abstain", type=int, default=2000)
    ap.add_argument("--answer-share", type=float, default=0.35)
    ap.add_argument("--salt", default="round2")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261118)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pool = {r["id"]: r for r in read_jsonl(a.pool)}
    curated = read_jsonl(a.curated)

    abstain, answer = [], []
    for c in curated:
        q = pool[c["id"]]
        if half(c["id"], a.salt) != "A":
            continue
        row = {"id": f"warm-{c['case']}-{c['id']}", "messages": [{"role": "user", "content": q["question"]},
                                                                 {"role": "assistant", "content": c["chosen"]}],
               "source": f"oellm-natural-halu-warmstart:{c['case']}", "chosen_source": c["chosen_source"],
               "language": q["language"]}
        (abstain if c["case"] in ("unknown", "made_up", "style") else answer).append(row)
    if a.coconot:
        for p in read_jsonl(a.coconot):
            answer.append({"id": f"warm-{p['id']}", "messages": p["prompt"] + p["chosen"], "source": p["source"],
                           "chosen_source": "coconot", "language": p.get("language", "en")})
    rng.shuffle(abstain)
    cap = min(a.max_abstain, round(len(answer) * (1 - a.answer_share) / a.answer_share))
    abstain = abstain[:cap]

    identity = []
    if a.identity:
        rows = [r for r in read_jsonl(a.identity) if r.get("language") in a.identity_languages]
        rng.shuffle(rows)
        identity = [{"id": r["id"], "messages": r["messages"], "source": "openeurollm-model-identity",
                     "chosen_source": "identity", "language": r["language"]} for r in rows[: a.identity_n]]

    general = []
    if a.general_parquet:
        import pyarrow.parquet as pq

        rows = [r for p in a.general_parquet for r in pq.read_table(p, columns=["messages"]).to_pylist()]
        rng.shuffle(rows)
        for i, r in enumerate(rows[: a.general_n]):
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"] if m["role"] in ("user", "assistant")]
            if msgs and msgs[-1]["role"] == "assistant":
                general.append({"id": f"general-{i}", "messages": msgs, "source": "general-sft", "chosen_source": "general",
                                "language": "en"})
    elif a.general:
        rows = [r for f in sorted(a.general.glob("general-train.jsonl.gz")) for r in read_jsonl(f)]
        rng.shuffle(rows)
        general = [{"id": f"general-{r['id']}", "messages": r["prompt"] + r["chosen"], "source": r["source"],
                    "chosen_source": "general", "language": r.get("language", "")} for r in rows[: a.general_n]]

    rows = abstain + answer + identity + general
    rng.shuffle(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    split = {"train": [r for r in rows if h(r["id"]) >= 0.03], "validation": [r for r in rows if h(r["id"]) < 0.03]}
    for name, part in split.items():
        with gzip.open(a.out / f"sft-{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    pool_b = [q for qid, q in pool.items() if half(qid, a.salt) == "B"]
    (a.out / "pool-b.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in pool_b))
    manifest = {"kind": "oellm_natural_halu_warmstart_v1", "curated": str(a.curated), "salt": a.salt,
                "rows": len(rows), "splits": {k: len(v) for k, v in split.items()},
                "by_part": {"abstain": len(abstain), "answer": len(answer), "identity": len(identity), "general": len(general)},
                "answer_share_of_hallucination_rows": round(len(answer) / max(1, len(answer) + len(abstain)), 3),
                "by_source": dict(Counter(r["source"].split(":")[0] + ":" + r["chosen_source"] for r in rows).most_common()),
                "by_language": dict(sorted(Counter(r["language"] for r in rows).items())),
                "pool_b": {"questions": len(pool_b), "by_band": dict(Counter(q["band"] for q in pool_b))}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(manifest, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
