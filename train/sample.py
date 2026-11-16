#!/usr/bin/env python3
"""Sample k answers per pool question from the policy (the model to be trained), closed-book, as a user would ask.

  python sample.py --pool ../data/train/pool.jsonl --model <policy dir> --out runs/<name>/samples.jsonl [--k 8]

One output row per sample (the question fields plus "sample", "answer", "truncated"), so benchmark/judge.py grades
them unchanged. Default mode only: the DPO pipeline trains default-mode replies. Temperature 0.7 shows the spread of
what the model would say (including the honest "I don't know" it sometimes gives) without the garbled words that
temperature 1 produces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=768)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--shard", default="0/1", help="i/n: this process samples every n-th question from i (one per GPU)")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams

    i, n = map(int, a.shard.split("/"))
    qs = [json.loads(line) for line in a.pool.open(encoding="utf-8")][i::n]
    llm = LLM(a.model, dtype="bfloat16", max_model_len=4096, tensor_parallel_size=a.tp, seed=a.seed,
              gpu_memory_utilization=0.88, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False) for q in qs]
    params = SamplingParams(n=a.k, temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with a.out.open("w", encoding="utf-8") as f:
        for q, res in zip(qs, llm.generate(prompts, params)):
            for j, o in enumerate(res.outputs):
                f.write(json.dumps({**q, "mode": "default", "sample": j, "answer": o.text.strip(),
                                    "truncated": o.finish_reason != "stop"}, ensure_ascii=False) + "\n")
                count += 1
    a.out.with_suffix(".manifest.json").write_text(json.dumps(
        {"policy": a.model, "pool": str(a.pool), "shard": a.shard, "questions": len(qs), "k": a.k,
         "temperature": a.temperature, "top_p": a.top_p, "max_tokens": a.max_tokens, "seed": a.seed, "mode": "default"}, indent=1) + "\n")
    print(count, "samples ->", a.out)


if __name__ == "__main__":
    main()
