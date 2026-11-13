#!/usr/bin/env python3
"""Answer the benchmark questions with the model under test, closed-book, exactly as a user would ask them.

  python generate.py --questions ../data/questions.jsonl --model <dir> --out answers.jsonl [--modes default thinking]
  python generate.py ... --base-url http://host:8000/v1 --model <served name>     # any OpenAI-compatible server

No system prompt and no answer format. Greedy decoding. "thinking" passes enable_thinking=True to the chat template
(models whose template has no such switch: use --modes default). The answer stored for scoring is the text after
the last </think>.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def final(text: str) -> str:
    return re.sub(r"^.*</think>", "", text, flags=re.S).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--modes", nargs="+", default=["default", "thinking"])
    ap.add_argument("--base-url", help="OpenAI-compatible endpoint instead of local vLLM")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--thinking-tokens", type=int, default=6144)
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()
    qs = [json.loads(line) for line in a.questions.open(encoding="utf-8")]
    rows = []
    if a.base_url:
        from concurrent.futures import ThreadPoolExecutor

        from openai import OpenAI

        client = OpenAI(base_url=a.base_url, api_key="none")

        def ask(q, mode):
            r = client.chat.completions.create(
                model=a.model, messages=[{"role": "user", "content": q["question"]}], temperature=0.0,
                max_tokens=a.thinking_tokens if mode == "thinking" else a.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": mode == "thinking"}})
            return r.choices[0].message.content or ""

        with ThreadPoolExecutor(16) as pool:
            for mode in a.modes:
                for q, text in zip(qs, pool.map(lambda q: ask(q, mode), qs)):
                    rows.append({**q, "mode": mode, "raw": text, "answer": final(text)})
    else:
        from vllm import LLM, SamplingParams

        llm = LLM(a.model, dtype="bfloat16", max_model_len=8192, tensor_parallel_size=a.tp, seed=0,
                  gpu_memory_utilization=0.88)
        tok = llm.get_tokenizer()
        for mode in a.modes:
            prompts = [tok.apply_chat_template([{"role": "user", "content": q["question"]}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=mode == "thinking")
                       for q in qs]
            params = SamplingParams(temperature=0.0, max_tokens=a.thinking_tokens if mode == "thinking" else a.max_tokens)
            for q, o in zip(qs, llm.generate(prompts, params)):
                text = o.outputs[0].text
                rows.append({**q, "mode": mode, "raw": text, "answer": final(text),
                             "truncated": o.outputs[0].finish_reason != "stop"})
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(len(rows), "answers ->", a.out)


if __name__ == "__main__":
    main()
