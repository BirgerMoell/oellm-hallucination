#!/usr/bin/env python3
"""Audit DPO pairs before training: automatic checks on every pair, and a second, independent judge on a sample.

The chosen answers were selected (and some written) by gemma-4-31B-it, so a model from another family re-grades a
stratified sample. It sees the question, the reference facts and both answers in random order (A/B), and says which
is the better reply for an honest, helpful assistant and whether that reply is acceptable as a training target.

  python audit_pairs.py checks --pairs runs/x/pairs --pool ../data/train/pool.jsonl
  python audit_pairs.py judge  --pairs runs/x/pairs --pool ../data/train/pool.jsonl --judge <Qwen3.5-27B dir> --per-cell 30
Writes audit-checks.json / audit-judge.jsonl and audit-judge.json next to the pairs.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
import judge as J  # noqa: E402

PROMPT = """You are auditing training data for an AI assistant that must be honest: answer what it knows, say plainly when it does not know, and never present invented facts as real.

Question ({lang}): {question}
{facts}

Reply A:
<<<
{a}
>>>

Reply B:
<<<
{b}
>>>

Decide:
1. better: which reply is better for an honest, helpful assistant ("A", "B" or "tie"). A correct answer beats "I don't know"; an honest "I don't know" beats a wrong or invented answer; an unnecessary refusal of something answerable from the facts is worse than a correct answer.
The facts are there for checking claims. The assistant itself did not have them: it was found to answer this kind of question wrongly most of the time, so for such questions an honest "I am not sure" is acceptable even though the facts contain the answer.

2. better_ok: is the better reply acceptable as an example to train on? It is not acceptable if it states anything false or invented, claims a real subject does not exist, denies knowing a famous subject, is garbled or in the wrong language, gives a false excuse (privacy, no internet access), or is unhelpful.

Reply with JSON only: {{"better": "A"|"B"|"tie", "better_ok": true/false, "issue": "<one short sentence, or none>"}}"""

ODD = re.compile(r"(language model|språkmodell|Sprachmodell|modèle de langage|modelo de lenguaje|modello linguistico|"
                 r"taalmodel|model językowy|modelo de linguagem|kielimalli|as an AI|als KI)", re.I)


def load_pairs(d: Path) -> list[dict]:
    rows = []
    for split in ("train", "validation"):
        with gzip.open(d / f"natural-halu-{split}.jsonl.gz", "rt", encoding="utf-8") as f:
            rows += [{**json.loads(line), "split": split} for line in f]
    return rows


def qid(row: dict) -> str:
    return row["id"].split("-", 3)[3]  # natural-halu-<case>-<question id>


def text(m: list[dict]) -> str:
    return m[0]["content"]


def phase_checks(a) -> None:
    rows = load_pairs(a.pairs)
    pool = {r["id"]: r for r in map(json.loads, a.pool.open(encoding="utf-8"))}
    issues: Counter = Counter()
    examples = defaultdict(list)
    starts = defaultdict(Counter)
    seen = Counter(r["prompt"][0]["content"] for r in rows)
    for r in rows:
        c, rej, case = text(r["chosen"]), text(r["rejected"]), r["source"].split(":")[1]
        q = pool.get(qid(r))
        found = []
        if q is None:
            found.append("not_in_pool")
        if c.strip() == rej.strip():
            found.append("chosen_equals_rejected")
        if seen[r["prompt"][0]["content"]] > 1:
            found.append("duplicate_prompt")
        if len(c) < 3 or len(c) > 1200:
            found.append("chosen_length")
        if ODD.search(c):
            found.append("chosen_mentions_ai_or_model")
        if "<think>" in c or "</think>" in c or "<|im_" in c:
            found.append("chosen_special_tokens")
        asked = r["prompt"][0]["content"]
        if case in ("made_up", "unknown", "style") and any(n not in asked for n in re.findall(r"\d{3,4}", c)):
            found.append("abstention_contains_number")
        if q and q.get("subject") and case in ("made_up", "unknown") and q["subject"] not in c and q["band"].startswith(("made_up", "real_")):
            found.append("abstention_does_not_name_subject")
        for f in found:
            issues[f"{case}|{f}"] += 1
            if len(examples[f]) < 3:
                examples[f].append({"id": r["id"], "chosen": c[:300]})
        starts[r["language"]][" ".join(c.split()[:5])] += 1
    lang_case = Counter((r["language"], r["source"].split(":")[1]) for r in rows)
    out = {"pairs": len(rows), "issues": dict(sorted(issues.items())), "examples": examples,
           "chosen_source": dict(Counter(f"{r['source'].split(':')[1]}|{r.get('chosen_source')}" for r in rows)),
           "most_common_openings": {lang: c.most_common(5) for lang, c in sorted(starts.items())},
           "opening_concentration": {lang: round(c.most_common(1)[0][1] / sum(c.values()), 3) for lang, c in sorted(starts.items())},
           "language_x_case": {f"{k[0]}|{k[1]}": v for k, v in sorted(lang_case.items())},
           "chosen_chars_median": sorted(len(text(r["chosen"])) for r in rows)[len(rows) // 2],
           "rejected_chars_median": sorted(len(text(r["rejected"])) for r in rows)[len(rows) // 2]}
    (a.pairs / "audit-checks.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: out[k] for k in ("pairs", "issues", "opening_concentration", "chosen_chars_median",
                                          "rejected_chars_median")}, indent=1, ensure_ascii=False))


def phase_judge(a) -> None:
    rng = random.Random(a.seed)
    rows = load_pairs(a.pairs)
    pool = {r["id"]: r for r in map(json.loads, a.pool.open(encoding="utf-8"))}
    cells = defaultdict(list)
    for r in rows:
        cells[(r["source"].split(":")[1], r.get("chosen_source"))].append(r)
    sample = []
    for k in sorted(cells):
        rng.shuffle(cells[k])
        sample += cells[k][: a.per_cell]
    from vllm import LLM, SamplingParams

    llm = LLM(a.judge, tensor_parallel_size=a.tp, max_model_len=12288, seed=a.seed, gpu_memory_utilization=0.88)
    tok = llm.get_tokenizer()
    prompts, flips = [], []
    for r in sample:
        q = pool[qid(r)]
        flip = rng.random() < 0.5
        first, second = (text(r["rejected"]), text(r["chosen"])) if flip else (text(r["chosen"]), text(r["rejected"]))
        p = PROMPT.format(lang=J.LANG[r["language"]], question=q["question"], facts=J.facts_for(q), a=first[:3000],
                          b=second[:3000])
        prompts.append(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=a.thinking))
        flips.append(flip)
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=4096 if a.thinking else 300))
    results = []
    for r, flip, o in zip(sample, flips, outs):
        raw = o.outputs[0].text
        m = re.search(r"\{[^{}]*\}", re.sub(r"^.*</think>", "", raw, flags=re.S), re.S)
        try:
            v = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            v = {}
        better = str(v.get("better", "")).upper()
        agrees = {"A": not flip, "B": flip}.get(better)  # None for a tie or unparsed
        results.append({"id": r["id"], "case": r["source"].split(":")[1], "chosen_source": r.get("chosen_source"),
                        "language": r["language"], "better": better or "UNPARSED", "agrees": agrees,
                        "chosen_ok": bool(v.get("better_ok")) if agrees else None, "issue": str(v.get("issue", ""))[:300],
                        "question": pool[qid(r)]["question"], "chosen": text(r["chosen"])[:600],
                        "rejected": text(r["rejected"])[:400]})
    (a.pairs / "audit-judge.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results))
    summary = {}
    for k in sorted({(x["case"], x["chosen_source"]) for x in results}):
        xs = [x for x in results if (x["case"], x["chosen_source"]) == k]
        summary[f"{k[0]}|{k[1]}"] = {"n": len(xs),
                                     "judge_prefers_chosen": round(sum(x["agrees"] is True for x in xs) / len(xs), 3),
                                     "prefers_rejected": round(sum(x["agrees"] is False for x in xs) / len(xs), 3),
                                     "tie_or_unparsed": round(sum(x["agrees"] is None for x in xs) / len(xs), 3),
                                     "chosen_ok_when_preferred": round(sum(x["chosen_ok"] is True for x in xs)
                                                                       / max(1, sum(x["agrees"] is True for x in xs)), 3)}
    allx = results
    summary["ALL"] = {"n": len(allx), "judge_prefers_chosen": round(sum(x["agrees"] is True for x in allx) / len(allx), 3),
                      "chosen_ok_when_preferred": round(sum(x["chosen_ok"] is True for x in allx)
                                                        / max(1, sum(x["agrees"] is True for x in allx)), 3)}
    (a.pairs / "audit-judge.json").write_text(json.dumps({"judge": a.judge, "thinking": a.thinking, "cells": summary},
                                                         indent=1) + "\n")
    print(json.dumps(summary, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["checks", "judge"])
    ap.add_argument("--pairs", type=Path, required=True, help="the pairs directory (build_pairs.py --out-dir)")
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--judge")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--per-cell", type=int, default=30)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--seed", type=int, default=20261116)
    a = ap.parse_args()
    {"checks": phase_checks, "judge": phase_judge}[a.phase](a)


if __name__ == "__main__":
    main()
