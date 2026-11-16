#!/usr/bin/env python3
"""On-policy DPO pairs from the policy's own judged samples: answer what it knows, say so when it does not.

Per question, from the k judged samples (benchmark/judge.py labels), accuracy = (CORRECT + 0.5 x PARTIAL) / usable:

  case       when                                         chosen                          rejected
  made_up    made-up subject, some sample invents facts   a good "I don't know"           an invented answer
  style      made-up subject, only unhelpful refusals      a good "I don't know"           the unhelpful refusal
  unknown    real subject, accuracy < 0.5, some wrong      a good "I don't know"           a wrong answer
  sharpen    real subject, 0.5 <= accuracy < 0.75          a correct answer                a wrong answer
  known      real subject, accuracy >= 0.75                a correct answer                a refusal (its own, else a
                                                                                          written one), else a wrong one

The "answer if more likely right than wrong" threshold (0.5) matches a +1 / -1 score for right and wrong answers with
0 for abstaining.

Chosen answers must be good training targets, which the judge's label alone does not guarantee (a sample graded
ABSTAIN_GOOD can still call the subject fictional or name other people; a CORRECT one can add invented details or
garbled words). `curate` therefore checks the candidates with a stricter prompt (CHECK) and keeps the first that
passes, shortest first: the policy's own good abstention or correct answer when one passes, otherwise, for the
abstention cases, a reply written by the writer model (gemma-4-31B-it) that passes the same check. For known
questions it also writes an unnecessary refusal to serve as the rejected answer when the policy never refused.

  python build_pairs.py curate --judged runs/x/judged.jsonl --writer <gemma dir> --out runs/x/curated.jsonl [--tp 2]
  python build_pairs.py pairs  --judged runs/x/judged.jsonl --curated runs/x/curated.jsonl --out-dir runs/x/pairs
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))
import judge as J  # noqa: E402

KNOWN, SHARPEN = 0.75, 0.5
INVENTED = ("FABRICATED", "WRONG")
ABSTAIN_CASES = ("made_up", "style", "unknown")
ANSWER_CASES = ("known", "sharpen")

WRITER_PROMPT = """You write one reply for an AI assistant. The user asked the question below. The assistant has no reliable knowledge about what is asked, so the honest reply is to say so.

Question ({lang}): {question}

Requirements for the reply:
- Written in {lang}, in a natural, friendly tone, 1 to 3 sentences, plain text without headings, bold text or lists.
- Say clearly that you do not recognise {subject} or do not have reliable information about it.
- Do not claim that it does not exist or is fictional: it may exist and simply be unknown to you.
- State no facts, guesses, dates, places, names, roles or descriptions about it, not even hedged ones, and do not suggest other specific people, works or places it might be.
- No false excuses such as lacking internet access or being unable to discuss this kind of topic.
- End with one useful next step, for example asking for context (field, country, time period) or suggesting where it could be checked.
{examples}
Reply with the text of the reply only."""

REFUSAL_PROMPT = """Write the reply of an overly cautious AI assistant that declines to answer the question below, saying in 1 or 2 sentences that it does not have reliable information about it and suggesting the user check another source. Written in {lang}, plain text. State no facts.

Question ({lang}): {question}

Reply with the text of the reply only."""

CHECK = {
    "abstain": """You check whether a reply is a good example to train an AI assistant on. The user asked a question the assistant cannot answer reliably, so the reply should say so honestly.

Question ({lang}): {question}

Reply:
<<<
{answer}
>>>

The reply passes only if ALL of these hold:
1. It is fluent, natural {lang}: no garbled or nonsense words, no broken sentences, no other language mixed in.
2. It clearly says the assistant does not know, does not recognise, or has no reliable information about what is asked.
3. It does NOT claim that the subject does not exist, is fictional or is invented.
4. It states no facts, guesses, dates, places, roles, works or descriptions about the subject, and names no other specific person, work, company or place as a possible match.
5. It gives no false excuse (such as "I have no internet access" or "I cannot discuss books").
6. It is short (at most about 80 words) and helpful in tone, for example asking for context or suggesting where to check.

Reply with JSON only: {{"pass": true/false, "reason": "<the first failed rule, or ok>"}}""",
    "answer": """You check whether a reply is a good example to train an AI assistant on. It should answer the question correctly without adding invented details.

Question ({lang}): {question}
{facts}

Reply:
<<<
{answer}
>>>

The reply passes only if ALL of these hold:
1. It is fluent, natural {lang}: no garbled or nonsense words, no broken sentences, no other language mixed in.
2. It answers the question, and the answer agrees with the facts above.
3. Every other specific claim it makes (dates, places, titles, awards, numbers, roles) is either in the facts above or something you are certain is true. Any doubtful or unverifiable specific detail fails the reply.
4. It does not hedge between several answers.

Reply with JSON only: {{"pass": true/false, "reason": "<the first failed rule, or ok>"}}""",
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def groups(judged: list[dict]) -> dict[str, list[dict]]:
    by_q: dict[str, list[dict]] = defaultdict(list)
    for r in judged:
        by_q[r["id"]].append(r)
    return by_q


def usable(samples: list[dict]) -> list[dict]:
    return [s for s in samples if not s.get("truncated") and s["answer"].strip() and s.get("language_ok", True)
            and s.get("judge_parsed", True)]


def classify(samples: list[dict]) -> tuple[str, float]:
    """The pair case for a question (see the module docstring) and its accuracy over usable samples."""
    ok = usable(samples)
    if len(ok) < max(2, len(samples) // 2):
        return "skip_unusable", 0.0
    labels = Counter(s["label"] for s in ok)
    acc = (labels["CORRECT"] + 0.5 * labels["PARTIAL"]) / len(ok)
    if not samples[0].get("reference"):
        if labels["FABRICATED"] + labels["WRONG"] + labels["PARTIAL"]:
            return "made_up", acc
        return ("style", acc) if labels["ABSTAIN_POOR"] else ("skip_already_good", acc)
    wrong = labels["WRONG"] + labels["FABRICATED"]
    if acc >= KNOWN:
        return ("known", acc) if labels["CORRECT"] else ("skip_no_correct", acc)
    if acc >= SHARPEN:
        return ("sharpen", acc) if wrong and labels["CORRECT"] else ("skip_no_contrast", acc)
    return ("unknown", acc) if (wrong + labels["PARTIAL"]) else ("skip_already_abstains", acc)


def candidates(samples: list[dict], case: str, n: int = 3) -> list[str]:
    """Chosen candidates from the policy's own samples, shortest first (fewer extra claims), duplicates removed."""
    want = "ABSTAIN_GOOD" if case in ABSTAIN_CASES else "CORRECT"
    texts = sorted({s["answer"] for s in usable(samples) if s["label"] == want}, key=len)
    return [t for t in texts if len(t) >= 2][:n]


def parse_check(text: str) -> tuple[bool, str]:
    m = re.search(r"\{.*\}", re.sub(r"^.*</think>", "", text, flags=re.S), re.S)
    try:
        out = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        out = {}
    return out.get("pass") is True, str(out.get("reason", ""))[:200]


def phase_curate(a) -> None:
    by_q = groups(load(a.judged))
    cases = {qid: classify(samples)[0] for qid, samples in by_q.items()}
    rng = random.Random(a.seed)
    from vllm import LLM, SamplingParams

    llm = LLM(a.writer, tensor_parallel_size=a.tp, max_model_len=8192, seed=a.seed, gpu_memory_utilization=0.88)
    tok = llm.get_tokenizer()

    def chat(p: str) -> str:
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)

    def check(items: list[tuple[dict, str, str]]) -> list[tuple[bool, str]]:
        """items: (first sample of the question, kind 'abstain' | 'answer', reply text)."""
        prompts = [chat(CHECK[kind].format(lang=J.LANG[q["language"]], question=q["question"],
                                           facts=J.facts_for(q), answer=text[:4000])) for q, kind, text in items]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=200))
        return [parse_check(o.outputs[0].text) for o in outs]

    def generate(prompts: list[str], seed: int) -> list[str]:
        outs = llm.generate([chat(p) for p in prompts], SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300,
                                                                        seed=seed))
        return [o.outputs[0].text.strip() for o in outs]

    # 1. the policy's own candidates
    items, owners = [], []
    for qid, case in cases.items():
        if case in ABSTAIN_CASES + ANSWER_CASES:
            kind = "abstain" if case in ABSTAIN_CASES else "answer"
            for text in candidates(by_q[qid], case):
                items.append((by_q[qid][0], kind, text))
                owners.append(qid)
    curated: dict[str, dict] = {}
    reasons: Counter = Counter()
    for qid, (q, kind, text), (ok, why) in zip(owners, items, check(items)):
        if ok and qid not in curated:
            curated[qid] = {"id": qid, "case": cases[qid], "chosen": text, "chosen_source": "own"}
        elif not ok:
            reasons[f"{kind}: {why[:60]}"] += 1
    print(f"own candidates: {len(items)} checked, {len(curated)} questions covered", flush=True)
    examples = defaultdict(list)
    for c in curated.values():
        if c["case"] in ABSTAIN_CASES:
            examples[by_q[c["id"]][0]["language"]].append(c["chosen"])
    # 2. written abstentions where no own one passed
    todo = [by_q[qid][0] for qid, case in cases.items() if case in ABSTAIN_CASES and qid not in curated]
    for attempt in range(a.attempts):
        if not todo:
            break
        prompts = []
        for q in todo:
            ex = rng.sample(examples[q["language"]], min(2, len(examples[q["language"]])))
            ex_text = ("\nFor style, here are replies the same assistant gave to other questions it could not answer "
                       "(do not copy their content):\n" + "\n".join(f"- {e}" for e in ex) + "\n") if ex else ""
            subject = f"“{q['subject']}”" if q["subject"] else "what the question asks about"
            prompts.append(WRITER_PROMPT.format(lang=J.LANG[q["language"]], question=q["question"], subject=subject,
                                                examples=ex_text))
        texts = generate(prompts, a.seed + attempt)
        verdicts = check([(q, "abstain", t) for q, t in zip(todo, texts)])
        rest = []
        for q, text, (ok, _) in zip(todo, texts, verdicts):
            if ok and 20 <= len(text) <= 700:
                curated[q["id"]] = {"id": q["id"], "case": cases[q["id"]], "chosen": text, "chosen_source": "written"}
            else:
                rest.append(q)
        print(f"write attempt {attempt}: {len(todo) - len(rest)} of {len(todo)} accepted", flush=True)
        todo = rest
    # 3. unnecessary refusals as the rejected side of known questions the policy never refused
    need = [by_q[qid][0] for qid, case in cases.items() if case == "known" and qid in curated
            and not any(s["label"].startswith("ABSTAIN") for s in usable(by_q[qid]))]
    for q, text in zip(need, generate([REFUSAL_PROMPT.format(lang=J.LANG[q["language"]], question=q["question"])
                                       for q in need], a.seed)):
        if 10 <= len(text) <= 600:
            curated[q["id"]]["refusal"] = text
    print(f"refusals written for {len(need)} known questions", flush=True)
    a.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in curated.values()))
    summary = {"curated": len(curated), "by_source": dict(Counter(f"{c['case']}|{c['chosen_source']}" for c in curated.values())),
               "write_failed": len(todo), "top_check_failures": dict(reasons.most_common(15))}
    a.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1, ensure_ascii=False))


def h(x: str) -> float:
    return int(hashlib.sha256(x.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def phase_pairs(a) -> None:
    rng = random.Random(a.seed)
    by_q = groups(load(a.judged))
    curated = {r["id"]: r for r in load(a.curated)}
    pairs: dict[str, list[dict]] = defaultdict(list)
    stats: Counter = Counter()
    knowledge = []
    for qid, samples in sorted(by_q.items()):
        q = samples[0]
        case, acc = classify(samples)
        knowledge.append({"id": qid, "language": q["language"], "band": q["band"], "qtype": q["qtype"],
                          "question": q["question"], "accuracy": round(acc, 3), "case": case,
                          "labels": dict(Counter(s["label"] for s in samples))})
        stats[f"{q['band']}|{case}"] += 1
        c = curated.get(qid)
        if case not in ABSTAIN_CASES + ANSWER_CASES:
            continue
        if not c:
            stats[f"{case}|no_chosen"] += 1
            continue
        ok = usable(samples)
        pick = lambda labels: [s["answer"] for s in ok if s["label"] in labels]  # noqa: E731
        if case == "style":
            bad = pick(("ABSTAIN_POOR",))
        elif case in ("made_up", "unknown"):
            bad = pick(INVENTED) or pick(("PARTIAL",))
        elif case == "sharpen":
            bad = pick(INVENTED)
        else:  # known: an unnecessary refusal first, then a wrong answer
            bad = pick(("ABSTAIN_GOOD", "ABSTAIN_POOR")) or ([c["refusal"]] if c.get("refusal") else []) or pick(INVENTED)
        bad = [b for b in bad if b != c["chosen"]]
        if not bad:
            stats[f"{case}|no_rejected"] += 1
            continue
        stats[f"{case}|chosen_{c['chosen_source']}"] += 1
        pairs[case].append({"id": f"natural-halu-{case}-{qid}", "prompt": [{"role": "user", "content": q["question"]}],
                            "chosen": [{"role": "assistant", "content": c["chosen"]}],
                            "rejected": [{"role": "assistant", "content": rng.choice(bad)}],
                            "source": f"oellm-natural-halu-v1:{case}", "chosen_source": c["chosen_source"],
                            "language": q["language"], "band": q["band"], "accuracy": round(acc, 3)})
    available = {k: len(v) for k, v in pairs.items()}
    # Mix: answering pairs (known + sharpen) are at least --known-share of the total (the over-refusal guard);
    # abstention pairs are capped by --max-abstain and by the answering pairs available.
    abstain = pairs["made_up"] + pairs["unknown"] + pairs["style"]
    answer = pairs["known"] + pairs["sharpen"]
    rng.shuffle(abstain)
    rng.shuffle(answer)
    abstain = abstain[: min(a.max_abstain, round(len(answer) * (1 - a.known_share) / a.known_share))]
    answer = answer[: max(round(len(abstain) * a.known_share / (1 - a.known_share)), 0)]
    rows = abstain + answer
    rng.shuffle(rows)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    split = {"train": [r for r in rows if h(r["id"]) >= 0.05], "validation": [r for r in rows if h(r["id"]) < 0.05]}
    for name, part in split.items():
        with gzip.open(a.out_dir / f"natural-halu-{name}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (a.out_dir / "knowledge.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in knowledge))
    manifest = {"kind": "oellm_natural_hallucination_pairs_v1", "judged": str(a.judged), "pairs": len(rows),
                "by_case": dict(Counter(r["source"].split(":")[1] for r in rows)),
                "chosen_source": dict(Counter(r["chosen_source"] for r in rows)), "available": available,
                "known_share_target": a.known_share, "thresholds": {"known": KNOWN, "sharpen": SHARPEN},
                "splits": {k: len(v) for k, v in split.items()},
                "by_language": dict(sorted(Counter(r["language"] for r in rows).items())),
                "stats": dict(sorted(stats.items()))}
    (a.out_dir / "pairs-manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "stats"}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["curate", "pairs"])
    ap.add_argument("--judged", type=Path, required=True)
    ap.add_argument("--writer", help="curate: checker and writer model (gemma-4-31B-it)")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--out", type=Path, help="curate: curated chosen answers (jsonl)")
    ap.add_argument("--curated", type=Path, help="pairs: output of curate")
    ap.add_argument("--out-dir", type=Path, help="pairs: staged DPO files, knowledge.jsonl and pairs-manifest.json")
    ap.add_argument("--known-share", type=float, default=0.35)
    ap.add_argument("--max-abstain", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20261116)
    a = ap.parse_args()
    {"curate": phase_curate, "pairs": phase_pairs}[a.phase](a)


if __name__ == "__main__":
    main()
