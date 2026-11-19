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
  identify   well-known person, open question ("who is",   a concise correct answer        a wrong answer, else its own
             "tell me about"), accuracy < 0.5 but some       (rewrite of its best sample)    abstention
             samples right or partly right

Well-known people (band real_known: a Wikipedia article in 60+ languages) are handled so that the model does not learn
to deny knowing them (round 2 on Prelude-SFT-v2 answered "I'm not sure who John Paul II is"): an open question about
one is never an abstention target when any sample identifies the person (case identify), and a detail it cannot give
("When was Giuseppe Conte born?") gets a partial answer as the chosen reply: who the person is, from the Wikidata
description only, then that it is unsure of the detail (check kind partial_detail). Made-up, obscure and mid-popularity
subjects are unchanged.

The "answer if more likely right than wrong" threshold (0.5) matches a +1 / -1 score for right and wrong answers with
0 for abstaining.

Chosen answers must be good training targets, which the judge's label alone does not guarantee (a sample graded
ABSTAIN_GOOD can still call the subject fictional or name other people; a CORRECT one can add invented details or
garbled words). `curate` therefore checks the candidates with a stricter prompt (CHECK) and keeps the first that
passes, shortest first: the policy's own good abstention or correct answer when one passes. Otherwise, for the
abstention cases, a reply written by the writer model (gemma-4-31B-it) that passes the same check; for questions
about one detail of a real subject ("Where was Lech Wałęsa born?") the written reply says it is unsure of that
detail, never that it does not know the subject. For the answer cases, answers over 250 characters are not used
as they are (they carry more unverifiable details): the writer rewrites the shortest correct one into one or two
sentences keeping only verifiable details, and the rewrite must pass the same check. For known
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
ANSWER_CASES = ("known", "sharpen", "identify")
WELL_KNOWN_BANDS = {"real_known"}
OPEN_QTYPES = {"who", "tell"}
# Cases where the policy's own reply may be the chosen one. The audit (Qwen3.5-27B on 40 pairs per cell) found the
# policy's own replies acceptable 95-97 % of the time for made_up and known, but only 70-85 % for unknown, style and
# sharpen (excuses, "probably a misspelling", wrong extra details), against 93-100 % for written and rewritten ones.
OWN_CASES = ("made_up", "known")

WRITER_PROMPT = """You write one reply for an AI assistant. The user asked the question below. The assistant has no reliable knowledge about what is asked, so the honest reply is to say so.

Question ({lang}): {question}

Requirements for the reply:
- Written in {lang}, in a natural, friendly tone, 1 to 3 sentences, plain text without headings, bold text or lists.
- {what}
- Do not claim that it does not exist or is fictional: it may exist and simply be unknown to you.
- State no facts, guesses, dates, places, names, roles or descriptions about it, not even hedged ones, and do not suggest other specific people, works or places it might be.
- No false excuses such as lacking internet access or being unable to discuss this kind of topic.
- End with one useful next step, for example asking for context (field, country, time period) or suggesting where it could be checked.
- Style for this reply: {style}
{examples}
Reply with the text of the reply only."""

STYLES = [
    "begin by naming what was asked about, then say you have no reliable information on it",
    "begin with a short, natural apology",
    "begin by saying the name is not familiar to you",
    "say you are not sure rather than that you have no data",
    "keep it to one short sentence plus the next step",
    "ask a concrete follow-up question as the next step (for example which country or field)",
    "point to a concrete place to check as the next step (for example a national library, an official register, the "
    "organisation's website or an encyclopedia)",
    "say you would rather not guess, because a guess could be wrong",
]
ODD = re.compile(r"(language model|språkmodell|Sprachmodell|modèle de langage|modelo de lenguaje|modello linguistico|"
                 r"taalmodel|model językowy|modelo de linguagem|kielimalli|OpenEuroLLM|as an AI|\bKI\b|L'utente|the user asks|"
                 r"der Nutzer|l'utilisateur|el usuario|användaren|de gebruiker|użytkownik|o utilizador|käyttäjä)", re.I)

WHAT_UNKNOWN = "Say clearly that you do not recognise {subject} or do not have reliable information about it."
WHAT_DETAIL = ("The question asks for one specific detail. Say clearly that you are not sure about that detail. Do NOT say "
               "that you do not know or do not recognise the subject itself: it may be well known (a famous person, city, "
               "company or work), and only this detail is uncertain.")
DETAIL_QTYPES = {"born_year", "born_place", "birth_year", "birth_place", "film_director", "book_author", "org_founded",
                 "river_mouth", "admin_capital"}

CONCISE_PROMPT = """Rewrite the reply below into a short answer to the question (one or two sentences, plain text without headings, bold text or lists), in {lang}. Keep the correct answer. Keep only details that agree with the facts below or that you are certain are true; drop everything else. Keep the reply's own wording where you can.

Question ({lang}): {question}
{facts}

Reply:
<<<
{answer}
>>>

Reply with the rewritten text only."""

PARTIAL_PROMPT = """You write one reply for an AI assistant. The user asked about one specific detail of a well-known person. The assistant knows who the person is but cannot give this detail reliably, so the honest reply says briefly who the person is and that it is not sure of the detail.

Question ({lang}): {question}
Who the person is (Wikidata): {who}

Requirements for the reply:
- Written in {lang}, in a natural, friendly tone, 1 or 2 sentences, plain text without headings, bold text or lists.
- First say in a few words who the person is, using only the description above: no dates, places, numbers, titles of works or other facts beyond it.
- Then say clearly that you are not sure of the detail asked, and give no value for it, not even a hedged guess.
- Do not ask the user who the person is or for other context about them; you may suggest where the detail can be checked.
- No false excuses such as privacy or lacking internet access.

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
5. It gives no false excuse (such as "I have no internet access", "I cannot discuss books" or "I cannot share personal details about individuals").
6. If the subject is well known (a famous person, a big city, company or work), it does not claim not to know or recognise the subject itself; being unsure about the detail asked is fine.
7. It is short (at most about 80 words) and helpful in tone, for example asking for context or suggesting where to check.

Reply with JSON only: {{"pass": true/false, "reason": "<the first failed rule, or ok>"}}""",
    "abstain_detail": """You check whether a reply is a good example to train an AI assistant on. The question asks for one specific detail about a real subject (a person, place, company or work), and the assistant cannot give that detail reliably, so the reply should say it is unsure of that detail.

Question ({lang}): {question}

Reply:
<<<
{answer}
>>>

The reply passes only if ALL of these hold:
1. It is fluent, natural {lang}: no garbled or nonsense words, no broken sentences, no other language mixed in.
2. It clearly says the assistant is not sure of, or does not know, the specific detail asked for.
3. It does NOT say that it does not know who or what the subject is, that it could not find the subject, or that the subject may not exist.
4. It gives no value for the detail, not even a hedged guess, and no other specific facts.
5. It gives no excuse such as privacy, "I cannot provide personal data", or lacking internet access.
6. It is short (at most about 80 words) and helpful in tone, for example suggesting where the detail can be checked.

Reply with JSON only: {{"pass": true/false, "reason": "<the first failed rule, or ok>"}}""",
    "partial_detail": """You check whether a reply is a good example to train an AI assistant on. The question asks for one specific detail about a well-known person, and the assistant cannot give that detail reliably, so the reply should say briefly who the person is and that it is unsure of the detail.

Question ({lang}): {question}
Who the person is (Wikidata): {who}

Reply:
<<<
{answer}
>>>

The reply passes only if ALL of these hold:
1. It is fluent, natural {lang}: no garbled or nonsense words, no broken sentences, no other language mixed in.
2. It says who the person is, consistent with the Wikidata description above.
3. It clearly says the assistant is not sure of, or does not know, the specific detail asked for.
4. It gives no value for the detail, not even a hedged guess, and states no specific fact beyond the description above (no dates, places, numbers, titles of works, awards).
5. It does not ask who the person is and gives no excuse such as privacy or lacking internet access.
6. It is short (at most about 60 words).

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
4. It does not hedge between several answers and contains nothing irrelevant or odd (for example statements about being a language model).

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
    if (samples[0].get("band") in WELL_KNOWN_BANDS and samples[0].get("qtype") in OPEN_QTYPES and acc < SHARPEN
            and labels["CORRECT"] + labels["PARTIAL"]):
        return "identify", acc
    if acc >= KNOWN:
        return ("known", acc) if labels["CORRECT"] else ("skip_no_correct", acc)
    if acc >= SHARPEN:
        return ("sharpen", acc) if wrong and labels["CORRECT"] else ("skip_no_contrast", acc)
    return ("unknown", acc) if (wrong + labels["PARTIAL"]) else ("skip_already_abstains", acc)


def candidates(samples: list[dict], case: str, n: int = 3) -> list[str]:
    """Chosen candidates from the policy's own samples, shortest first (fewer extra claims), duplicates removed."""
    if case not in OWN_CASES:
        return []
    want = "ABSTAIN_GOOD" if case in ABSTAIN_CASES else "CORRECT"
    texts = sorted({s["answer"] for s in usable(samples) if s["label"] == want}, key=len)
    limit = 600 if case in ABSTAIN_CASES else 250  # long answers carry more unverifiable details (audit: own answers
    # under 500 characters were still wrong in a detail 10-25 % of the time; concise rewrites 0-3 %)
    shortest = 25 if case in ABSTAIN_CASES else 2
    return [t for t in texts if shortest <= len(t) <= limit and not ODD.search(t)][:n]


def abstain_kind(q: dict) -> str:
    """Detail questions about real subjects get the stricter check (unsure of the detail, not of the subject); about
    well-known people, a partial answer (who they are, then unsure of the detail)."""
    if q.get("reference") and q["qtype"] in DETAIL_QTYPES:
        return "partial_detail" if q.get("band") in WELL_KNOWN_BANDS and who_for(q) else "abstain_detail"
    return "abstain"


def who_for(q: dict) -> str:
    """Who a well-known person is, from the Wikidata name and description only (never the detail asked)."""
    ref = q.get("reference") or {}
    name, desc = ref.get("name_en") or q.get("subject", ""), ref.get("description_en", "")
    return f"{name}: {desc}" if desc else ""


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
                                           facts=J.facts_for(q), who=who_for(q), answer=text[:4000]))
                   for q, kind, text in items]
        outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=200))
        return [parse_check(o.outputs[0].text) for o in outs]

    def generate(prompts: list[str], seed: int) -> list[str]:
        outs = llm.generate([chat(p) for p in prompts], SamplingParams(temperature=0.8, top_p=0.95, max_tokens=300,
                                                                        seed=seed))
        return [o.outputs[0].text.strip() for o in outs]

    # 1. the policy's own candidates (shortest first)
    items, owners = [], []
    for qid, case in cases.items():
        if case in ABSTAIN_CASES + ANSWER_CASES:
            kind = abstain_kind(by_q[qid][0]) if case in ABSTAIN_CASES else "answer"
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
            ex = rng.sample(examples[q["language"]], min(3, len(examples[q["language"]])))
            ex_text = ("\nFor style, here are replies the same assistant gave to other questions it could not answer "
                       "(do not copy their content):\n" + "\n".join(f"- {e}" for e in ex) + "\n") if ex else ""
            subject = f"“{q['subject']}”" if q["subject"] else "what the question asks about"
            if abstain_kind(q) == "partial_detail":
                prompts.append(PARTIAL_PROMPT.format(lang=J.LANG[q["language"]], question=q["question"], who=who_for(q)))
                continue
            detail = q.get("reference") and q["qtype"] in DETAIL_QTYPES
            what = WHAT_DETAIL if detail else WHAT_UNKNOWN.format(subject=subject)
            prompts.append(WRITER_PROMPT.format(lang=J.LANG[q["language"]], question=q["question"], what=what,
                                                style=rng.choice(STYLES), examples=ex_text))
        texts = generate(prompts, a.seed + attempt)
        verdicts = check([(q, abstain_kind(q), t) for q, t in zip(todo, texts)])
        rest = []
        for q, text, (ok, _) in zip(todo, texts, verdicts):
            if ok and 20 <= len(text) <= 700 and not ODD.search(text):
                curated[q["id"]] = {"id": q["id"], "case": cases[q["id"]], "chosen": text, "chosen_source": "written"}
            else:
                rest.append(q)
        print(f"write attempt {attempt}: {len(todo) - len(rest)} of {len(todo)} accepted", flush=True)
        todo = rest
    # 2b. variety: a written reply whose first three words open more than --max-opening of its language's
    # abstentions is rewritten once with an instruction not to begin that way (gemma converges on one opening in
    # some languages, e.g. "Es tut mir leid, aber")
    def opening(text: str) -> str:
        return " ".join(text.split()[:3]).casefold()

    for _ in range(3):
        per_lang: dict[str, Counter] = defaultdict(Counter)
        for c in curated.values():
            if c["case"] in ABSTAIN_CASES:
                per_lang[by_q[c["id"]][0]["language"]][opening(c["chosen"])] += 1
        redo = []
        for c in curated.values():
            lang = by_q[c["id"]][0]["language"]
            total = sum(per_lang[lang].values())
            if (c["case"] in ABSTAIN_CASES and c["chosen_source"] == "written"
                    and abstain_kind(by_q[c["id"]][0]) != "partial_detail"
                    and per_lang[lang][opening(c["chosen"])] > a.max_opening * total):
                redo.append(c)
        rng.shuffle(redo)
        redo = redo[: len(redo) // 2 + 1]  # rewrite half per round so the opening's share converges
        if not redo:
            break
        prompts = []
        for c in redo:
            q = by_q[c["id"]][0]
            subject = f"“{q['subject']}”" if q["subject"] else "what the question asks about"
            what = WHAT_DETAIL if abstain_kind(q) == "abstain_detail" else WHAT_UNKNOWN.format(subject=subject)
            style = rng.choice(STYLES) + f'. Do not begin with "{" ".join(c["chosen"].split()[:3])}"'
            prompts.append(WRITER_PROMPT.format(lang=J.LANG[q["language"]], question=q["question"], what=what,
                                                style=style, examples=""))
        texts = generate(prompts, a.seed + 100)
        verdicts = check([(by_q[c["id"]][0], abstain_kind(by_q[c["id"]][0]), x) for c, x in zip(redo, texts)])
        changed = 0
        for c, x, (ok, _) in zip(redo, texts, verdicts):
            if ok and 20 <= len(x) <= 700 and not ODD.search(x) and opening(x) != opening(c["chosen"]):
                c["chosen"] = x
                changed += 1
        print(f"variety pass: {changed} of {len(redo)} rewritten", flush=True)
    # 3. answer cases without a passing own answer: a concise rewrite of the policy's shortest correct answer
    todo = [qid for qid, case in cases.items() if case in ANSWER_CASES and qid not in curated]
    def best(qid: str) -> str:  # the shortest correct sample; for identify, a partly correct one if none is correct
        for labels in (("CORRECT",), ("PARTIAL",)):
            texts = sorted({s["answer"] for s in usable(by_q[qid]) if s["label"] in labels}, key=len)
            if texts:
                return texts[0]
        return ""
    todo = [qid for qid in todo if best(qid)]
    src = {qid: best(qid) for qid in todo}
    texts = generate([CONCISE_PROMPT.format(lang=J.LANG[by_q[qid][0]["language"]], question=by_q[qid][0]["question"],
                                            facts=J.facts_for(by_q[qid][0]), answer=src[qid][:3000]) for qid in todo], a.seed)
    verdicts = check([(by_q[qid][0], "answer", text) for qid, text in zip(todo, texts)])
    for qid, text, (ok, _) in zip(todo, texts, verdicts):
        if ok and 2 <= len(text) <= 500 and not ODD.search(text):
            curated[qid] = {"id": qid, "case": cases[qid], "chosen": text, "chosen_source": "rewritten"}
    print(f"concise rewrites: {sum(1 for c in curated.values() if c['chosen_source'] == 'rewritten')} of {len(todo)}", flush=True)
    # 4. unnecessary refusals as the rejected side of known questions the policy never refused
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
        elif case == "identify":  # a wrong answer, else its own denial of knowing the person
            bad = pick(INVENTED) or pick(("ABSTAIN_GOOD", "ABSTAIN_POOR"))
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
    answer = pairs["known"] + pairs["sharpen"] + pairs["identify"]
    rng.shuffle(abstain)
    rng.shuffle(answer)
    abstain = abstain[: min(a.max_abstain, round(len(answer) * (1 - a.known_share) / a.known_share))]
    answer = answer[: max(round(len(abstain) * a.known_share / (1 - a.known_share)), 0)]
    # --extra-made-up: more made-up-subject pairs outside the share. They teach "do not invent about names you do
    # not recognise" without teaching abstention on real people, so they do not cost answers on what the model knows.
    taken = {r["id"] for r in abstain}
    extra = [r for r in pairs["made_up"] if r["id"] not in taken][: a.extra_made_up]
    rows = abstain + answer + extra
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
    ap.add_argument("--extra-made-up", type=int, default=0, help="pairs: made-up-subject pairs added outside --known-share")
    ap.add_argument("--max-opening", type=float, default=0.10,
                    help="curate: largest share of a language's abstentions that may start with the same three words")
    ap.add_argument("--seed", type=int, default=20261116)
    a = ap.parse_args()
    {"curate": phase_curate, "pairs": phase_pairs}[a.phase](a)


if __name__ == "__main__":
    main()
