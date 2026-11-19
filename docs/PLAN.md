# Hallucination plan: stop the model from making things up

**Goal:** the model does not present invented facts. When it knows, it answers; when it does not, it says so plainly,
in the user's language. Not refusing what it knows is a constraint on that goal, not the goal itself.

Updated 2026-11-19. Details: [SFT_HALLUCINATION_DESIGN.md](SFT_HALLUCINATION_DESIGN.md) (research and SFT data design),
[`paper/`](../paper) (write-up), [`results/round2-well-known-people.md`](../results/round2-well-known-people.md).

## Where we are

| Model | Fabrication on made-up subjects | Wrong on well-known people | Denies knowing a well-known person | Lost answers | Status |
|---|---:|---:|---:|---:|---|
| Prelude-9B-SFT-v2 (start) | 97.7 % | 43.8 % | 0 % | – | baseline |
| + DPO round 1 only | 92.7 % | 38.8 % | 0 % | 0 % | DPO alone barely moves a model that never abstains |
| + SFT warm start | 3.6 % | 13.8 % | 0 % | 16.3 % | SFT moves the behaviour |
| + SFT warm start + DPO round 2 | **1.8 %** | **7.5 %** | 12.5 % | 25.6 % | [released](https://huggingface.co/birgermoell/prelude-9b-sft-v2-hallucination-r2) as a research checkpoint, with [its data](https://huggingface.co/datasets/birgermoell/prelude-9b-sft-v2-hallucination-r2-data) |
| RL merge (start) | 63.6 % | 28.8 % | 0 % | – | baseline |
| RL merge + DPO round 1 | 38.2 % | 23.8 % | 0 % | 3.6 % | works better on a model that already abstains sometimes |
| RL merge + SFT warm start | 23.6 % | 1.2 % | 0 % | 17.9 % | round 2 running |

Default mode, 500-question benchmark in 10 languages. *Lost answers*: well-known people the untrained model answered
(partly) right and the trained one declines.

## What we learned

1. **Fabrication can be fixed.** SFT on curated, model-specific abstentions followed by on-policy DPO takes a model from
   inventing answers 98 % of the time to 2 %, in ten languages, without leaving the question's language.
2. **SFT first, then DPO.** DPO alone cannot teach a behaviour the model never produces (19 % of chosen answers were its
   own). After an SFT warm start, DPO is truly on-policy.
3. **Abstention generalises by question form if the data lets it.** Our pool had 750 open questions about obscure
   people (target "I'm not sure") against 177 about well-known ones, and round 2 taught "I'm not sure who John Paul II
   is". Every question template needs known and unknown subjects in similar numbers.
4. **Knowing who someone is differs from knowing a detail.** A strict judge marked mostly-right biographies wrong for one
   detail, which turned "Tell me about Angela Merkel" into an abstention target. Fixed in `train/build_pairs.py`
   (`identify` case, partial answers for details).
5. **Over-refusal was the wrong gate.** Of 31 well-known people the round-2 model declines, 18 had been answered wrongly
   before (calibrated), 8 were birth details it knew, 5 were open questions it knew (the real failure). The new gates
   are *lost answers* and *denies known*.
6. **Format-based abstention (`ANSWER: UNKNOWN`) does not transfer** to natural questions.
7. **Thinking mode lags**: all pairs so far were default mode.

## The plan

### Phase A: fix the current models (now, ~250 local-h)

| # | Step | Cost (local-h) | Done when |
|---|---|---|---|
| A1 | Pair builder fixes: `identify` case, partial answers for details of well-known people, no requests for context about them | 0 | **done** (`c608cb6`) |
| A2 | New metrics in `score.py`: `known_wrong`, `denies_known`, `lost_answers` (`--baseline`) | 0 | **done** (`8cf2842`) |
| A3 | RL merge round 2: warm start (answers 50 % of the hallucination rows), re-sample half B with the A1 fixes, audit, DPO | ~150 | running; pairs checked before DPO |
| A4 | Pool of ~50 well-known people per language, open questions only (`build_pool.py --extra-known`) | 0 (Wikidata) | building |
| A5 | Prelude-SFT-v2 round 3: small on-policy DPO from the released model on A4 + the half-B pool with the A1 fixes, plus ~300 made-up / unknown pairs so abstention holds | ~100 | gates below; replaces the released checkpoint if it passes |
| A6 | General evaluations (math, code, IFEval, tools, long context) for the best checkpoint | ~20 | no regression |

### Phase B: build it into the next Prelude SFT (the lasting fix)

From the research (Gekhman et al. 2024; Ghosal et al. 2024; Kang et al. 2024; FLAME 2024; R-Tuning; Cheng et al. 2024)
and our rounds; full design in [SFT_HALLUCINATION_DESIGN.md](SFT_HALLUCINATION_DESIGN.md).

| # | Step | Why |
|---|---|---|
| B1 | Probe the mid-train base: 8 samples per fact-seeking prompt, per language; sort into known / partly known / unknown / made up | the split must be specific to the checkpoint being fine-tuned |
| B2 | Clean the general SFT mix: for fact-seeking rows, drop or replace teacher answers that teach facts the model does not know | fine-tuning on unknown or less-known facts raises hallucination |
| B3 | Knowledge-boundary set, ~1,000 rows per language: known 35 %, partly known 15 %, unknown 20 %, made up 15 %, false premise 10 %, identity 5 % | unfamiliar examples set the default; label them with the behaviour we want |
| B4 | Balance by question form and name style; short, verifiable answers; 1–2 epochs | so abstention depends on knowledge, not on the template |
| B5 | Thinking-mode rows: the reasoning recalls what the model knows, then decides | thinking mode lagged in every round |
| B6 | On-policy DPO after SFT (Phase A pipeline) | SFT then preference training works best |

Cost: the probe is ~50 local-h per 9K prompts; cleaning touches only the fact-seeking subset; the SFT run is unchanged.

### Phase C: RL with a judge reward (if DPO plateaus)

Keep the exact `ANSWER:` rows (cheap, verifiable) and add a judge-graded natural-question slice:

| Outcome | Reward |
|---|---|
| Correct answer | +1 |
| Partial answer: what it knows, unsure of the detail | +0.5 |
| Good "I don't know" on a subject it cannot know | +0.5 |
| Unhelpful refusal or false excuse | 0 |
| Declining a question it knows, or "I don't know who X is" for a well-known person | −0.5 |
| Fabrication or a wrong answer | −1 |

This rewards abstention over guessing (Kalai et al. 2025) while penalising lost knowledge. With a search tool, the
policy should look up obscure facts before stating them and answer in the user's language. ~500–700 local-h.

### Phase D: long-form factual precision (later)

Biographies and summaries: state only what it is confident of. Shorter answers carry fewer invented details (own
answers under 500 characters were still wrong in a detail 10–25 % of the time; concise rewrites 0–3 %).

## Release gates

Benchmark, closed-book, default **and** thinking mode, `score.py --baseline <untrained model>`:

| Gate | Target |
|---|---|
| Fabrication on made-up subjects | < 10 % |
| Wrong answers on well-known people (`known_wrong`) | below the untrained model |
| Lost answers | < 5 % |
| Denies knowing a well-known person | < 2 % |
| Answers in the question's language | ≥ 95 % |
| General evaluations | no regression |

## What not to do

- More `ANSWER: UNKNOWN`-only training: it does not transfer to natural questions.
- DPO alone on a model that never abstains.
- Abstention data whose question forms or name styles predict the label.
- Long, detail-heavy targets for fact-seeking prompts.
- Judging success by over-refusal alone: it counts calibrated abstentions as failures.

## Budget

Our budget is ~50K local-h (1 GPU-h = 8 local-h). Spent on this project so far: roughly 700 local-h by our estimate
(pair runs, five DPO runs including two that failed, two SFT warm starts, benchmarks). Phase A remaining: ~270. Phase B adds the probe and cleaning to the
next SFT run. Phase C only if needed.
