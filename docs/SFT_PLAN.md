# SFT plan: fix hallucination inside the main SFT run

**Goal:** the next Prelude SFT comes out of its single SFT run already answering what it knows and saying plainly
when it does not, so we do not need a separate warm-start SFT or hallucination DPO round afterwards. Target, on our
benchmark in default **and** thinking mode: fabrication on made-up subjects < 10 %, fewer wrong answers on
well-known people than today's SFT, lost answers < 5 %, "I don't know who X is" for a well-known person < 2 %,
language ≥ 95 %, no regression on the general evaluations.

Updated 2026-11-19. Research basis: [SFT_HALLUCINATION_DESIGN.md](SFT_HALLUCINATION_DESIGN.md). Overall plan:
[PLAN.md](PLAN.md).

## Why this can work in one SFT run

- **Our own evidence.** A 12-minute SFT on ~2,100 curated, model-specific hallucination rows took Prelude-9B-SFT-v2
  from 97.7 % to 3.6 % fabrication. The behaviour is learnable from SFT data alone. What went wrong (over-refusal,
  "I'm not sure who John Paul II is") came from how the data was built, and we know the fixes (below).
- **The research.** Hallucination is largely *taught* by SFT: answers about facts the model does not know (Gekhman et
  al., 2024) or knows only weakly (Ghosal et al., 2024), and teacher-written answers that carry new knowledge (FLAME,
  Lin et al., 2024), train the model to answer anyway. How the unfamiliar examples are labelled decides what the model
  does on unfamiliar questions (Kang et al., 2024). So the fix belongs in the SFT data: remove what teaches guessing,
  and label the unfamiliar part with the behaviour we want.
- **Why we needed DPO so far:** we added the behaviour *after* an SFT that had taught the opposite. Building it in from
  the start removes that conflict; DPO stays available as a polish step, not a requirement.

## The five changes to the SFT data

### 1. Probe the checkpoint we will fine-tune (the mid-train base)

Knowledge is model-specific, so the known / unknown split must come from the actual starting checkpoint
(`prelude-9b-midtrain-v1` or its successor), not from an SFT model.

- **Questions:** the existing pool (8,644 questions, 10 languages) plus the new well-known-people pool
  (`build_pool.py --extra-known`), extended to ~20K questions over 10 languages in phase 1 and the other OELLM languages
  in phase 2. Benchmark subjects excluded.
- **Sampling a base model:** few-shot prompts (5 question-answer examples in the same language, as in Gekhman et al.),
  8 samples, temperature 0.7, short answers. A base model gives short factual completions, which the judge grades
  more reliably than long chat answers.
- **Judging:** gemma-4-31B-it against Wikidata, with separate labels for *knows who the subject is* and *knows the
  detail asked*.
- **Categories:** known (≥ 6/8 right), partly known (knows who but not the detail, or 3–5/8), unknown (≤ 2/8),
  made up.
- **Cost:** ~20K × 8 short samples + judging ≈ 100 local-h.

### 2. Clean the general SFT mix (stop teaching guessing)

The v2 mix is 4.34M conversations, mostly teacher-written (INTELLECT-3 math, code, science, chat; AgentTrove; SWE).
Math, code and agent traces do not teach entity facts and stay unchanged. The risk is in the chat and IF parts
(`am_chat`, `am_if`, thinking-off chat; ~1.4M rows).

| Step | How | Cost |
|---|---|---|
| a. Find fact-seeking rows | a small classifier prompt (or Qwen3.5-27B) over the chat/IF prompts: does the answer depend on facts about specific people, works, places, organisations or events? | ~1.4M short classifications, ~40 local-h |
| b. Rate how well-known the entities are | extract the main entities, look up Wikipedia sitelinks (popularity proxy, Ghosal et al.) | CPU |
| c. Act on it | well-known entities: keep; long-tail entities (< ~10 sitelinks) with confident detailed claims: drop, or replace the answer with the base model's own probed knowledge (FLAME) or an honest "I'm not sure" | judge on the long-tail subset only |
| d. Check the estimate | probe a 2 % sample of the kept fact-seeking rows as in step 1; if > 10 % of their claims are unknown to the base, tighten the threshold | ~20 local-h |

Also fix the cheap things the old plan listed: stale cutoff claims ("as of my last update in 2023"), false capability
excuses ("I can't access databases"), identity claims of other models. Regex plus judge.

### 3. Add a knowledge-boundary set built from the probe

About **5,000 rows per language** (50K for 10 languages, ~1 % of the mix): enough not to be diluted in a 4.3M-row single
pass at lr 2e-5. Our warm start used 2,100 rows at lr 5e-6 for two epochs on a much smaller mix.

| Part | Share | Prompt | Target |
|---|---:|---|---|
| Known | 35 % | real subjects the base knows: open questions and details | a short correct answer from the model's own samples, checked against Wikidata |
| Partly known | 15 % | well-known subjects whose detail it gets wrong; open questions where it knows who but not the details | "X is …[what it knows, verified]…; I'm not sure about [detail]" |
| Unknown real | 20 % | obscure and mid-popularity subjects it gets wrong | a short "I'm not sure about X / that detail", no invented facts, one useful next step |
| Made up | 15 % | invented people (odd and plausible names), works, places, companies | "I don't recognise X", never "X does not exist", no guesses |
| False premise / underspecified | 10 % | CoCoNot subset (false presuppositions, universal unknowns, temporal, underspecified), translated | correct the premise or ask what is meant |
| Identity | 5 % | identity dataset v3.1 | who the model is, its cutoff, what it cannot know |

Rules learned from rounds 1–2 (all already in `train/build_pairs.py` or `build_sft_warmstart.py`):

- **Balance by question form:** every template (who is, tell me about, birth date, birthplace, author, director,
  capital, …) appears with known, partly known, unknown and made-up subjects in similar numbers. Well-known subjects are
  at least as common as obscure ones in open questions. *(Round 2 failed here.)*
- **Never make a well-known person an abstention target** for an open question when the model's samples identify them
  (the `identify` rule); details it does not know get a partial answer.
- **Do not ask for context about well-known subjects.**
- **Short, verifiable targets** (≤ 250 characters for answers, ≤ 600 for abstentions); every target passes the strict
  check (`CHECK` in `build_pairs.py`) and a Qwen3.5-27B audit (target ≥ 95 % acceptable).
- **Varied wording:** no opening phrase over 10 % of a language's abstentions; plausible made-up names, not only odd ones.
- **Native per language,** not translated from English (Feng et al., 2024), since knowledge differs per language.

### 4. Thinking mode, half of the set

The v2 SFT is mostly reasoning traces, and thinking mode lagged in every round (all our pairs were default mode). Half
of the knowledge-boundary rows get a reasoning trace that recalls what the model knows about the subject and decides:
"I know X is an Italian politician; I don't recall his birth date reliably, so I shouldn't give one." The final answer
follows the same rules. Traces are written by gemma-4-31B-it from the probe's labels and checked like the answers.

### 5. Where and how long

- Mix the knowledge-boundary set evenly through the single pass (no separate stage).
- Keep the run at one pass (the v2 recipe). Gekhman et al. show unknown examples are fitted late, so extra epochs on the
  factual part would raise hallucination; the cleaning in change 2 makes this less sensitive.

## Phases

| Phase | What | Cost (local-h) | Decision |
|---|---|---:|---|
| 0. Small-scale test (de-risk) | Rebuild the warm-start set on Prelude-SFT-v2 with the round-2 lessons (form balance, `identify`, partial answers, thinking rows) and the new well-known-people pool; train the 12-minute warm start; benchmark with `score.py --baseline` | ~60 | Go if fabrication < 10 %, lost answers < 5 %, denies known < 2 % in both modes. Otherwise fix the data recipe here, where a run costs minutes, not days |
| 1. Probe the mid-train base | Change 1 on 10 languages | ~100 | Category shares per language look sane (known share falls with popularity) |
| 2. Clean the general mix | Change 2 | ~60 | Estimated unknown-claim share in kept fact-seeking rows < 10 % |
| 3. Build the set | Changes 3 and 4 from the probe; audit | ~50 | Audit ≥ 95 % acceptable per part |
| 4. Hand over | Rows in the SFT team's `messages` format with a manifest (counts by part, language, mode; template hashes) | 0 | SFT team signs off on the format |
| 5. SFT run | The team's normal Prelude SFT with the cleaned mix + the set | as the normal run | – |
| 6. Measure | Benchmark (both modes, `--baseline` = previous SFT), identity eval, the general suite | ~30 | Release gates; DPO only if a gate fails |

Phases 1–3 can run while phase 0 is evaluated; the data recipe is frozen only after phase 0 passes. Total for our side:
~300 local-h, about the same as one warm-start + DPO round, but done once, before the SFT, for every later model.

## Ablation (cheap, worth having for the paper)

At phase 0 scale, train three variants on the same base: (a) the round-1 recipe, (b) + form balance and `identify`,
(c) + partial answers and thinking rows. This shows which change fixes lost answers and denies-known, at ~20 local-h each.

## Risks

| Risk | Mitigation |
|---|---|
| The set is diluted in a 4.3M-row single pass | 5K rows per language (~1 % of rows); phase 0 checks the behaviour; if the full SFT misses the gates, a short DPO polish from the SFT checkpoint (our pipeline) is the fallback |
| Probing a base model with few-shot prompts mislabels knowledge | compare categories on 500 questions probed both as base (few-shot) and as the current SFT (chat); expect broad agreement |
| Cleaning drops useful chat data | only long-tail fact-seeking rows are touched (expected a few % of chat/IF); report counts |
| Over-refusal returns in other forms | gates on lost answers and denies-known in both modes, plus CoCoNot contrast prompts and the 35 % known share |
| Judge blind spots (gemma both writes and grades) | Qwen3.5-27B audit of targets; hand-check 50 rows per language in phase 3 |

## What we need from the SFT team

- The starting checkpoint for the probe (mid-train base) and the planned mix manifest.
- Agreement on the cleaning rules for the chat/IF parts.
- A slot to add ~50K rows and to drop the flagged ones; the evaluation hooks (our benchmark as a standard eval).
