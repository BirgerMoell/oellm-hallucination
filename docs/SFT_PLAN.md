# SFT plan: reduce hallucination inside the main SFT run

**Goal:** the next Prelude SFT answers what it knows, gives a partial answer when it knows part, and says "I'm not
sure" when it does not, without a separate warm-start or DPO round. Whether SFT alone is enough is tested in step 6;
DPO stays the fallback.

Updated 2026-11-19. Background: [SFT_HALLUCINATION_DESIGN.md](SFT_HALLUCINATION_DESIGN.md) (research),
[PLAN.md](PLAN.md) (project), [round-2 analysis](../results/round2-well-known-people.md) (failures we must avoid).

## Release gates (both modes, locked test set, against the previous SFT and an unchanged-mix control)

| Metric | Gate |
|---|---|
| Fabrication on made-up subjects | < 10 % |
| Wrong requested facts on real subjects | lower than the previous SFT |
| Lost answers (baseline answered the requested fact right, candidate declines) | < 5 % |
| "I don't know who X is" when the baseline reliably identifies X | < 2 % |
| Answers in the question's language | ≥ 95 % |
| General evals (math, code, IFEval, science, tools, identity) | within margins agreed with the SFT team before step 6 |

## The recipe in four changes

1. **Probe** the mid-train base to learn what *it* knows, per question and language.
2. **Clean** the general mix: remove or shorten closed-book answers that teach facts the base does not know.
3. **Add a boundary set** of ~50K conversations whose targets follow the probe: answer / partial / "I'm not sure" /
   correct the premise, balanced by question form, in both modes.
4. **Train once**, mixed through the run, at a measured supervised-token share, selecting the checkpoint on a
   development set.

## Step 1. Measurement first (week 1, ~40 local-h)

- **Splits by entity.** Assign every Wikidata QID (and every synthetic name) to train / dev / test by hash, before
  any probing. All languages and question forms of an entity go to the same split.
- **Locked test set (new):** 3,000 questions, 300 per language: 100 made-up, 100 real people with ≥ 60 Wikipedia
  editions (open + detail questions), 50 obscure, 50 false premise / grounded. 1,000 well-known-person items give ~150
  eligible cases per language group for the < 2 % and < 5 % gates. Built with `benchmark/build_questions.py`, a new
  seed, excluded from the pool. Never used for tuning.
- **Dev set:** the current 500-question benchmark (already used for tuning).
- **Scorer:** add a requested-fact label (right / wrong / not given) next to the existing labels, so "lost answers"
  and "wrong requested facts" do not depend on `PARTIAL`; add 95 % intervals (bootstrap by entity).
- **Baselines:** run the previous Prelude SFT on dev and test now, in both modes.

## Step 2. Probe the mid-train base (week 2, ~130 local-h)

- **Questions:** 20K unique, 2K per language: the current pool (8,644) + `build_pool.py --extra-known` (500) + new
  questions for the rest, same bands; only train-split entities.
- **Prompt:** native-language few-shot completion, 5 verified Q/A examples from train-split entities, e.g.
  `Q: Where was Olof Palme born?\nA: Stockholm\n\nQ: …\nA:`. For open questions ("Who is X?") the few-shot answers are
  one-sentence identifications.
- **Samples:** 8 at temperature 0.7 + 1 greedy, max 48 tokens (open questions 96).
- **Judging (gemma-4-31B-it against Wikidata, aliases normalised first):** per sample, `requested_fact` ∈ {right,
  wrong, none} and, for people, `identifies_subject` ∈ {yes, no}.
- **Routing:**

  | Probe result | Category |
  |---|---|
  | requested fact right in ≥ 6/8 and greedy right | known |
  | identifies subject in ≥ 6/8, requested fact right in ≤ 2/8 | partly known |
  | 3–5/8 right | unstable: re-probe with a second prompt form; still 3–5 → drop |
  | ≤ 2/8 right, does not identify | unknown |
  | made-up subject | made up |

- **Output:** `probe.jsonl`, one row per question: checkpoint, QID, language, form, all samples and labels, category,
  the claims the model recovered.
- **Check:** 500 questions probed again with a fresh seed; category agreement ≥ 85 %, else revise prompts.

## Step 3. Clean the general mix (week 2–3, ~230 local-h)

Scope: the chat/IF part of the v2 mix (`am_chat`, `am_if`, thinking-off chat: ~1.4M rows). Math, code, science and
agent traces: audit a 2,000-row sample per source; clean only if the audit finds > 2 % rows with unsupported entity
facts.

1. **Classify** prompt + answer (first 512 tokens) with a ≤ 9B model into `closed_book_fact`, `grounded` (answer uses
   text or tool output in the conversation), `creative`, `other`. Validate on 2,000 rows against Qwen3.5-27B; recall on
   `closed_book_fact` ≥ 90 %. ~40 local-h.
2. **Check claims** in `closed_book_fact` rows (expected 5–10 %, 70–140K rows): gemma-4-31B-it extracts the specific
   claims (names, dates, places, numbers, titles) and marks each supported / contradicted / unverifiable against
   Wikidata or Wikipedia. ~180 local-h.
3. **Act:**

   | Row | Action |
   |---|---|
   | all claims supported | keep |
   | some claims contradicted or unverifiable, the core answer supported | rewrite to the supported claims (as `CONCISE_PROMPT` in `build_pairs.py`) |
   | core answer contradicted | drop |
   | `grounded`, `creative`, `other` | keep (fix only stale cutoff / identity / false-excuse strings by regex) |

4. **Report:** kept / rewritten / dropped per source and language; a 500-row human-reviewed sample of decisions.

## Step 4. Build the boundary set (week 3, ~80 local-h)

~50K conversations from the probe, train-split entities only; 5K per language; half default mode, half thinking mode.

| Part | Per language | Prompt | Target (example) |
|---|---:|---|---|
| Known | 1,500 | known real subjects, all forms | *"Olof Palme was born in Stockholm."* (the model's own verified answer, ≤ 2 sentences) |
| Partly known | 1,000 | detail questions where it identifies the subject but not the detail | *"Giuseppe Conte is an Italian politician who was prime minister; I'm not sure of his exact birth date."* Only claims the probe shows the model recovered and Wikidata supports |
| Unknown | 750 | unknown real subjects | *"I'm not sure who Karl Möllinger is. Could you tell me which field he works in?"* |
| Made up | 500 | invented people (half plausible names), works, places, companies | *"I don't recognise a novel called 'Melnordra Galsar'."* Never "it does not exist" |
| False premise | 500 | CoCoNot false presuppositions, translated, + our own | correct the premise |
| Grounded | 750 | a passage with the answer, or the same passage with the answer removed | answer from the passage / say the passage does not give it |

Hard rules (checked by script before handover):

- **Form balance:** for each question form, known + partly known ≥ unknown + made up. Open questions about known people
  ≥ open questions about unknown people. *(Round 2 broke this: 750 vs 177.)*
- **No "I don't know who X is"** when the probe says the model identifies X.
- **Lengths:** answers ≤ 60 words, abstentions ≤ 50 words; no opening phrase > 10 % of a language's abstentions; no
  context request about known subjects.
- **Thinking rows:** a ≤ 60-word trace using only probe-recovered claims ("I know Conte is an Italian prime minister;
  I don't recall his birth date, so I won't guess."), then the same final answer. No reference facts in traces.
- **Audit:** Qwen3.5-27B on 100 rows per part and language (6,000), blind to category; ≥ 95 % acceptable per cell,
  else fix the generator and rebuild that cell. 50 rows per language reviewed by a native speaker.

## Step 5. Main-mix transfer pilot (week 3–4, ~1,500 local-h; needs approval)

Same mid-train base, a 0.2B-token stratified sample of the full mix (same source and language shares), same LR
schedule shape, on Leonardo (2 nodes):

| Arm | Mix |
|---|---|
| A | unchanged mix (control) |
| B | cleaned mix |
| C | cleaned mix + boundary set at 0.5 % of supervised tokens |
| D | cleaned mix + boundary set at 2 % of supervised tokens |
| E | best of C/D + known-answer replay (the known part repeated ×2) |

~300 local-h per arm (0.2B tokens at the ~1,400 tokens/s per GPU we measured for 9B training on Leonardo). Evaluate every 20 % of the run on dev (both modes) and a 500-row general-eval subset. Pick the
arm and the token share by the gates on dev; if no arm passes, fix the data at step 4 scale, not the main run.
Rows are counted after templating, masking and packing: report supervised tokens per part, mode and language.

## Step 6. Hand over and train (week 4)

- **Deliverable:** `boundary-sft/{train,dev}.jsonl.gz` in `messages` format, the cleaned-mix row list (keep / rewrite /
  drop with reasons) and `manifest.json` (counts and supervised tokens by part / language / mode / source, entity
  split hashes, checkpoint / template / judge versions, audit results, licences).
- **The SFT team** trains the main run with the chosen token share, saving every 10 % of the pass.
- **We evaluate** each saved checkpoint on dev (both modes) and the general suite; the team picks the checkpoint;
  then one run on the locked test set against the gates.
- **If a gate fails:** read the failure slices; run our on-policy DPO (`train/run_slurm.sbatch`) from that checkpoint
  as the fallback.

## Code to write

| File | Change | Effort |
|---|---|---|
| `train/probe_base.py` (new) | few-shot completion probing, greedy + 8 samples, `requested_fact` / `identifies_subject` labels, re-probe | 1 day |
| `benchmark/judge.py`, `score.py` | requested-fact label, eligibility from the baseline's requested fact, bootstrap intervals | 1 day |
| `train/clean_mix.py` (new) | classify, extract and check claims, rewrite / drop, report | 2 days |
| `train/build_boundary_set.py` (new, from `build_sft_warmstart.py`) | parts and counts above, entity splits, form-balance and length checks, thinking traces | 2 days |
| `train/pretok_sft.py` | thinking-mode serialisation and masks; token-share report | 0.5 day |

## Budget

| Step | local-h |
|---|---:|
| 1. Measurement and baselines | 40 |
| 2. Probe | 130 |
| 3. Clean the mix | 230 |
| 4. Boundary set and audit | 80 |
| 5. Transfer pilot (5 arms) | 1,500 |
| 6. Checkpoint and test evaluation | 150 |
| **Total** | **~2,100** |

Steps 1–4 (~480 local-h) can start now. Step 5 is the one large spend and needs approval. The main SFT run itself is
on the SFT team's allocation.

## Decisions

1. Approve steps 1–4 now.
2. Approve the step-5 pilot (~1,500 local-h) after step 4, or run it smaller (3 arms, A / C / D: ~900).
3. Agree with the SFT team on general-eval margins and the handover format before step 5.
