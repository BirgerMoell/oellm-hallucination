# SFT plan: teaching the model what it does not know

For the next supervised fine-tuning (SFT) round of the OpenEuroLLM instruct model. The goal: the model answers what
it knows, says plainly and helpfully when it does not, and never presents invented facts as real. It must do this
in natural conversation, not only when a prompt offers "write UNKNOWN".

## Why SFT first

Our probe of the current RL model (see the [README](../README.md)) found:

- closed-book, it invents full biographies for made-up people (12 of 12) and for obscure real people (11 of 12);
- its few refusals use a false excuse ("I don't have the capability to access information about novels");
- the calibration learned from `ANSWER: UNKNOWN` prompts does not transfer to natural questions at all.

The likely causes are in SFT. The mix has almost no examples of a good "I don't know" in natural form. It also
contains confident answers about long-tail facts the base model never learned, which teaches the model to answer
whether it knows or not (Gekhman et al., 2024). Later DPO and RL rounds fix what SFT cannot, but they work much
better from a model that has seen the target behaviour.

## What goes in

| Part | Rows | Languages | Source | Share of a 1M-row mix |
|---|---|---|---|---|
| A. Identity v3 | 5,508 | 36 | [openeurollm-model-identity](https://huggingface.co/datasets/openeurollm/openeurollm-model-identity) v3, as is | 0.55 % |
| B. CoCoNot subset | ~2,300 English + ~5,200 translated | 36 | `allenai/coconot`, filtered and rewritten (below) | 0.75 % |
| C. Knowledge-boundary set (new) | ~6,000 | 36 | Built per model with the Idk / R-Tuning recipe (below) | 0.6 % |
| D. Audit of the existing mix | removes or rewrites | all | grep + sampling (below) | n/a |

About 19,000 new rows in total. For a mix of a different size, keep the same counts rather than the same share:
these are behaviours to learn, not knowledge to absorb, and a few thousand examples per behaviour is enough.

### A. Identity v3

Use the train split unchanged; keep validation out. It teaches the stated knowledge cutoff (pretraining data to
December 2025, annealing data from the first half of 2026), "I don't know today's date", "my information may be out
of date" for recent events, and no invented model names. These are the self-knowledge half of not hallucinating.

### B. CoCoNot subset

| Subcategory | Take | Processing |
|---|---|---|
| false presuppositions | all 717 | none |
| universal unknowns | all 302 | none |
| temporal limitations | all 341 | rewrite every stated cutoff ("as of my last update in August 2023") to ours, with a template plus a judge check |
| underspecified | 800 of 2,729 | sample across topics |
| incomprehensible | 150 of 392 | none |

Translate about 150 rows per language into the other 35 languages with the same pipeline as identity v3
(gemma-4-31B-it on Leonardo, with automatic checks). Leave out the humanizing, safety, modality and style
subcategories (see [DATASETS.md](DATASETS.md)). The 927 contrast DPO pairs go to the DPO stage, not SFT.

### C. Knowledge-boundary set (the core)

The Idk / R-Tuning recipe, adapted so that the targets are natural answers instead of a fixed refusal string.

1. **Questions.** About 12,000 natural questions without any answer format, in 36 languages:
   - real people, works and places from Wikidata, across popularity bands (Wikipedia sitelinks);
   - made-up subjects from the benchmark generator (`benchmark/build_questions.py`), with names checked against
     Wikidata;
   - **no overlap** with the benchmark questions (different seed; drop shared subjects).
2. **Probe the model's knowledge.** Sample the SFT starting checkpoint 8 times per real question (temperature 0.7,
   closed-book) and grade each sample against the Wikidata reference with the benchmark judge.
3. **Targets by what the model knows:**

   | Model gets it right | Target answer |
   |---|---|
   | 6-8 of 8 | Its own correct answer (a judged-correct sample), so SFT teaches no new facts |
   | 3-5 of 8 | A hedged answer: the majority answer with an honest caveat ("I believe ..., but I'm not certain") |
   | 0-2 of 8, real subject | A good "I don't know" (below) |
   | made-up subject | A good "I don't know" |

4. **What a good "I don't know" looks like** (written by gemma-4-31B-it from a style guide, then checked by the
   judge: no invented facts, correct language):
   - says plainly that it doesn't recognise or can't recall the subject: "I don't know of a composer called
     Elin Velkenqvist";
   - no false excuse ("I can't access databases") when the honest reason is that it doesn't know;
   - adds what genuinely helps: asks for context, checks for a likely misspelling of a *real* subject it does know
     (without claiming they are the same), or suggests where to look or that search could help;
   - short, and in the user's language.
5. **Balance:** about 40 % made-up subjects, 30 % unknown real, 30 % known real. The known share keeps the model from
   learning "refuse anything about a person".
6. **Modes:** 80 % default mode, 20 % thinking mode, where the reasoning trace notices the uncertainty ("I don't
   recognise this name; I shouldn't guess") before the answer.

Cost on Leonardo: about 12,000 × 8 samples plus judging, roughly 40-60 local-h.

### D. Audit of the existing SFT mix

| Find | How | Action |
|---|---|---|
| Stale cutoff claims ("as of my last update in 2023", "my knowledge cutoff is ...") | regex in every language of the mix | rewrite to our cutoff or drop |
| False capability excuses ("I don't have the capability to access ...") on knowledge questions | regex + judge | rewrite as an honest "I don't know" or drop |
| Long-tail factual answers the base model does not know | step-C sampling on a 5 % sample of factual QA rows; estimate the unknown share | if over ~10 %, drop or convert the unknown ones |
| Identity claims about other models ("As an OpenAI model ...") | regex | drop (already part of the identity audit) |

## Evaluation and gates

Run before and after SFT, on the same checkpoints:

1. **This repo's benchmark** (`benchmark/`): fabrication, good abstention, obscure-person accuracy and errors,
   over-refusal on known people, language following. Targets for the SFT model, closed-book default mode:
   - fabrication on made-up subjects **below 30 %** (from 64 % today for our best model);
   - over-refusal on known people **below 5 %**;
   - known-person accuracy no more than 2 points below the previous model.
2. The identity v3 validation split: cutoff, organisations and languages stated correctly.
3. The existing suite (EU QA, deep search, reasoning, code, EUROPA): no regression beyond noise.

SFT alone will probably not reach the final targets (fabrication below 10 %). The DPO and RL steps in
[PLAN.md](PLAN.md) take it the rest of the way, and they need the benchmark and judge built here.

## Order of work

1. Build the benchmark and measure the current models (done: see the README).
2. Run the audit (D): cheap, and it tells us how much of the problem the existing mix causes.
3. Build B and C, then train SFT with A + B + C mixed in.
4. Measure, then run on-policy DPO (PLAN.md step 3) from the SFT checkpoint.
