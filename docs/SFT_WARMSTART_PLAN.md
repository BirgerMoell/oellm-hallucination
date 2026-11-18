# Plan: SFT warm start, then on-policy DPO (round 2)

Status 18 November 2026. Written after round 1, where one DPO round on the curated pairs moved Prelude-9B-SFT-v2
only from 97.7 % to 92.7 % fabrication (benchmark, default mode), with no over-refusal and no language loss. The
same plan applies to the RL merge if its round-1 result (training) is also weak.

## Why round 1 was weak

- The model almost never says "I don't know" (2 % of made-up subjects), so 81 % of its chosen answers are written
  or rewritten by gemma. DPO raised their likelihood (held-out reward accuracy 79 %) without making them the
  model's own first choice.
- The recipe was the pilot's: learning rate 1e-6, one epoch, 43 steps, built for a model that already abstained.

## The plan

```
round-1 curated answers ──► 1. SFT warm start ──► 2. re-sample + judge + curate ──► 3. DPO (stronger) ──► 4. eval
   (half A of the pool)                            (half B of the pool, new policy)
```

### 0. Split the question pool

Split the 8,644 pool questions into halves A and B by a hash of the question id, stratified by band and language.
Half A trains the SFT warm start; half B is re-sampled from the warm-started model for DPO. The DPO pairs then come
from questions the model was not fine-tuned on, so they show what it actually learned rather than what it memorised.

### 1. SFT warm start

Data (half A, from round 1's `curated.jsonl`, which already passed the strict check and the audit):

| Part | Rows (Prelude-SFT-v2) | Source | Purpose |
|---|---|---|---|
| Honest "I'm not sure" on real subjects it gets wrong | ~1,300 | `unknown`, written, detail-question rule | unsure of the detail, not of the subject |
| Honest "I don't know" on made-up subjects | ~700 | `made_up`, own where it passed, else written | no invented facts, no "does not exist" |
| Correct answers it knows | ~400 | `known` + `sharpen` (concise rewrites) | keeps answering what it knows |
| False-premise answers | ~350 | CoCoNot, gemma-checked | real things are not called non-existent |
| Identity (Prelude, v3.1.0) | ~1,000 | openeurollm-model-identity, 10 languages | cutoff, no invented facts about itself |
| General replay | ~4,000 | thinking-off chat / IF rows in the model's own SFT style (INTELLECT-3-SFT `am_chat`, `am_if`; fallback: chosen side of the general DPO pairs) | other behaviour does not drift |

Answers (known + sharpen + CoCoNot) stay at least 35 % of the hallucination rows: if half A has fewer known rows,
the abstention rows are cut to match, never the other way round.

Training: assistant-only loss; the model's own chat template in default mode (generation prompt with the empty think
block + reply + `<|im_end|>`, as in `train/pretok_pairs.py --mode concat`); packed to 16K; 2 nodes x 4 A100.
Starting point: the deep-search warm-start recipe (`sft_oellm_deep_search_warmstart_v1_leonardo.yaml`) with
learning rate 5e-6 and 2 epochs: these are short replies and a behaviour the model lacks, not scripted
trajectories, and the data is only ~8K rows (about 40 optimizer steps).

Quick check before step 2 (benchmark, default mode only, ~15 local-h): fabrication should already drop clearly
(target below 60 %), over-refusal must stay below 5 %, language at least 95 %. If over-refusal rises, cut the
abstention rows and retrain before going on.

### 2. Re-sample, judge, curate (half B)

`train/run_slurm.sbatch` with the warm-started model as policy and half B as pool: 8 samples, temperature 0.7,
gemma judge, the round-4 curation (own answers only where they audited at 95 % or more), Qwen3.5-27B audit with the
same bars (prefers chosen at least 95 %, every cell at least 90 % acceptable, openings at most 10-15 % per
language). Expected: many more of the model's own good abstentions, so a larger own share than round 1's 19 %.

### 3. DPO, stronger

| Setting | Round 1 | Round 2 |
|---|---|---|
| Learning rate | 1e-6 | 3e-6 (5e-6 if the quick check after SFT shows little change) |
| Loss | sigmoid | sigmoid + SFT loss on chosen (TRL `loss_type: [sigmoid, sft]`, SFT weight 0.5) |
| Epochs | 1 | 1 |
| beta | 0.1 | 0.1 |
| Checkpoints | end only | every 20 steps (resumable; round 1's merge run lost an hour to an out-of-memory crash) |
| Memory | `PYTORCH_ALLOC_CONF` only | also `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

Mix as in round 1: the hallucination pairs, the CoCoNot contrast pairs, 3,000 general preference pairs.

### 4. Evaluation and gates

| Gate | Threshold |
|---|---|
| Fabrication on made-up subjects (benchmark, default) | below 10 % for release; this round: below 40 % |
| Over-refusal on well-known people | below 5 % |
| Answers in the question's language | at least 95 % |
| Well-known people correct | not below the starting model minus 3 points |
| Identity eval (`eval_identity.py`) | the gates in the identity card |
| General evals | no regression beyond noise (the model team's suite for Prelude-SFT-v2; reasoning-quick, tool held-out and oellm-eval for the merge) |

## Cost (local-h, per model)

| Step | Estimate |
|---|---|
| SFT warm start (2 nodes, about 30 min) plus conversion | 40 |
| Quick benchmark check | 15 |
| Re-sample, judge, curate, audit (half B) | 50 |
| DPO (2 nodes, about 1 h) | 70 |
| Full benchmark and general evals | 30 |
| **Total** | **~200** |

## Decisions for the user

- Run it on Prelude-SFT-v2 only, or also on the merge (after its round-1 result)?
- General replay for Prelude-SFT-v2: its own SFT data (INTELLECT-3-SFT; check that it is public and its licence) is the safer choice for keeping its
  style; confirm we may use it.
