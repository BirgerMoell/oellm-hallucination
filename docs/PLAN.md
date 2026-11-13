# Plan: stop the model from making things up

## What we know

- The model knows what it knows: it answers famous people correctly, and with a search tool it gets obscure people
  right 12 of 12.
- It cannot tell when it does not know: closed-book, it invents full biographies for made-up and obscure people.
- Our calibration training so far (RL with `ANSWER: UNKNOWN` prompts) only works on prompts that offer that option.
- Its few refusals use a false excuse instead of saying "I don't know".

## Steps

| # | Step | Cost (local-h) | Result |
|---|---|---|---|
| 1 | Natural-question hallucination benchmark (`benchmark/`) | ~20 per model | Baseline numbers; the yardstick for every later step |
| 2 | SFT data: audit the mix, add identity v3, CoCoNot subset and the knowledge-boundary set ([SFT_PLAN.md](SFT_PLAN.md)) | ~60 to build, plus the SFT run | First fix; sets the target style |
| 3 | On-policy DPO on the model's own knowledge limits | ~200-250 | Main lever (below) |
| 4 | RL with a judge-graded natural-question reward | ~500-700 | Combined fix that keeps search skills |
| 5 | Long-form factual precision (biographies, summaries) | later | Only state what it is confident of |
| 6 | Release gate | in every eval | Fabrication and over-refusal thresholds |

### 3. On-policy DPO

For every natural question, sample the current model 6-8 times:

- real question, most samples right: the preferred answer is the model's correct answer;
- real question, samples disagree or are wrong: the preferred answer is an honest hedge or "I don't know";
- made-up subject: the preferred answer is always a good "I don't know".

The rejected answer is the model's own fabrication, or a needless refusal on a question it knows. The judge
(gemma-4-31B-it) picks the pairs, and CoCoNot's 927 contrast pairs are added against over-refusal. This teaches the
model its own limits instead of a fixed rule.

### 4. RL reward for natural questions

Keep the exact `ANSWER:` rows (cheap, verifiable) and add a judge-graded natural-question slice:

| Outcome | Reward |
|---|---|
| Correct answer | +1 |
| Good "I don't know" on a subject the model cannot know | +0.5 |
| Unhelpful refusal or false excuse | 0 |
| Refusing a question about something it knows | -0.5 |
| Fabrication or a wrong answer | -1 |

The penalty for refusing known questions is what prevents over-refusal. With a search tool, the policy should look
up obscure facts before stating them, answer in the user's language (it sometimes switches to the article's
language), and not mention the search snapshot's date.

### 6. Release gate

Thresholds on the benchmark, closed-book, default mode:

- fabrication on made-up subjects below 10 %;
- over-refusal on known people below 5 %;
- no drop in the other evals.

## What not to do

- More `ANSWER: UNKNOWN`-only training: it does not transfer to natural questions.
- A large reward for abstaining: the model learns to refuse things it knows.
- SFT on long-tail facts the model does not know: it teaches confident guessing.
