# On-policy DPO data against hallucination

Builds preference pairs from the model's own answers so that it learns where its knowledge ends: answer what it
knows, say plainly and helpfully when it does not, never present invented facts. This is step 3 of the
[plan](../docs/PLAN.md). The same question pool and judged samples also give the knowledge-boundary SFT set
(part C of the [SFT plan](../docs/SFT_PLAN.md)).

## Pipeline

| Step | Script | Where | What it does |
|---|---|---|---|
| 1 | `build_pool.py` | any machine with internet | ~9,000 natural questions in 10 languages (below), no answer format |
| 2 | `sample.py` | 1 GPU per shard | 8 answers per question from the policy, default mode, temperature 0.7, top-p 0.95 |
| 3 | `../benchmark/judge.py` | 2 GPUs | gemma-4-31B-it labels every answer (same rubric as the benchmark) |
| 4 | `build_pairs.py curate` | 2 GPUs | a stricter check of every chosen candidate (below); gemma writes a good "I don't know" where none of the policy's own passes, and an unnecessary refusal as the rejected side of questions the policy always answers right |
| 5 | `build_pairs.py pairs` | CPU | pairs, train/validation split, `knowledge.jsonl` (accuracy per question), manifest |

Steps 2-5 run as one job: `run_slurm.sbatch` (see its header for the environment variables).

## Questions (`build_pool.py`)

Per language (en de fr es it nl sv pl pt fi):

| Part | Count | Source | Reference for the judge |
|---|---|---|---|
| Short factual questions | 400 (tail 150, torso 130, head 120) | eu_qa train split (birth year and place, film director, book author, company founding, river mouth, regional capital), format instruction removed | the Wikidata answer and its other names |
| Open questions about real people | up to 200 (obscure 90, mid 60, known 50) | Wikidata, the benchmark sampler with another seed and 14 occupations | birth date and place, occupations, description |
| Made-up subjects | 300 | invented surnames, plausible everyday names (common first name + common surname), invented novels, films, companies, villages, rivers, paintings, albums | "does not exist" |

Nothing overlaps with the benchmark: its Wikidata ids and subject names are excluded, and the made-up names come
from other name lists and syllables. The plausible everyday names matter: with only odd-sounding invented names the
model could learn "strange name, so abstain" instead of checking what it knows.

## Pairs (`build_pairs.py`)

Accuracy per question = (CORRECT + ½ PARTIAL) / usable samples (not cut off, right language).

| Case | When | Chosen | Rejected |
|---|---|---|---|
| made_up | made-up subject, some answer invents facts | a good "I don't know" | an invented answer |
| style | made-up subject, only unhelpful refusals | a good "I don't know" | the unhelpful refusal |
| unknown | real subject, accuracy < 0.5 | a good "I don't know" | a wrong answer |
| sharpen | real subject, 0.5 ≤ accuracy < 0.75 | a correct answer | a wrong answer |
| known | real subject, accuracy ≥ 0.75 | a correct answer | a refusal, else a wrong answer |

Chosen answers are checked more strictly than the judge's label (`CHECK` in `build_pairs.py`), shortest candidate
first. In the smoke test, samples labelled `ABSTAIN_GOOD` still called made-up subjects fictional or suggested other,
probably invented, people, and some `CORRECT` answers added doubtful details or garbled words (more of those at
temperature 1, hence 0.7). A chosen "I don't know" must say the model does not know, without claiming the subject
does not exist, without any facts or suggested matches, and without false excuses; a chosen answer must be fluent,
right, and free of doubtful extra details. The policy's own answer is used whenever one passes, so the pair stays
on-policy; gemma writes one only when none does. Answering pairs (known + sharpen) are at least 35 % of the total, so the model is also pushed to
answer what it knows. The 0.5 threshold is where answering pays off under a +1 / −1 score for right and wrong
answers with 0 for abstaining.

The output (`pairs/natural-halu-{train,validation}.jsonl.gz`) uses the usual DPO row format
(`prompt`, `chosen`, `rejected` as message lists, plus `source` and `language`). Mix it with the CoCoNot contrast
pairs and some general preference data before training.

## Limitations (version 1)

- Default mode only; thinking-mode pairs are a later step.
- The middle band is trained toward the right answer, not toward a hedged one ("I believe ..., but I'm not sure").
- Made-up titles are syllable-built, so they look less natural than real titles.
- The judge is an LLM; read a sample of pairs before training.
