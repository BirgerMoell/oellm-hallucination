# Datasets for reducing hallucination

What exists, what each one teaches, and how we would use it for OpenEuroLLM. Checked 13 November 2026.

The main lesson from this line of work: **"I don't know" data is model-specific.** Whether a model should answer or
abstain depends on what that model knows, so the most useful datasets are *recipes* that are rebuilt for each model
(Idk, R-Tuning), plus a few model-independent categories (false premises, unknowable questions) that transfer.
Gekhman et al. (2024) also showed that fine-tuning on facts a model does not already know *increases* its
hallucination, which is why the SFT plan audits the existing SFT mix as well as adding new data.

| Dataset | What it is | Size | Language | Licence | How we use it |
|---|---|---|---|---|---|
| [allenai/coconot](https://huggingface.co/datasets/allenai/coconot) ([paper](https://arxiv.org/abs/2407.12043), [code](https://github.com/allenai/noncompliance)) | When *not* to simply answer: false premises, unknowable questions, temporal limits, underspecified and incomprehensible requests, modality limits, plus safety. Contrast set of look-alike requests that *should* be answered, and DPO pairs. Used in AI2's Tulu 3 mix. | 11,477 train, 1,001 test, 379 contrast test, 927 DPO pairs | English | The card states both AI2 ImpACT-LR and ODC-BY; the code repo is MIT | SFT: the hallucination-relevant subcategories (below), translated. DPO: the contrast pairs against over-refusal. |
| [OpenMOSS/Say-I-Dont-Know](https://github.com/OpenMOSS/Say-I-Dont-Know) "Idk" ([paper, ICML 2024](https://arxiv.org/abs/2401.13275)) | TriviaQA questions split per model into known and unknown; unknown ones get a refusal target. Preference data and best-of-N recipe. | Per model (Llama-2-7B/70B, Mistral-7B, Baichuan2-7B) | English | None stated (fine for our use) | **The recipe** for our own knowledge-boundary data (SFT plan step 3). Their released data is for other models. |
| [shizhediao/R-Tuning](https://github.com/shizhediao/R-Tuning) ([paper, NAACL 2024](https://arxiv.org/abs/2311.09677)) | Same idea: answers the model gets right are kept, the rest are marked uncertain. | Per model, 7-9 QA sets | English | None stated (fine for our use) | Recipe, as Idk. |
| [openeurollm/openeurollm-model-identity](https://huggingface.co/datasets/openeurollm/openeurollm-model-identity) v3 | Self-knowledge: who trained the model, knowledge cutoff, languages, what it cannot know (date, recent events), no invented model names. | 5,508 train, 923 validation | 36 languages | CC BY 4.0 | SFT, as is. (v3 is built; publishing it to the org repo is pending.) |

## Evaluation-only sets (never train on these)

| Set | Measures |
|---|---|
| SelfAware (Yin et al., 2023) | Unanswerable vs answerable questions |
| FalseQA (Hu et al., 2023) | False-premise questions |
| SimpleQA (OpenAI, 2024) | Short factual answers, with "not attempted" counted separately |
| TruthfulQA (Lin et al., 2022) | Common misconceptions |
| AbstentionBench (Meta, 2025) | Abstention across many sources |
| **This repo's benchmark** (`benchmark/`) | Natural questions about made-up and real people/things in 10 languages, judged for fabrication, accuracy and over-refusal |

## CoCoNot subcategories for hallucination

From the `original` train split:

| Category / subcategory | Train rows | Use | Note |
|---|---|---|---|
| Incomplete / false presuppositions | 717 | yes | Correct the premise instead of going along with it |
| Indeterminate / universal unknowns | 302 | yes | Questions nobody can answer |
| Unsupported / temporal limitations | 341 | yes, rewritten | Responses say "as of my last update in August 2023": replace with our cutoff (December 2025, some 2026 data) |
| Incomplete / underspecified | 2,729 | partly (~800) | Ask for clarification; large, so subsample |
| Incomplete / incomprehensible | 392 | small share | |
| Indeterminate / subjective questions | 599 | optional | Not hallucination |
| Humanizing requests (11 subcategories) | 1,795 | no | Refuses opinions and preferences; risks over-refusal |
| Safety (6 subcategories) | 3,136 | no (handled by the safety data) | |
| Unsupported / modality, style and length | 1,466 | no | Depends on the deployment |

CoCoNot does **not** cover our main failure: confident answers about made-up or obscure people, places and works.
That needs our own data (SFT plan steps 2-3).
