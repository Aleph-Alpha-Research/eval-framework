# Differences to lm-evaluation-harness

What `eval-framework` offers that [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) does
not, and the other way round.

> **Snapshot: 13 November 2026.** Compared `eval-framework` 0.23.0 (252 registered task ids) with
> lm-evaluation-harness `main` at commit `205802a8` from 3 October 2026 (228 task directories, 18,854 task YAML files).
> Both projects change quickly. If a row is out of date, please open an issue.

## Summary

| | eval-framework | lm-evaluation-harness |
| --- | --- | --- |
| "I don't know" variants of standard benchmarks | ✅ 5 tasks | ❌ |
| Distributional-correctness, ternary and confidence-weighted scores | ✅ | ❌ |
| Accuracy with a length correction fitted from the data | ✅ Bayesian loglikelihood accuracy | ❌ fixed normalisation only (`acc_norm`, `acc_bytes`) |
| Bits-per-byte on the answer of a question | ✅ | ❌ running text only |
| Corpus- and prior-level bits-per-byte | ✅ | ❌ |
| EllaMind European-language translations | ✅ 13 datasets | other translations for some |
| OLMES prompt variants | ✅ | ❌ |
| Number of tasks | 252 task ids | 18,854 task files in 228 directories |
| Model backends | OpenAI-compatible endpoints | Hugging Face, vLLM, GGUF, OpenAI, Anthropic and many more |

The metrics and the "I don't know" tasks are currently only available through the Python API, with the model served
by vLLM. The CLI cannot run them yet.

## Metrics

| eval-framework metric | Closest metric in lm-evaluation-harness | Difference |
| --- | --- | --- |
| Accuracy Bayesian Loglikelihood | `acc_norm`, `acc_bytes` | lm-eval divides each answer's log-likelihood by its length in characters or bytes. eval-framework fits one length-correction factor per run from the data ([paper](https://arxiv.org/abs/2607.12767)). |
| BitsPerByte | `bits_per_byte` | lm-eval reports it for running-text (perplexity) tasks only. eval-framework reports it for the answer of each question. |
| Corpus BPB, Prior BPB | none | |
| Distributional Correctness Score | none | Reported for the "I don't know" tasks only. |
| Ternary Score | none | Reported for the "I don't know" tasks only. |
| Confidence-weighted Accuracy | none | Reported for the "I don't know" tasks only. |

## "I don't know" variants

These tasks add one scored abstention answer to an existing benchmark and tell the model that abstaining scores 0.

| Task | Base benchmark | In lm-evaluation-harness |
| --- | --- | --- |
| `ARC_IDK` | ARC | base benchmark only |
| `MMLU_IDK` | MMLU | base benchmark only |
| `MMLU_PRO_IDK` | MMLU-Pro | base benchmark only |
| `PIQA_IDK` | PIQA | base benchmark only |
| `GoldenSwag_IDK` | GoldenSwag | not available |

lm-evaluation-harness has abstention only where it is part of the original dataset, such as BBQ and SQuAD v2.

## Datasets

Only in eval-framework:

| Dataset | Source |
| --- | --- |
| BigCodeBench | `bigcode/bigcodebench` |
| GoldenSwag | `PleIAs/GoldenSwag` |
| LBPP | `CohereLabs/lbpp` |
| MultiPL-E | `nuprl/MultiPL-E` |
| German IFEval | `jzhang86/de_ifeval` |
| HLE, German | `ellamind/hle-multilingual` |
| SimpleQA, German | `ellamind/simpleqa-verified-multilingual` |
| Natural Questions and DROP as multiple choice | `allenai/nq-gen2mc`, `allenai/drop-gen2mc` |

EllaMind translations. lm-evaluation-harness has none of these uploads; for some benchmarks it has translations from
other sources.

| Benchmark | eval-framework source | Other translations in lm-evaluation-harness |
| --- | --- | --- |
| ARC | `ellamind/arc-multilingual` | yes, e.g. `LumiOpen/arc_challenge_mt`, `alexandrainst/m_arc` |
| HellaSwag | `ellamind/hellaswag-multilingual` | yes, e.g. `alexandrainst/m_hellaswag` |
| PIQA | `ellamind/piqa-multilingual` | yes, e.g. `HiTZ/PIQA-eu` |
| Social IQA | `ellamind/siqa-multilingual` | yes, e.g. `projecte-aina/siqa_ca` |
| WinoGrande | `ellamind/winogrande-multilingual` | yes, e.g. `mideind/icelandic-winogrande` |
| CommonsenseQA | `ellamind/csqa-multilingual` | related sets in Bangla, Japanese and Norwegian |
| GPQA | `ellamind/gpqa-multilingual` | none found |
| GSM8K Platinum | `ellamind/gsm8k-platinum-multilingual` | Platinum in English only; MGSM covers other languages of GSM8K |
| Hendrycks MATH | `ellamind/hendrycks-math-multilingual` | English only |
| HumanEval | `ellamind/humaneval-multilingual` | English only |
| MBPP | `ellamind/mbpp-multilingual` | English only |

## Prompt styles

eval-framework offers the same dataset as multiple choice, cloze, bits-per-byte or chain of thought, plus OLMES
prompts; see the [naming convention](naming_convention.md). lm-evaluation-harness does this for some benchmarks: MMLU comes in six styles, GPQA in five and ARC in two, while
HellaSwag, PIQA, CommonsenseQA and WinoGrande have one prompt style each.
It has no OLMES variants.

## How this was checked

All task files of the lm-evaluation-harness checkout were scanned for dataset identifiers, task names, metric names
and abstention wording in answer options. eval-framework's datasets and metrics were then matched against them by
exact name, by name similarity and by shared words.

Limits:

* A dataset re-uploaded under an unrelated name would not be found.
* "None found" means no match in this snapshot, not a guarantee.
* The comparison lists differences only. Benchmarks that both projects implement are not compared for score parity.
