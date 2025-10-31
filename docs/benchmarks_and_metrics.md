# Included Benchmark Tasks

Currently, the framework covers a wide range of pre-training and post-training benchmarks for completion and loglikelihood tasks, as well as benchmarks that use LLM-as-a-judge evaluation methods. The suggested few-shot counts are extracted from other leaderboards and literature.

Additional task documentation can be generated with the script `utils/generate-task-docs.py` as documented in [installation.md](installation.md). The documention can thereafter be found in [docs/tasks](tasks/README.md).

## Completion (updated table)

| **Task** | **Capability** | **Benchmarks** | **Long Context** |
|-|-|-|-|
|   Logical Reasoning   |   Math   |   `AIME2024`, `GSM8K`, `GSM8K_EU20_DE`, `GSM8K_EU20_FR`, `GSM8KEvalHarness`,`GSM8KReasoning`, `MATH`, `MATH500`, `MATHLvl5`, `TableBench`   |   `InfiniteBench_MathFind`   |
|   Logical Reasoning   |   Programming   |   `BigCodeBench`, `BigCodeBenchHard`, `BigCodeBenchInstruct`, `BigCodeBenchHardInstruct`, `HumanEval`, `HumanEvalInstruct`, `MBPP`, `MBPP_PROMPT_WITHOUT_TESTS`, `MBPP_SANITIZED`, `MBPP_PROMPT_WITHOUT_TESTS_SANITIZED`   |   `InfiniteBench_CodeRun`   |
|   Logical Reasoning   |   Puzzle   |   `SPHYR`   |    |
|   Output Control   |   Structure   |   `IFEval`, `IFEvalDe`, `IFEvalFiSv`, `RenderableStructEval`, `StructEval`   |    |
|   Text Distillation   |   Aggregation   |    |   `ZERO_SCROLLS_SPACE_DIGEST`   |
|   Text Distillation   |   Classification   |   `GPQA_COT`, `MMLU`, `MMLU_PRO_COT`, `MMMLU_GERMAN_COT`, `PAWSX`, `TRIVIAQA`   |    |
|   TextDistillation   |   Closed QA   | `SQUAD`, `SQUAD2` |   `InfiniteBench_EnDia`   `InfiniteBench_EnQA`   |
|   Text Distilation   |   Extraction   |   `DUC_ABSTRACTIVE`, `DUC_EXTRACTIVE`   |   `InfiniteBench_RetrieveKV2`, `InfiniteBench_RetrieveNumber`, `InfiniteBench_RetrievePassKey1`   |
|   Text Distillation   |   QA   |    |   `ZERO_SCROLLS_GOV_REPORT`, `ZERO_SCROLLS_MUSIQUE`, `ZERO_SCROLLS_NARRATIVEQA`, `ZERO_SCROLLS_QASPER`, `ZERO_SCROLLS_QMSUM`, `ZERO_SCROLLS_SQUALITY`   |
|   Text Transformation   |   Translation   |   `Flores200`, `FloresPlus`, `WMT14`, `WMT14_INSTRUCT`, `WMT16, WMT16_INSTRUCT, WMT20, WMT20_INSTRUCT`   |    |

## Completion (old table) TODO: delete table

| Task Name | Tag | Task | Capability | Common Few-Shot Counts | Language |
|-|-|-|-|-|-|
| AIME 2024 | `AIME2024` | Logical Reasoning | Math | [0] | en |
| DUC Abstractive | `DUC_ABSTRACTIVE` | Text Distilation | Extraction | [0, 5] | en |
| DUC Extractive | `DUC_EXTRACTIVE` | Text Distilation | Extraction | [0, 5] | en |
| Flores 200 | `FLoRes-200` | Text Transformation | Translation | [0, 5, 8] | en, fin |
| Graduate School Math 8K | `GSM8K` | Logical Reasoning | Math | [0, 5, 8] | en |
| Graduate School Math 8K Llama Style Prompt Formatting | `GSM8K Llama Version` | Logical Reasoning | Math | [0, 5, 8] | en |
| IFEval | `IFEval` | Output Control | Structure | not supported | en |
| IFEval Finnish & Swedish | `IFEval Finnish & Swedish` | Output Control | Structure | not supported | en |
| HumanEval | `Human Eval` | Logical Reasoning | Programming | [0] | en |
| Math | `Math` | Logical Reasoning | Math | [0, 5, 8] | en |
| Math Lvl 5 | `Math Lvl 5` | Logical Reasoning | Math | [0, 5, 8] | en |
| Math 500 | `MATH500` | Logical Reasoning | Math | [0] | en |
| MBPP | `MBPP` | Logical Reasoning | Programming | [0] | en |
| MBPP Sanitized | `MBPP_SANITIZED` | Logical Reasoning | Programming | [0] | en |
| MBPP | `MBPP_PROMPT_WITHOUT_TESTS` | Logical Reasoning | Programming | [0] | en |
| MBPP | `MBPP_PROMPT_WITHOUT_TESTS_SANITIZED` | Logical Reasoning | Programming | [0] | en |
| PAWS-X | `PAWS-X` | Text Distillation | Classification | [0, 5] | en |
| SPhyR  | `SPHYR` | Logical Reasoning | Puzzle | [0] | en |
| SQuAD | `SQuAD` | Text Distillation | Closed QA | [0, 5] | en |
| SQuAD v2 | `SQuAD2` | Text Distillation | Closed QA | [0, 5] | en |
| TableBench | `TableBench` | Text Distillation, Logical Reasoning | Classification, Math | [0] | en |
| TriviaQA | `TriviaQA` | Text Distillation | Classification | [0, 5] | en |
| WMT 14 | `WMT14` | Text Translation | Translation | [0, 5] | en, fr |
| WMT 16 | `WMT16` | Text Translation | Translation | [0, 5] | en, ger |
| WMT 20 | `WMT20` | Text Translation | Translation | [0, 5] | en, ger, fr |

## Loglikelihoods (updated table)

|   **Task**    | **Capability** | **Benchmarks** | **Long Context** |
|-|-|-|-|
|   Output Control   |   Bias   |   `WINOGENDER`   |    |
|   Text Distillation   |   Classification   |   `ARC`,`ARC_DE`, `ARC_EU20_DE`, `ARC_EU20_FR`, `ARC_FI`, `BELEBELE`, `ChemBench`, `FullTextMMLU`, `GPQA`, `INCLUDE`, `MMLU`, `MMLU_DE`,`MMLU_EU20_DE`, `MMLU_EU20_FR`, `MMMLU`, `MMLU_PRO`, `OPENBOOKQA`, `PIQA`, `SCIQ`, `SCIQEvalHarness`, `TRUTHFULQA`, `TRUTHFULQA_EU20_DE, TRUTHFULQA_EU20_FR`   |    |
|   Text Distillation   |   QA   |    |   `QUALITY`, `ZERO_SCROLLS_QUALITY`   |
|   Text Generation   |   Open QA   |   `CASEHOLD`   |    |
|   Logical Reasoning   |   Closed QA   |    |   `InfiniteBench_EnMC`   |
|   Logical Reasoning   |   Programming   |    |   `InfiniteBench_CodeDebug`   |
|   Logical Reasoning   |   Reasoning   |   `COPA`, `HELLASWAG, HELLASWAG_EU20_DE, HELLASWAG_EU20_FR`, `WINOGRANDE`, `WINOX_DE`, `WINOX_FR`   |    |

## Loglikelihoods (old table)  TODO: delete table

| Task Name | Tag | Task | Capability | Common Few-Shot Counts | Language |
|-|-|-|-|-|-|
| Abstract Reasoning Challenge | `ARC` | Text Distillation | Classification | [0, 5, 25] | en |
| Abstract Reasoning Challenge German | `ARC German` | Text Distillation | Classification | [0, 5, 25] | ger |
| Abstract Reasoning Challenge Finnish | `ARC Finnish` | Text Distillation | Classification | [0, 5, 25] | fin |
| Casehold | `CaseHold` | Text Generation | Open QA | [0, 5] | en |
| COPA | `COPA` | Logical Reasoning | Reasoning | [0] | en |
| GPQA | `GPQA` | Text Distillation | Classification | [0] | en |
| Hellaswag | `HellaSwag` | Logical Reasoning | Reasoning | [0, 5, 10] | en |
| Hellaswag German | `HellaSwag German` | Logical Reasoning | Reasoning | [0, 5, 10] | en, ger |
| MMLU | `MMLU` | Text Distillation | Classification | [0, 5] | en |
| Full Text MMLU | `Full Text MMLU` | Text Distillation | Classification | [0, 5] | en |
| MMLU Pro | `MMLU Pro` | Text Distillation | Classification | [0, 5] | en |
| MMMLU | `MMMLU` | Text Distillation | Classification | [0, 5] | fr, de, es, it, pt, ar |
| PIQA | `PIQA` | Text Distillation | Classification | [0] | en |
| OpenBookQA | `OpenBookQA` | Text Distillation | Classification | [0] | en |
| SciQ | `SciQ` | Text Distillation | Classification | [0] | en |
| TruthfulQA | `TruthfulQA` | Text Distillation | Classification | [0, 6] | en |
| TruthfulQA French | `TRUTHFULQA_EU20_FR` | Text Distillation | Classification | [0, 6] | en |
| TruthfulQA German | `TRUTHFULQA_EU20_DE` | Text Distillation | Classification | [0, 6] | en |
| Winogender | `Winogender` | Output Control | Bias | [0, 5] | en |
| Winogrande | `Winogrande` | Logical Reasoning | Reasoning | [0, 5] | en |

## Long-Context

| Task Name                      | Tag                              | Task | Capability                                       | Domain | Common Few-Shot Counts | Avg #Words                                   | Language     |
|--------------------------------|----------------------------------|-|--------------------------------------------------|-|-|----------------------------------------------|--------------|
| Babilong                       | `Eval Suite Long Context`        | Text Generation, Long Context | Completion, Long Context                         | ? | not supported | 22003                                        | en           |
| InfiniteBench_CodeDebug        | `InfiniteBench_CodeDebug`        | LogicalReasoning | Programming                                      | ? | not supported | 127761                                       | en           |
| InfiniteBench_CodeRun          | `InfiniteBench_CodeRun`          | LogicalReasoning | Programming                                      | ? | not supported | 34851                                        | en           |
| InfiniteBench_EnDia            | `InfiniteBench_EnDia`            | TextDistillation | Closed QA                                        | ? | not supported | 73240                                        | en           |
| InfiniteBench_EnMC             | `InfiniteBench_EnMC`             | TextDistillation | Closed QA                                        | ? | not supported | 139966                                       | en           |
| InfiniteBench_EnQA             | `InfiniteBench_EnQA`             | TextDistillation | Closed QA                                        | ? | not supported | 149442                                       | en           |
| InfiniteBench_MathFind         | `InfiniteBench_MathFind`         | LogicalReasoning | Math                                             | ? | not supported | 30017                                        | en           |
| InfiniteBench_RetrieveKV2      | `InfiniteBench_RetrieveKV2`      | TextDistillation | Extraction                                       | ? | not supported | 5010                                         | en           |
| InfiniteBench_RetrieveNumber   | `InfiniteBench_RetrieveNumber`   | TextDistillation | Extraction                                       | ? | not supported | 99199                                        | en           |
| InfiniteBench_RetrievePassKey1 | `InfiniteBench_RetrievePassKey1` | TextDistillation| Extraction                                       | ? | not supported | 99196                                        | en           |
| QuALITY                        | `QuALITY`                        | Text Distillation | QA                                               | Literature, Misc | not supported | 4248                                         | en           |
| ZeroSCROLLS GovReport          | `ZeroSCROLLS GovReport`          | Text Distillation | QA                                               | Government | not supported | 7273                                         | en           |
| ZeroSCROLLS SQuALITY           | `ZeroSCROLLS SQuALITY`           | Text Distillation | QB-Summ?                                         | Literature | not supported | 4971                                         | en           |
| ZeroSCROLLS Qasper             | `ZeroSCROLLS Qasper`             | Text Distillation | QA                                               | Science | not supported | 3531                                         | en           |
| ZeroSCROLLS NarrativeQA        | `ZeroSCROLLS NarrativeQA`        | Text Distillation | QA                                               | Literature, Film | not supported | 49384                                        | en           |
| ZeroSCROLLS QuALITY            | `ZeroSCROLLS QuALITY`            | Text Distillation | QA                                               | Literature, Misc | not supported | 4248                                         | en           |
| ZeroSCROLLS MuSiQue            | `ZeroSCROLLS MuSiQue`            | Text Distillation | QA                                               | Wikipedia | not supported | 1749                                         | en           |
| ZeroSCROLLS SpaceDigest        | `ZeroSCROLLS SpaceDigest`        | Text Distillation | Aggregation                                      | Reviews | not supported | 5481                                         | en           |

# Languages

- Languages in Likelihood tasks: ENG (39), DEU (7), FRA (4), FIN (2), NLD (2), ITA (1), POL (1), RUS (1), SPA (1), SWE (1), UKR (1)
- Languages in Completion tasks: ENG (20), DEU (5), FRA (5), ARB (1), FIN (1), ITA (1), POR (1), SPA (1) and 44 languages in INCLUDE.
- Languages in both types of tasks: ENG (59), DEU (12), FRA (9), FIN (3), NLD (2), SPA (2), ARB (1), ITA (1), POL (1), POR (1), RUS (1), SWE (1), UKR (1) and 44 languages in INCLUDE.

# Metrics

| Metrics Type | Metrics                       |
|-|-------------------------------|
| Completion Metrics | Accuracy
|| Bleu                          |
|| Chrf                          |
|| Ter                           |
|| F1                            |
|| Rouge 1                       |
|| Rouge 2                       |
|| Rouge-L                       |
|| Code Assertion                |
|| Language Checker              |
|| Length Checker                |
|| Math Reasoning                |
|| Placeholder Checker           |
|| Text Counter                  |
|| CSV Format                    |
|| JSON Format                   |
|| Postscript Format             |
|| Custom IFEval Checker         |
|| Custom CWE Checker            |
|| Custom NIAH Checker           |
|| Custom Grid Comparison Checker |
|| Repetition Checker            |
| Loglikelihood Metrics | Accuracy Loglikelihood        |
|| Normalized Accuracy Loglikelihood |
|| Probability Mass              |
| LLM Metrics | Chatbot Style Judge           |
|| Completion Accuracy Judge
|| Conciseness Judge
|| Contains Names Judge
|| Instruction Judge
|| SQL Format
|| World Knowledge Judge
| Efficiency Metrics | Bytes per Sequence Position   |
