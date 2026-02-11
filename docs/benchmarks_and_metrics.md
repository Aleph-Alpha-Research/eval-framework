# Included Benchmark Tasks

Currently, the framework covers a wide range of pre-training and post-training benchmarks for completion and loglikelihood tasks, as well as benchmarks that use LLM-as-a-judge evaluation methods. The suggested few-shot counts are extracted from other leaderboards and literature.

Additional task documentation can be generated with the script `utils/generate-task-docs.py` as documented in [installation.md](installation.md). The documention can thereafter be found in [docs/tasks](tasks/README.md).

## Completion

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

## Loglikelihoods

|   **Task**    | **Capability** | **Benchmarks** | **Long Context** |
|-|-|-|-|
|   Output Control   |   Bias   |   `WINOGENDER`   |    |
|   Text Distillation   |   Classification   |   `ARC`,`ARC_DE`, `ARC_EU20_DE`, `ARC_EU20_FR`, `ARC_FI`, `BELEBELE`, `ChemBench`, `FullTextMMLU`, `GPQA`, `INCLUDE`, `MMLU`, `MMLU_DE`,`MMLU_EU20_DE`, `MMLU_EU20_FR`, `MMMLU`, `MMLU_PRO`, `OPENBOOKQA`, `PIQA`, `SCIQ`, `SCIQEvalHarness`, `TRUTHFULQA`, `TRUTHFULQA_EU20_DE, TRUTHFULQA_EU20_FR`   |    |
|   Text Distillation   |   QA   |    |   `QUALITY`, `ZERO_SCROLLS_QUALITY`   |
|   Text Generation   |   Open QA   |   `CASEHOLD`   |    |
|   Logical Reasoning   |   Closed QA   |    |   `InfiniteBench_EnMC`   |
|   Logical Reasoning   |   Programming   |    |   `InfiniteBench_CodeDebug`   |
|   Logical Reasoning   |   Reasoning   |   `BalancedCOPA`, `COPA`, `COPA_IDK`, `COPAEvalHarness`, `COPA_IDKEvalHarness`, `HELLASWAG, HELLASWAG_EU20_DE, HELLASWAG_EU20_FR`, `WINOGRANDE`, `WINOX_DE`, `WINOX_FR`   |    |

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

## Languages

- Languages in Likelihood tasks: ENG (39), DEU (7), FRA (4), FIN (2), NLD (2), ITA (1), POL (1), RUS (1), SPA (1), SWE (1), UKR (1)
- Languages in Completion tasks: ENG (20), DEU (5), FRA (5), ARB (1), FIN (1), ITA (1), POR (1), SPA (1) and 44 languages in INCLUDE.
- Languages in both types of tasks: ENG (59), DEU (12), FRA (9), FIN (3), NLD (2), SPA (2), ARB (1), ITA (1), POL (1), POR (1), RUS (1), SWE (1), UKR (1) and 44 languages in INCLUDE.

## Metrics

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
