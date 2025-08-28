import logging
import random
import time
from enum import Enum

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.benchmarks.arc import ARC
from eval_framework.tasks.benchmarks.arc_de import ARC_DE
from eval_framework.tasks.benchmarks.arc_fi import ARC_FI
from eval_framework.tasks.benchmarks.belebele import BELEBELE
from eval_framework.tasks.benchmarks.bigcodebench import (
    BigCodeBench,
    BigCodeBenchHard,
    BigCodeBenchHardInstruct,
    BigCodeBenchInstruct,
)
from eval_framework.tasks.benchmarks.casehold import CASEHOLD
from eval_framework.tasks.benchmarks.chembench import ChemBenchMultipleChoice
from eval_framework.tasks.benchmarks.copa import COPA
from eval_framework.tasks.benchmarks.duc import DUC_ABSTRACTIVE, DUC_EXTRACTIVE
from eval_framework.tasks.benchmarks.flores200 import Flores200

# from eval_framework.tasks.benchmarks.flores_plus import FloresPlus
from eval_framework.tasks.benchmarks.gpqa import GPQA, GPQA_COT
from eval_framework.tasks.benchmarks.gsm8k import GSM8K, GSM8KLlamaVersion
from eval_framework.tasks.benchmarks.hellaswag import HELLASWAG
from eval_framework.tasks.benchmarks.hellaswag_de import HELLASWAG_DE
from eval_framework.tasks.benchmarks.humaneval import HumanEval, HumanEvalInstruct
from eval_framework.tasks.benchmarks.ifeval import IFEval, IFEvalDe, IFEvalFiSv
from eval_framework.tasks.benchmarks.include import INCLUDE
from eval_framework.tasks.benchmarks.infinitebench import (
    InfiniteBench_CodeDebug,
    InfiniteBench_CodeRun,
    InfiniteBench_EnDia,
    InfiniteBench_EnMC,
    InfiniteBench_EnQA,
    InfiniteBench_MathFind,
    InfiniteBench_RetrieveKV2,
    InfiniteBench_RetrieveNumber,
    InfiniteBench_RetrievePassKey1,
)
from eval_framework.tasks.benchmarks.math_reasoning import (
    AIME2024,
    MATH,
    MATH500,
    GSM8KReasoning,
    MATHLvl5,
)
from eval_framework.tasks.benchmarks.mbpp import (
    MBPP,
    MBPP_PROMPT_WITHOUT_TESTS,
    MBPP_PROMPT_WITHOUT_TESTS_SANITIZED,
    MBPP_SANITIZED,
)
from eval_framework.tasks.benchmarks.mmlu import MMLU, MMLU_COT, FullTextMMLU
from eval_framework.tasks.benchmarks.mmlu_de import MMLU_DE
from eval_framework.tasks.benchmarks.mmlu_pro import MMLU_PRO, MMLU_PRO_COT
from eval_framework.tasks.benchmarks.mmmlu import MMMLU, MMMLU_GERMAN_COT
from eval_framework.tasks.benchmarks.openbookqa import OPENBOOKQA
from eval_framework.tasks.benchmarks.opengptx_eu20 import (
    ARC_EU20_DE,
    ARC_EU20_FR,
    GSM8K_EU20_DE,
    GSM8K_EU20_FR,
    HELLASWAG_EU20_DE,
    HELLASWAG_EU20_FR,
    MMLU_EU20_DE,
    MMLU_EU20_FR,
    TRUTHFULQA_EU20_DE,
    TRUTHFULQA_EU20_FR,
)
from eval_framework.tasks.benchmarks.pawsx import PAWSX
from eval_framework.tasks.benchmarks.piqa import PIQA
from eval_framework.tasks.benchmarks.quality import QUALITY
from eval_framework.tasks.benchmarks.sciq import SCIQ
from eval_framework.tasks.benchmarks.sphyr import SPHYR
from eval_framework.tasks.benchmarks.squad import SQUAD, SQUAD2
from eval_framework.tasks.benchmarks.struct_eval import RenderableStructEval, StructEval
from eval_framework.tasks.benchmarks.tablebench import TableBench
from eval_framework.tasks.benchmarks.triviaqa import TRIVIAQA
from eval_framework.tasks.benchmarks.truthfulqa import TRUTHFULQA
from eval_framework.tasks.benchmarks.winogender import WINOGENDER
from eval_framework.tasks.benchmarks.winogrande import WINOGRANDE
from eval_framework.tasks.benchmarks.winox import WINOX_DE, WINOX_FR
from eval_framework.tasks.benchmarks.wmt import WMT14, WMT14_INSTRUCT, WMT16, WMT16_INSTRUCT, WMT20, WMT20_INSTRUCT
from eval_framework.tasks.benchmarks.zero_scrolls import (
    ZERO_SCROLLS_GOV_REPORT,
    ZERO_SCROLLS_MUSIQUE,
    ZERO_SCROLLS_NARRATIVEQA,
    ZERO_SCROLLS_QASPER,
    ZERO_SCROLLS_QMSUM,
    ZERO_SCROLLS_QUALITY,
    ZERO_SCROLLS_SPACE_DIGEST,
    ZERO_SCROLLS_SQUALITY,
)

logger = logging.getLogger(__name__)


class TaskNameEnum(Enum):
    @property
    def value(self) -> type[BaseTask]:
        return super().value


class TaskName(TaskNameEnum):
    AIME2024 = AIME2024
    ARC = ARC
    ARC_DE = ARC_DE
    ARC_EU20_DE = ARC_EU20_DE
    ARC_EU20_FR = ARC_EU20_FR
    ARC_FI = ARC_FI
    BELEBELE = BELEBELE
    BIG_CODE_BENCH = BigCodeBench
    BIG_CODE_BENCH_INSTRUCT = BigCodeBenchInstruct
    BIG_CODE_BENCH_HARD = BigCodeBenchHard
    BIG_CODE_BENCH_HARD_INSTRUCT = BigCodeBenchHardInstruct
    CASEHOLD = CASEHOLD
    CHEM_BENCH_MULTIPLE_CHOICE = ChemBenchMultipleChoice
    COPA = COPA
    DUC_ABSTRACTIVE = DUC_ABSTRACTIVE
    DUC_EXTRACTIVE = DUC_EXTRACTIVE
    FLORES200 = Flores200
    # FLORES_PLUS = FloresPlus
    GPQA = GPQA
    GPQA_COT = GPQA_COT
    GSM8K = GSM8K
    GSM8K_LLAMA_VERSION = GSM8KLlamaVersion
    GSM8KReasoning = GSM8KReasoning
    GSM8K_EU20_DE = GSM8K_EU20_DE
    GSM8K_EU20_FR = GSM8K_EU20_FR
    HELLASWAG = HELLASWAG
    HELLASWAG_DE = HELLASWAG_DE
    HELLASWAG_EU20_DE = HELLASWAG_EU20_DE
    HELLASWAG_EU20_FR = HELLASWAG_EU20_FR
    HUMAN_EVAL = HumanEval
    HUMAN_EVAL_INSTRUCT = HumanEvalInstruct
    IFEVAL = IFEval
    IFEVAL_DE = IFEvalDe
    IFEVAL_FI_SV = IFEvalFiSv
    INCLUDE = INCLUDE
    INFINITE_BENCH_CODE_DEBUG = InfiniteBench_CodeDebug
    INFINITE_BENCH_CODE_RUN = InfiniteBench_CodeRun
    INFINITE_BENCH_EN_DIA = InfiniteBench_EnDia
    INFINITE_BENCH_EN_MC = InfiniteBench_EnMC
    INFINITE_BENCH_EN_QA = InfiniteBench_EnQA
    INFINITE_BENCH_MATH_FIND = InfiniteBench_MathFind
    INFINITE_BENCH_RETRIEVE_KV2 = InfiniteBench_RetrieveKV2
    INFINITE_BENCH_RETRIEVE_NUMBER = InfiniteBench_RetrieveNumber
    INFINITE_BENCH_RETRIEVE_PASSKEY1 = InfiniteBench_RetrievePassKey1
    MATH = MATH
    MATHLvl5 = MATHLvl5
    MATH500 = MATH500
    MBPP = MBPP
    MBPP_SANITIZED = MBPP_SANITIZED
    MBPP_PROMPT_WITHOUT_TESTS = MBPP_PROMPT_WITHOUT_TESTS
    MBPP_PROMPT_WITHOUT_TESTS_SANITIZED = MBPP_PROMPT_WITHOUT_TESTS_SANITIZED
    MMLU = MMLU
    FULL_TEXT_MMLU = FullTextMMLU
    MMLU_EU20_DE = MMLU_EU20_DE
    MMLU_EU20_FR = MMLU_EU20_FR
    MMLU_DE = MMLU_DE
    MMLU_PRO = MMLU_PRO
    MMLU_PRO_COT = MMLU_PRO_COT
    MMLU_COT = MMLU_COT
    MMMLU = MMMLU
    MMMLU_GERMAN_COT = MMMLU_GERMAN_COT
    PAWSX = PAWSX
    PIQA = PIQA
    OPENBOOKQA = OPENBOOKQA
    SCIQ = SCIQ
    SQUAD = SQUAD
    SQUAD2 = SQUAD2
    TABLEBENCH = TableBench
    TRIVIAQA = TRIVIAQA
    TRUTHFULQA = TRUTHFULQA
    TRUTHFULQA_EU20_DE = TRUTHFULQA_EU20_DE
    TRUTHFULQA_EU20_FR = TRUTHFULQA_EU20_FR
    WINOGENDER = WINOGENDER
    WINOGRANDE = WINOGRANDE
    WINOX_DE = WINOX_DE
    WINOX_FR = WINOX_FR
    WMT14 = WMT14
    WMT16 = WMT16
    WMT20 = WMT20
    WMT14_INSTRUCT = WMT14_INSTRUCT
    WMT16_INSTRUCT = WMT16_INSTRUCT
    WMT20_INSTRUCT = WMT20_INSTRUCT
    ZERO_SCROLLS_QUALITY = ZERO_SCROLLS_QUALITY
    ZERO_SCROLLS_SQUALITY = ZERO_SCROLLS_SQUALITY
    ZERO_SCROLLS_QMSUM = ZERO_SCROLLS_QMSUM
    ZERO_SCROLLS_QASPER = ZERO_SCROLLS_QASPER
    ZERO_SCROLLS_GOV_REPORT = ZERO_SCROLLS_GOV_REPORT
    ZERO_SCROLLS_NARRATIVEQA = ZERO_SCROLLS_NARRATIVEQA
    ZERO_SCROLLS_MUSIQUE = ZERO_SCROLLS_MUSIQUE
    ZERO_SCROLLS_SPACE_DIGEST = ZERO_SCROLLS_SPACE_DIGEST
    QUALITY = QUALITY
    SPHYR = SPHYR
    STRUCT_EVAL = StructEval
    RENDERABLE_STRUCT_EVAL = RenderableStructEval

    @classmethod
    def from_name(cls, name: str) -> "TaskName":
        name_upper = name.upper()
        matches = [task_name for task_name in cls if task_name.value.NAME.upper() == name_upper]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            raise ValueError(f"Ambiguous TaskName for name: {name} (multiple matches found)")
        else:
            raise ValueError(f"No TaskName found for name: {name}")

    @classmethod
    def _check_no_duplicate_names(cls) -> None:
        seen: dict = {}
        for task_name in cls:
            name_upper = task_name.value.NAME.upper()
            if name_upper in seen:
                raise ValueError(
                    f"Duplicate task name found (case-insensitive): {task_name.value.NAME} and {seen[name_upper]}"
                )
            seen[name_upper] = task_name.value.NAME


TaskName._check_no_duplicate_names()


def make_sure_all_hf_datasets_are_in_cache() -> None:
    for task_name in TaskName:
        task = task_name.value()
        for attempt in range(100):
            try:
                for _ in task.iterate_samples(num_samples=1):
                    pass
                break
            except Exception as e:
                logger.info(f"{e} Will retry loading {task_name.name} in a few seconds, attempt #{attempt + 1}.")
                time.sleep(random.randint(1, 5))
        logger.info(f"Processed {task_name.name}")
