import logging
import random
import time
from enum import Enum

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import register_lazy_task, registered_tasks_iter

logger = logging.getLogger(__name__)


class TaskNameEnum(Enum):
    @property
    def value(self) -> type[BaseTask]:
        return super().value


def register_all_tasks() -> None:
    """Register all the benchmark tasks with the eval framework."""
    register_lazy_task("AIME2024", class_path="eval_framework.tasks.benchmarks.math_reasoning.AIME2024")
    register_lazy_task("ARC", class_path="eval_framework.tasks.benchmarks.arc.ARC")
    register_lazy_task("ARC_DE", class_path="eval_framework.tasks.benchmarks.arc_de.ARC_DE")
    register_lazy_task("ARC_EU20_DE", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.ARC_EU20_DE")
    register_lazy_task("ARC_EU20_FR", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.ARC_EU20_FR")
    register_lazy_task("ARC_FI", class_path="eval_framework.tasks.benchmarks.arc_fi.ARC_FI")
    register_lazy_task("BELEBELE", class_path="eval_framework.tasks.benchmarks.belebele.BELEBELE")
    register_lazy_task("BIG_CODE_BENCH", class_path="eval_framework.tasks.benchmarks.bigcodebench.BigCodeBench")
    register_lazy_task(
        "BIG_CODE_BENCH_INSTRUCT", class_path="eval_framework.tasks.benchmarks.bigcodebench.BigCodeBenchInstruct"
    )
    register_lazy_task(
        "BIG_CODE_BENCH_HARD", class_path="eval_framework.tasks.benchmarks.bigcodebench.BigCodeBenchHard"
    )
    register_lazy_task(
        "BIG_CODE_BENCH_HARD_INSTRUCT",
        class_path="eval_framework.tasks.benchmarks.bigcodebench.BigCodeBenchHardInstruct",
    )
    register_lazy_task("CASEHOLD", class_path="eval_framework.tasks.benchmarks.casehold.CASEHOLD")
    register_lazy_task(
        "CHEM_BENCH_MULTIPLE_CHOICE", class_path="eval_framework.tasks.benchmarks.chembench.ChemBenchMultipleChoice"
    )
    register_lazy_task("COPA", class_path="eval_framework.tasks.benchmarks.copa.COPA")
    register_lazy_task("DUC_ABSTRACTIVE", class_path="eval_framework.tasks.benchmarks.duc.DUC_ABSTRACTIVE")
    register_lazy_task("DUC_EXTRACTIVE", class_path="eval_framework.tasks.benchmarks.duc.DUC_EXTRACTIVE")
    register_lazy_task("FLORES200", class_path="eval_framework.tasks.benchmarks.flores200.Flores200")
    register_lazy_task("FLORES_PLUS", class_path="eval_framework.tasks.benchmarks.flores_plus.FloresPlus")
    register_lazy_task("GPQA", class_path="eval_framework.tasks.benchmarks.gpqa.GPQA")
    register_lazy_task("GPQA_COT", class_path="eval_framework.tasks.benchmarks.gpqa.GPQA_COT")
    register_lazy_task("GSM8K", class_path="eval_framework.tasks.benchmarks.gsm8k.GSM8K")
    register_lazy_task("GSM8K_LLAMA_VERSION", class_path="eval_framework.tasks.benchmarks.gsm8k.GSM8KLlamaVersion")
    register_lazy_task("GSM8KReasoning", class_path="eval_framework.tasks.benchmarks.math_reasoning.GSM8KReasoning")
    register_lazy_task("GSM8K_EU20_DE", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.GSM8K_EU20_DE")
    register_lazy_task("GSM8K_EU20_FR", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.GSM8K_EU20_FR")
    register_lazy_task("HELLASWAG", class_path="eval_framework.tasks.benchmarks.hellaswag.HELLASWAG")
    register_lazy_task("HELLASWAG_DE", class_path="eval_framework.tasks.benchmarks.hellaswag_de.HELLASWAG_DE")
    register_lazy_task(
        "HELLASWAG_EU20_DE", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.HELLASWAG_EU20_DE"
    )
    register_lazy_task(
        "HELLASWAG_EU20_FR", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.HELLASWAG_EU20_FR"
    )
    register_lazy_task("HUMAN_EVAL", class_path="eval_framework.tasks.benchmarks.humaneval.HumanEval")
    register_lazy_task("HUMAN_EVAL_INSTRUCT", class_path="eval_framework.tasks.benchmarks.humaneval.HumanEvalInstruct")
    register_lazy_task("IFEVAL", class_path="eval_framework.tasks.benchmarks.ifeval.IFEval")
    register_lazy_task("IFEVAL_DE", class_path="eval_framework.tasks.benchmarks.ifeval.IFEvalDe")
    register_lazy_task("IFEVAL_FI_SV", class_path="eval_framework.tasks.benchmarks.ifeval.IFEvalFiSv")
    register_lazy_task("INCLUDE", class_path="eval_framework.tasks.benchmarks.include.INCLUDE")
    register_lazy_task(
        "INFINITE_BENCH_CODE_DEBUG", class_path="eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_CodeDebug"
    )
    register_lazy_task(
        "INFINITE_BENCH_CODE_RUN", class_path="eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_CodeRun"
    )
    register_lazy_task(
        "INFINITE_BENCH_EN_DIA", class_path="eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_EnDia"
    )
    register_lazy_task(
        "INFINITE_BENCH_EN_MC", class_path="eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_EnMC"
    )
    register_lazy_task(
        "INFINITE_BENCH_EN_QA", class_path="eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_EnQA"
    )
    register_lazy_task(
        "INFINITE_BENCH_MATH_FIND", class_path="eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_MathFind"
    )
    register_lazy_task(
        "INFINITE_BENCH_RETRIEVE_KV2",
        class_path="eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_RetrieveKV2",
    )
    register_lazy_task(
        "INFINITE_BENCH_RETRIEVE_NUMBER",
        class_path="eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_RetrieveNumber",
    )
    register_lazy_task(
        "INFINITE_BENCH_RETRIEVE_PASSKEY1",
        class_path="eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_RetrievePassKey1",
    )
    register_lazy_task("MATH", class_path="eval_framework.tasks.benchmarks.math_reasoning.MATH")
    register_lazy_task("MATHLvl5", class_path="eval_framework.tasks.benchmarks.math_reasoning.MATHLvl5")
    register_lazy_task("MATH500", class_path="eval_framework.tasks.benchmarks.math_reasoning.MATH500")
    register_lazy_task("MBPP", class_path="eval_framework.tasks.benchmarks.mbpp.MBPP")
    register_lazy_task("MBPP_SANITIZED", class_path="eval_framework.tasks.benchmarks.mbpp.MBPP_SANITIZED")
    register_lazy_task(
        "MBPP_PROMPT_WITHOUT_TESTS", class_path="eval_framework.tasks.benchmarks.mbpp.MBPP_PROMPT_WITHOUT_TESTS"
    )
    register_lazy_task(
        "MBPP_PROMPT_WITHOUT_TESTS_SANITIZED",
        class_path="eval_framework.tasks.benchmarks.mbpp.MBPP_PROMPT_WITHOUT_TESTS_SANITIZED",
    )
    register_lazy_task("MMLU", class_path="eval_framework.tasks.benchmarks.mmlu.MMLU")
    register_lazy_task("FULL_TEXT_MMLU", class_path="eval_framework.tasks.benchmarks.mmlu.FullTextMMLU")
    register_lazy_task("MMLU_EU20_DE", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.MMLU_EU20_DE")
    register_lazy_task("MMLU_EU20_FR", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.MMLU_EU20_FR")
    register_lazy_task("MMLU_DE", class_path="eval_framework.tasks.benchmarks.mmlu_de.MMLU_DE")
    register_lazy_task("MMLU_PRO", class_path="eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO")
    register_lazy_task("MMLU_PRO_COT", class_path="eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_COT")
    register_lazy_task("MMLU_COT", class_path="eval_framework.tasks.benchmarks.mmlu.MMLU_COT")
    register_lazy_task("MMMLU", class_path="eval_framework.tasks.benchmarks.mmmlu.MMMLU")
    register_lazy_task("MMMLU_GERMAN_COT", class_path="eval_framework.tasks.benchmarks.mmmlu.MMMLU_GERMAN_COT")
    register_lazy_task("PAWSX", class_path="eval_framework.tasks.benchmarks.pawsx.PAWSX")
    register_lazy_task("PIQA", class_path="eval_framework.tasks.benchmarks.piqa.PIQA")
    register_lazy_task("OPENBOOKQA", class_path="eval_framework.tasks.benchmarks.openbookqa.OPENBOOKQA")
    register_lazy_task("SCIQ", class_path="eval_framework.tasks.benchmarks.sciq.SCIQ")
    register_lazy_task("SQUAD", class_path="eval_framework.tasks.benchmarks.squad.SQUAD")
    register_lazy_task("SQUAD2", class_path="eval_framework.tasks.benchmarks.squad.SQUAD2")
    register_lazy_task("TABLEBENCH", class_path="eval_framework.tasks.benchmarks.tablebench.TableBench")
    register_lazy_task("TRIVIAQA", class_path="eval_framework.tasks.benchmarks.triviaqa.TRIVIAQA")
    register_lazy_task("TRUTHFULQA", class_path="eval_framework.tasks.benchmarks.truthfulqa.TRUTHFULQA")
    register_lazy_task(
        "TRUTHFULQA_EU20_DE", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.TRUTHFULQA_EU20_DE"
    )
    register_lazy_task(
        "TRUTHFULQA_EU20_FR", class_path="eval_framework.tasks.benchmarks.opengptx_eu20.TRUTHFULQA_EU20_FR"
    )
    register_lazy_task("WINOGENDER", class_path="eval_framework.tasks.benchmarks.winogender.WINOGENDER")
    register_lazy_task("WINOGRANDE", class_path="eval_framework.tasks.benchmarks.winogrande.WINOGRANDE")
    register_lazy_task("WINOX_DE", class_path="eval_framework.tasks.benchmarks.winox.WINOX_DE")
    register_lazy_task("WINOX_FR", class_path="eval_framework.tasks.benchmarks.winox.WINOX_FR")
    register_lazy_task("WMT14", class_path="eval_framework.tasks.benchmarks.wmt.WMT14")
    register_lazy_task("WMT16", class_path="eval_framework.tasks.benchmarks.wmt.WMT16")
    register_lazy_task("WMT20", class_path="eval_framework.tasks.benchmarks.wmt.WMT20")
    register_lazy_task("WMT14_INSTRUCT", class_path="eval_framework.tasks.benchmarks.wmt.WMT14_INSTRUCT")
    register_lazy_task("WMT16_INSTRUCT", class_path="eval_framework.tasks.benchmarks.wmt.WMT16_INSTRUCT")
    register_lazy_task("WMT20_INSTRUCT", class_path="eval_framework.tasks.benchmarks.wmt.WMT20_INSTRUCT")
    register_lazy_task(
        "ZERO_SCROLLS_QUALITY", class_path="eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_QUALITY"
    )
    register_lazy_task(
        "ZERO_SCROLLS_SQUALITY", class_path="eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_SQUALITY"
    )
    register_lazy_task(
        "ZERO_SCROLLS_QMSUM", class_path="eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_QMSUM"
    )
    register_lazy_task(
        "ZERO_SCROLLS_QASPER", class_path="eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_QASPER"
    )
    register_lazy_task(
        "ZERO_SCROLLS_GOV_REPORT", class_path="eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_GOV_REPORT"
    )
    register_lazy_task(
        "ZERO_SCROLLS_NARRATIVEQA", class_path="eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_NARRATIVEQA"
    )
    register_lazy_task(
        "ZERO_SCROLLS_MUSIQUE", class_path="eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_MUSIQUE"
    )
    register_lazy_task(
        "ZERO_SCROLLS_SPACE_DIGEST", class_path="eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_SPACE_DIGEST"
    )
    register_lazy_task("QUALITY", class_path="eval_framework.tasks.benchmarks.quality.QUALITY")
    register_lazy_task("SPHYR", class_path="eval_framework.tasks.benchmarks.sphyr.SPHYR")
    register_lazy_task("STRUCT_EVAL", class_path="eval_framework.tasks.benchmarks.struct_eval.StructEval")
    register_lazy_task(
        "RENDERABLE_STRUCT_EVAL", class_path="eval_framework.tasks.benchmarks.struct_eval.RenderableStructEval"
    )


def make_sure_all_hf_datasets_are_in_cache() -> None:
    for task_name, task_class in registered_tasks_iter():
        task = task_class()
        for attempt in range(100):
            try:
                for _ in task.iterate_samples(num_samples=1):
                    pass
                break
            except Exception as e:
                logger.info(f"{e} Will retry loading {task_name} in a few seconds, attempt #{attempt + 1}.")
                time.sleep(random.randint(1, 5))
        logger.info(f"Processed {task_name}")


if __name__ == "__main__":
    print(list(registered_tasks_iter()))
