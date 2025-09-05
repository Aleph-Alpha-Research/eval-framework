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
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2024")
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC")
    register_lazy_task("eval_framework.tasks.benchmarks.arc_de.ARC_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.ARC_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.ARC_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.arc_fi.ARC_FI")
    register_lazy_task("eval_framework.tasks.benchmarks.belebele.BELEBELE")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBench")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBenchInstruct")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBenchHard")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBenchHardInstruct")
    register_lazy_task("eval_framework.tasks.benchmarks.casehold.CASEHOLD")
    register_lazy_task("eval_framework.tasks.benchmarks.chembench.ChemBench")
    register_lazy_task("eval_framework.tasks.benchmarks.copa.COPA")
    register_lazy_task("eval_framework.tasks.benchmarks.duc.DUC_ABSTRACTIVE")
    register_lazy_task("eval_framework.tasks.benchmarks.duc.DUC_EXTRACTIVE")
    register_lazy_task("eval_framework.tasks.benchmarks.flores200.Flores200")
    register_lazy_task("eval_framework.tasks.benchmarks.flores_plus.FloresPlus", extras=["comet"])
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA")
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8K")
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8KLlamaVersion")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.GSM8KReasoning")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.GSM8K_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.GSM8K_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HELLASWAG")
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag_de.HELLASWAG_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.HELLASWAG_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.HELLASWAG_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEval")
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEvalInstruct")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEval")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEvalDe")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEvalFiSv")
    register_lazy_task("eval_framework.tasks.benchmarks.include.INCLUDE")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_CodeDebug")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_CodeRun")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_EnDia")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_EnMC")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_EnQA")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_MathFind")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_RetrieveKV2")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_RetrieveNumber")
    register_lazy_task("eval_framework.tasks.benchmarks.infinitebench.InfiniteBench_RetrievePassKey1")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATH")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHLvl5")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATH500")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_SANITIZED")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_PROMPT_WITHOUT_TESTS")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_PROMPT_WITHOUT_TESTS_SANITIZED")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.FullTextMMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.MMLU_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.MMLU_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_de.MMLU_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.mmmlu.MMMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.mmmlu.MMMLU_GERMAN_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.pawsx.PAWSX")
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA")
    register_lazy_task("eval_framework.tasks.benchmarks.openbookqa.OPENBOOKQA")
    register_lazy_task("eval_framework.tasks.benchmarks.sciq.SCIQ")
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQUAD")
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQUAD2")
    register_lazy_task("eval_framework.tasks.benchmarks.tablebench.TableBench")
    register_lazy_task("eval_framework.tasks.benchmarks.triviaqa.TRIVIAQA")
    register_lazy_task("eval_framework.tasks.benchmarks.truthfulqa.TRUTHFULQA")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.TRUTHFULQA_EU20_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.opengptx_eu20.TRUTHFULQA_EU20_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.winogender.WINOGENDER")
    register_lazy_task("eval_framework.tasks.benchmarks.winogrande.WINOGRANDE")
    register_lazy_task("eval_framework.tasks.benchmarks.winox.WINOX_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.winox.WINOX_FR")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT14")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT16")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT20")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT14_INSTRUCT")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT16_INSTRUCT")
    register_lazy_task("eval_framework.tasks.benchmarks.wmt.WMT20_INSTRUCT")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_QUALITY")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_SQUALITY")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_QMSUM")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_QASPER")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_GOV_REPORT")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_NARRATIVEQA")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_MUSIQUE")
    register_lazy_task("eval_framework.tasks.benchmarks.zero_scrolls.ZERO_SCROLLS_SPACE_DIGEST")
    register_lazy_task("eval_framework.tasks.benchmarks.quality.QUALITY")
    register_lazy_task("eval_framework.tasks.benchmarks.sphyr.SPHYR")
    register_lazy_task("eval_framework.tasks.benchmarks.struct_eval.StructEval")
    register_lazy_task("eval_framework.tasks.benchmarks.struct_eval.RenderableStructEval")

    try:
        # Importing the companion registers the additional tasks from the module.
        # This is mostly for convenience for internal use-cases
        import eval_framework_companion  # noqa
    except ImportError:
        pass


def make_sure_all_hf_datasets_are_in_cache() -> None:
    for task_name, task_class in registered_tasks_iter():
        task = task_class(0, None, None)
        for attempt in range(10):
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
