from enum import Enum

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import register_lazy_task


class TaskNameEnum(Enum):
    @property
    def value(self) -> type[BaseTask]:
        return super().value


def register_all_tasks() -> None:
    """Register all the benchmark tasks with the eval framework."""
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2024")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2025")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2026")
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC")
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.arc_de.ARC_DE")
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBench_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.copa.COPA_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.goldenswag.GOLDENSWAG")
    register_lazy_task("eval_framework.tasks.benchmarks.goldenswag.GOLDENSWAG_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8K_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8KBPB")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHMinervaBPB")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.GSM8KReasoning")
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HELLASWAG")
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HELLASWAG_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEvalBPB")
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEval_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEval")
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEvalDe")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATH500")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHMinerva_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHMinerva_OLMES_NONL")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalCpp")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalJava")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalJs")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalPhp")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalRs")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalSh")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPCpp")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPJava")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPJs")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPPhp")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPRs")
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPSh")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPPBPB")
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.FullTextMMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_COT")
    register_lazy_task("eval_framework.tasks.benchmarks.global_mmlu.GlobalMMLU")
    register_lazy_task("eval_framework.tasks.benchmarks.global_mmlu.GlobalMMLU_German")
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA")
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_IDK")
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.sciq.SCIQ_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD2_MA")
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD2_MA_NO_SYSPROMPT")
    register_lazy_task("eval_framework.tasks.benchmarks.triviaqa.TRIVIAQA")
    register_lazy_task("eval_framework.tasks.benchmarks.triviaqa.TriviaQA_MA")
    register_lazy_task("eval_framework.tasks.benchmarks.winogrande.WINOGRANDECloze")
    register_lazy_task("eval_framework.tasks.benchmarks.csqa.CommonsenseQAMC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.drop.DropCompletion_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.drop.DropMC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.naturalqs_open.NaturalQsOpen")
    register_lazy_task("eval_framework.tasks.benchmarks.naturalqs_open.NaturalQsOpenMC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.social_iqa.SocialIQAMC_OLMES")
    register_lazy_task("eval_framework.tasks.benchmarks.medqa.MedQAMC_OLMES")
    try:
        # Importing the companion registers the additional tasks from the module.
        # This is mostly for convenience for internal use-cases
        import eval_framework_companion  # noqa
    except ImportError:
        pass
