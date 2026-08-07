from enum import Enum

from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.registry import Registry, register_lazy_task
from eval_framework.tasks.registry import registry as global_registry


class TaskNameEnum(Enum):
    @property
    def value(self) -> type[BaseTask]:
        return super().value


def register_all_tasks(registry: Registry | None = None) -> None:
    """Register all the benchmark tasks with the eval framework

    Uses global registry by default.
    """
    registry = registry if registry is not None else global_registry()

    register_math_reasoning_tasks(registry=registry)
    register_arc_tasks(registry=registry)
    register_arc_de_tasks(registry=registry)
    register_bigcodebench_tasks(registry=registry)
    register_copa_tasks(registry=registry)
    register_goldenswag_tasks(registry=registry)
    register_gpqa_tasks(registry=registry)
    register_gsm8k_tasks(registry=registry)
    register_hellaswag_tasks(registry=registry)
    register_humaneval_tasks(registry=registry)
    register_ifeval_tasks(registry=registry)
    register_multipl_e_tasks(registry=registry)
    register_mbpp_tasks(registry=registry)
    register_mmlu_tasks(registry=registry)
    register_mmlu_pro_tasks(registry=registry)
    register_global_mmlu_tasks(registry=registry)
    register_piqa_tasks(registry=registry)
    register_sciq_tasks(registry=registry)
    register_squad_tasks(registry=registry)
    register_winogrande_tasks(registry=registry)
    register_csqa_tasks(registry=registry)
    register_drop_tasks(registry=registry)
    register_naturalqs_open_tasks(registry=registry)
    register_social_iqa_tasks(registry=registry)
    register_medqa_tasks(registry=registry)


def register_arc_tasks(registry: Registry):
    """Register arc benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_IDK", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_OLMES", registry=registry)


def register_hellaswag_tasks(registry: Registry):
    """Register hellaswag benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HELLASWAG", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HELLASWAG_OLMES", registry=registry)


def register_piqa_tasks(registry: Registry):
    """Register piqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_IDK", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_OLMES", registry=registry)


def register_gpqa_tasks(registry: Registry):
    """Register gpqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA_DIAMOND_COT", registry=registry)


def register_gsm8k_tasks(registry: Registry):
    """Register gsm8k benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8K_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8KBPB", registry=registry)


def register_math_reasoning_tasks(registry: Registry):
    """Register math_reasoning benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2024", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2026", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2025", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHMinervaBPB", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.GSM8KReasoning", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATH500", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHMinerva_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHMinerva_OLMES_NONL", registry=registry)


def register_mmlu_tasks(registry: Registry):
    """Register mmlu benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_IDK", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.FullTextMMLU", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_COT", registry=registry)


def register_humaneval_tasks(registry: Registry):
    """Register humaneval benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEvalBPB", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEval_OLMES", registry=registry)


def register_mbpp_tasks(registry: Registry):
    """Register mbpp benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPPBPB", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_EvalPlus", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_BPB_EvalPlus", registry=registry)


def register_bigcodebench_tasks(registry: Registry):
    """Register bigcodebench benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBench_OLMES", registry=registry)


def register_arc_de_tasks(registry: Registry):
    """Register arc_de benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.arc_de.ARC_DE", registry=registry)


def register_copa_tasks(registry: Registry):
    """Register copa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.copa.COPA_OLMES", registry=registry)


def register_goldenswag_tasks(registry: Registry):
    """Register goldenswag benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.goldenswag.GOLDENSWAG", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.goldenswag.GOLDENSWAG_IDK", registry=registry)


def register_ifeval_tasks(registry: Registry):
    """Register ifeval benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEval", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEvalDe", registry=registry)


def register_multipl_e_tasks(registry: Registry):
    """Register multipl_e benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalCpp", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalJava", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalJs", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalPhp", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalRs", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEHumanEvalSh", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPCpp", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPJava", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPJs", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPPhp", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPRs", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.multipl_e.MultiPLEMBPPSh", registry=registry)


def register_mmlu_pro_tasks(registry: Registry):
    """Register mmlu_pro benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_IDK", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_COT", registry=registry)


def register_global_mmlu_tasks(registry: Registry):
    """Register global_mmlu benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.global_mmlu.GlobalMMLU", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.global_mmlu.GlobalMMLU_German", registry=registry)


def register_sciq_tasks(registry: Registry):
    """Register sciq benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.sciq.SCIQ_OLMES", registry=registry)


def register_squad_tasks(registry: Registry):
    """Register squad benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD2_MA", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD2_MA_NO_SYSPROMPT", registry=registry)


def register_winogrande_tasks(registry: Registry):
    """Register winogrande benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.winogrande.WINOGRANDECloze", registry=registry)


def register_csqa_tasks(registry: Registry):
    """Register csqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.csqa.CommonsenseQAMC_OLMES", registry=registry)


def register_drop_tasks(registry: Registry):
    """Register drop benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.drop.DropCompletion_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.drop.DropMC_OLMES", registry=registry)


def register_naturalqs_open_tasks(registry: Registry):
    """Register naturalqs_open benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.naturalqs_open.NaturalQsOpen", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.naturalqs_open.NaturalQsOpenMC_OLMES", registry=registry)


def register_social_iqa_tasks(registry: Registry):
    """Register social_iqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.social_iqa.SocialIQAMC_OLMES", registry=registry)


def register_medqa_tasks(registry: Registry):
    """Register medqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.medqa.MedQAMC_OLMES", registry=registry)
