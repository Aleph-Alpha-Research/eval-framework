from enum import Enum

from eval_framework.benchmarks.arc_de import ARC_DE_BENCHMARK
from eval_framework.benchmarks.csqa_ellamind import CSQA_ELLAMIND_BENCHMARKS
from eval_framework.benchmarks.gpqa_ellamind import GPQA_ELLAMIND_BENCHMARKS
from eval_framework.benchmarks.piqa_ellamind import PIQA_ELLAMIND_BENCHMARKS
from eval_framework.benchmarks.simpleqa_ellamind import SIMPLEQA_ELLAMIND_BENCHMARKS
from eval_framework.benchmarks.siqa_ellamind import SIQA_ELLAMIND_BENCHMARKS
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
    register_arc_ellamind_tasks(registry=registry)
    register_csqa_ellamind_tasks(registry=registry)
    register_gpqa_ellamind_tasks(registry=registry)
    register_gsm8k_ellamind_tasks(registry=registry)
    register_hellaswag_ellamind_tasks(registry=registry)
    register_hendrycks_math_ellamind_tasks(registry=registry)
    register_hle_ellamind_tasks(registry=registry)
    register_humaneval_ellamind_tasks(registry=registry)
    register_mbpp_ellamind_tasks(registry=registry)
    register_piqa_ellamind_tasks(registry=registry)
    register_simpleqa_ellamind_tasks(registry=registry)
    register_siqa_ellamind_tasks(registry=registry)
    register_winogrande_ellamind_tasks(registry=registry)


def register_arc_tasks(registry: Registry) -> None:
    """Register arc benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_IDK", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.arc.ARC_OLMES", registry=registry)


def register_hellaswag_tasks(registry: Registry) -> None:
    """Register hellaswag benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HELLASWAG", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.hellaswag.HELLASWAG_OLMES", registry=registry)


def register_piqa_tasks(registry: Registry) -> None:
    """Register piqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_IDK", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.piqa.PIQA_OLMES", registry=registry)


def register_gpqa_tasks(registry: Registry) -> None:
    """Register gpqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.gpqa.GPQA_DIAMOND_COT", registry=registry)


def register_gsm8k_tasks(registry: Registry) -> None:
    """Register gsm8k benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8K_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k.GSM8KBPB", registry=registry)


def register_math_reasoning_tasks(registry: Registry) -> None:
    """Register math_reasoning benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2024", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2026", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.AIME2025", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHMinervaBPB", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.GSM8KReasoning", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATH500", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHMinerva_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.math_reasoning.MATHMinerva_OLMES_NONL", registry=registry)


def register_mmlu_tasks(registry: Registry) -> None:
    """Register mmlu benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_IDK", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.FullTextMMLU", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu.MMLU_COT", registry=registry)


def register_humaneval_tasks(registry: Registry) -> None:
    """Register humaneval benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEvalBPB", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEvalBPB_V2", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEval_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval.HumanEval_OLMES_V2", registry=registry)


def register_mbpp_tasks(registry: Registry) -> None:
    """Register mbpp benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPPBPB", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_EvalPlus", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp.MBPP_BPB_EvalPlus", registry=registry)


def register_bigcodebench_tasks(registry: Registry) -> None:
    """Register bigcodebench benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.bigcodebench.BigCodeBench_OLMES", registry=registry)


def register_arc_de_tasks(registry: Registry) -> None:
    """Register arc_de benchmark tasks."""
    registry.add(ARC_DE_BENCHMARK)


def register_copa_tasks(registry: Registry) -> None:
    """Register copa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.copa.COPA_OLMES", registry=registry)


def register_goldenswag_tasks(registry: Registry) -> None:
    """Register goldenswag benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.goldenswag.GOLDENSWAG", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.goldenswag.GOLDENSWAG_IDK", registry=registry)


def register_ifeval_tasks(registry: Registry) -> None:
    """Register ifeval benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEval", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.ifeval.IFEvalDe", registry=registry)


def register_multipl_e_tasks(registry: Registry) -> None:
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


def register_mmlu_pro_tasks(registry: Registry) -> None:
    """Register mmlu_pro benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_IDK", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mmlu_pro.MMLU_PRO_COT", registry=registry)


def register_global_mmlu_tasks(registry: Registry) -> None:
    """Register global_mmlu benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.global_mmlu.GlobalMMLU", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.global_mmlu.GlobalMMLU_German", registry=registry)


def register_sciq_tasks(registry: Registry) -> None:
    """Register sciq benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.sciq.SCIQ_OLMES", registry=registry)


def register_squad_tasks(registry: Registry) -> None:
    """Register squad benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD2_MA", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.squad.SQuAD2_MA_NO_SYSPROMPT", registry=registry)


def register_winogrande_tasks(registry: Registry) -> None:
    """Register winogrande benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.winogrande.WINOGRANDECloze", registry=registry)


def register_csqa_tasks(registry: Registry) -> None:
    """Register csqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.csqa.CommonsenseQAMC_OLMES", registry=registry)


def register_drop_tasks(registry: Registry) -> None:
    """Register drop benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.drop.DropCompletion_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.drop.DropMC_OLMES", registry=registry)


def register_naturalqs_open_tasks(registry: Registry) -> None:
    """Register naturalqs_open benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.naturalqs_open.NaturalQsOpen", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.naturalqs_open.NaturalQsOpenMC_OLMES", registry=registry)


def register_social_iqa_tasks(registry: Registry) -> None:
    """Register social_iqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.social_iqa.SocialIQAMC_OLMES", registry=registry)


def register_medqa_tasks(registry: Registry) -> None:
    """Register medqa benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.medqa.MedQAMC_OLMES", registry=registry)


def register_arc_ellamind_tasks(registry: Registry) -> None:
    """Register arc_ellamind benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.arc_ellamind.ARC_ELLAMIND_CLOZE_DE", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.arc_ellamind.ARC_ELLAMIND_MC_DE", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.arc_ellamind.ARC_ELLAMIND_BPB_DE", registry=registry)


def register_csqa_ellamind_tasks(registry: Registry) -> None:
    """Register csqa_ellamind benchmark tasks."""
    for benchmark in CSQA_ELLAMIND_BENCHMARKS:
        registry.add(benchmark)


def register_gpqa_ellamind_tasks(registry: Registry) -> None:
    """Register gpqa_ellamind benchmark tasks."""
    for benchmark in GPQA_ELLAMIND_BENCHMARKS:
        registry.add(benchmark)


def register_gsm8k_ellamind_tasks(registry: Registry) -> None:
    """Register gsm8k_ellamind benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.gsm8k_ellamind.GSM8K_Ellamind_DE_Platinum", registry=registry)
    register_lazy_task(
        "eval_framework.tasks.benchmarks.gsm8k_ellamind.GSM8K_Ellamind_DE_BPB_Platinum", registry=registry
    )


def register_hellaswag_ellamind_tasks(registry: Registry) -> None:
    """Register hellaswag_ellamind benchmark tasks."""
    register_lazy_task(
        "eval_framework.tasks.benchmarks.hellaswag_ellamind.HELLASWAG_ELLAMIND_EASY_DE", registry=registry
    )
    register_lazy_task(
        "eval_framework.tasks.benchmarks.hellaswag_ellamind.HELLASWAG_ELLAMIND_HARD_DE", registry=registry
    )
    register_lazy_task(
        "eval_framework.tasks.benchmarks.hellaswag_ellamind.HELLASWAG_ELLAMIND_BPB_DE", registry=registry
    )


def register_hendrycks_math_ellamind_tasks(registry: Registry) -> None:
    """Register hendrycks_math_ellamind benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.hendrycks_math_ellamind.MATHMinervaDE_OLMES", registry=registry)
    register_lazy_task(
        "eval_framework.tasks.benchmarks.hendrycks_math_ellamind.MATHMinervaDE_BPB_OLMES", registry=registry
    )
    register_lazy_task(
        "eval_framework.tasks.benchmarks.hendrycks_math_ellamind.MATHMinervaDE_OLMES_NONL", registry=registry
    )


def register_hle_ellamind_tasks(registry: Registry) -> None:
    """Register hle_ellamind benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.hle_ellamind.HLE_ELLAMIND_MC_DE", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.hle_ellamind.HLE_ELLAMIND_CLOZE_DE", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.hle_ellamind.HLE_ELLAMIND_MC_NATIVE_DE", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.hle_ellamind.HLE_ELLAMIND_CLOZE_NATIVE_DE", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.hle_ellamind.HLE_ELLAMIND_BPB_DE", registry=registry)


def register_humaneval_ellamind_tasks(registry: Registry) -> None:
    """Register humaneval_ellamind benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval_ellamind.HumanEvalDE_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval_ellamind.HumanEvalDE_BPB_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval_ellamind.HumanEvalDE_BPB_OLMES_V2", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.humaneval_ellamind.HumanEvalDE_OLMES_V2", registry=registry)


def register_mbpp_ellamind_tasks(registry: Registry) -> None:
    """Register mbpp_ellamind benchmark tasks."""
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp_ellamind.MBPPDE_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp_ellamind.MBPPDE_BPB_OLMES", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp_ellamind.MBPPDE_EvalPlus", registry=registry)
    register_lazy_task("eval_framework.tasks.benchmarks.mbpp_ellamind.MBPPDE_BPB_EvalPlus", registry=registry)


def register_piqa_ellamind_tasks(registry: Registry) -> None:
    """Register piqa_ellamind benchmark tasks."""
    for benchmark in PIQA_ELLAMIND_BENCHMARKS:
        registry.add(benchmark)


def register_simpleqa_ellamind_tasks(registry: Registry) -> None:
    """Register simpleqa_ellamind benchmark tasks."""
    for benchmark in SIMPLEQA_ELLAMIND_BENCHMARKS:
        registry.add(benchmark)


def register_siqa_ellamind_tasks(registry: Registry) -> None:
    """Register siqa_ellamind benchmark tasks."""
    for benchmark in SIQA_ELLAMIND_BENCHMARKS:
        registry.add(benchmark)


def register_winogrande_ellamind_tasks(registry: Registry) -> None:
    """Register winogrande_ellamind benchmark tasks."""
    register_lazy_task(
        "eval_framework.tasks.benchmarks.winogrande_ellamind.WINOGRANDE_ELLAMIND_CLOZE_DE", registry=registry
    )
    register_lazy_task(
        "eval_framework.tasks.benchmarks.winogrande_ellamind.WINOGRANDE_ELLAMIND_MC_DE", registry=registry
    )
    register_lazy_task(
        "eval_framework.tasks.benchmarks.winogrande_ellamind.WINOGRANDE_ELLAMIND_PARTIAL_EVAL_DE", registry=registry
    )
