import pytest
from datasets.exceptions import DatasetNotFoundError

from eval_framework.tasks.benchmarks.arc import ARC_OLMES
from eval_framework.tasks.benchmarks.balancedcopa import BalancedCOPA
from eval_framework.tasks.benchmarks.copa import COPA_OLMES, COPA_IDKEvalHarness, COPAEvalHarness
from eval_framework.tasks.benchmarks.csqa import (
    CommonsenseQACloze,
    CommonsenseQAFullTextCloze,
    CommonsenseQAMC,
    CommonsenseQAMC_OLMES,
)
from eval_framework.tasks.benchmarks.drop import DropCloze, DropCompletion, DropMC, DropMC_OLMES
from eval_framework.tasks.benchmarks.global_mmlu import GlobalMMLU
from eval_framework.tasks.benchmarks.goldenswag import GOLDENSWAG, GOLDENSWAG_IDK
from eval_framework.tasks.benchmarks.gpqa import GPQA_OLMES
from eval_framework.tasks.benchmarks.humaneval import HumanEvalBPB
from eval_framework.tasks.benchmarks.lab_bench import LabBenchCloze, LabBenchMC, LabBenchMC_OLMES
from eval_framework.tasks.benchmarks.math_reasoning import (
    MATH500Minerva,
    MATHMinerva,
    MATHMinervaBPB,
    MATHMinervaEvalHarness,
)
from eval_framework.tasks.benchmarks.mbpp import MBPPBPB
from eval_framework.tasks.benchmarks.medqa import MedQACloze, MedQAMC, MedQAMC_OLMES
from eval_framework.tasks.benchmarks.mmlu import MMLU_OLMES
from eval_framework.tasks.benchmarks.mmlu_pro import MMLU_PRO_OLMES
from eval_framework.tasks.benchmarks.naturalqs_open import (
    NaturalQsOpen,
    NaturalQsOpenCloze,
    NaturalQsOpenMC,
    NaturalQsOpenMC_OLMES,
)
from eval_framework.tasks.benchmarks.openbookqa import (
    OPENBOOKQA_EVAL_HARNESS_OLMES,
    OPENBOOKQA_OLMES,
)
from eval_framework.tasks.benchmarks.piqa import PIQA_OLMES
from eval_framework.tasks.benchmarks.sciq import SCIQ_IDK, SCIQ_OLMES, SCIQEvalHarness_IDK
from eval_framework.tasks.benchmarks.social_iqa import SocialIQACloze, SocialIQAMC, SocialIQAMC_OLMES
from eval_framework.tasks.benchmarks.squad import SQUAD2BPB
from eval_framework.tasks.benchmarks.truthfulqa import TRUTHFULQA_OLMES
from eval_framework.tasks.benchmarks.winogrande import WINOGRANDE_OLMES


def _smoke_test_task(task_cls) -> None:
    task = task_cls()
    samples = list(task.iterate_samples(num_samples=2))
    assert len(samples) > 0
    for sample in samples:
        assert sample.id is not None
        assert isinstance(sample.subject, str)
        assert sample.messages


@pytest.mark.cpu_slow
def test_csqa_tasks_smoke() -> None:
    _smoke_test_task(CommonsenseQACloze)
    _smoke_test_task(CommonsenseQAFullTextCloze)
    _smoke_test_task(CommonsenseQAMC)
    _smoke_test_task(CommonsenseQAMC_OLMES)


@pytest.mark.cpu_slow
def test_drop_tasks_smoke() -> None:
    _smoke_test_task(DropCompletion)
    _smoke_test_task(DropMC)
    _smoke_test_task(DropMC_OLMES)
    _smoke_test_task(DropCloze)


@pytest.mark.cpu_slow
def test_lab_bench_tasks_smoke() -> None:
    _smoke_test_task(LabBenchCloze)
    _smoke_test_task(LabBenchMC)
    _smoke_test_task(LabBenchMC_OLMES)


@pytest.mark.cpu_slow
def test_naturalqs_open_tasks_smoke() -> None:
    _smoke_test_task(NaturalQsOpen)
    _smoke_test_task(NaturalQsOpenCloze)
    _smoke_test_task(NaturalQsOpenMC)
    _smoke_test_task(NaturalQsOpenMC_OLMES)


@pytest.mark.cpu_slow
def test_math_minerva_tasks_smoke() -> None:
    _smoke_test_task(MATHMinervaEvalHarness)
    _smoke_test_task(MATHMinerva)
    _smoke_test_task(MATHMinervaBPB)
    _smoke_test_task(MATH500Minerva)


@pytest.mark.cpu_slow
def test_social_iqa_tasks_smoke() -> None:
    try:
        _smoke_test_task(SocialIQACloze)
        _smoke_test_task(SocialIQAMC)
        _smoke_test_task(SocialIQAMC_OLMES)
    except RuntimeError as e:
        if "no longer supported" in str(e) or "loading script" in str(e).lower():
            pytest.skip("allenai/social_i_qa uses a dataset loading script not supported by this datasets version")
        raise


@pytest.mark.cpu_slow
def test_medqa_tasks_smoke() -> None:
    _smoke_test_task(MedQACloze)
    _smoke_test_task(MedQAMC)
    _smoke_test_task(MedQAMC_OLMES)


@pytest.mark.cpu_slow
def test_olmes_variants_smoke() -> None:
    for task_cls in (
        ARC_OLMES,
        COPA_OLMES,
        GPQA_OLMES,  # gated; skipped when not authenticated
        MMLU_OLMES,
        MMLU_PRO_OLMES,
        OPENBOOKQA_OLMES,
        OPENBOOKQA_EVAL_HARNESS_OLMES,
        PIQA_OLMES,
        TRUTHFULQA_OLMES,
        WINOGRANDE_OLMES,
    ):
        try:
            _smoke_test_task(task_cls)
        except DatasetNotFoundError as e:
            if "gated" in str(e).lower():
                continue  # skip this task only when gated and not authenticated
            raise


@pytest.mark.cpu_slow
def test_balanced_copa_and_copa_harness_smoke() -> None:
    _smoke_test_task(BalancedCOPA)
    _smoke_test_task(COPAEvalHarness)
    _smoke_test_task(COPA_IDKEvalHarness)


@pytest.mark.cpu_slow
def test_goldenswag_tasks_smoke() -> None:
    _smoke_test_task(GOLDENSWAG)
    _smoke_test_task(GOLDENSWAG_IDK)


@pytest.mark.cpu_slow
def test_global_mmlu_smoke() -> None:
    _smoke_test_task(GlobalMMLU)


@pytest.mark.cpu_slow
def test_humaneval_bpb_smoke() -> None:
    _smoke_test_task(HumanEvalBPB)


@pytest.mark.cpu_slow
def test_mbpp_bpb_smoke() -> None:
    _smoke_test_task(MBPPBPB)


@pytest.mark.cpu_slow
def test_sciq_olmes_tasks_smoke() -> None:
    _smoke_test_task(SCIQ_OLMES)
    _smoke_test_task(SCIQ_IDK)
    _smoke_test_task(SCIQEvalHarness_IDK)


@pytest.mark.cpu_slow
def test_squad2_bpb_smoke() -> None:
    _smoke_test_task(SQUAD2BPB)
