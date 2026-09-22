import pytest

from eval_framework.tasks.benchmarks.drop import DropCloze, DropCompletion, DropMC, DropMC_OLMES
from eval_framework.tasks.benchmarks.humaneval import HumanEvalBPB, HumanEvalBPB_V2
from eval_framework.tasks.benchmarks.math_reasoning import (
    MATH500Minerva,
    MATHMinerva,
    MATHMinervaEvalHarness,
)
from eval_framework.tasks.benchmarks.mbpp import MBPPBPB
from eval_framework.tasks.benchmarks.naturalqs_open import (
    NaturalQsOpen,
    NaturalQsOpenCloze,
    NaturalQsOpenMC,
    NaturalQsOpenMC_OLMES,
)
from eval_framework.tasks.benchmarks.squad import SQUAD2BPB


def _smoke_test_task(task_cls, num_fewshot: int = 0) -> None:
    # Tasks must be built via `with_overwrite`, which is what seeds `task.rnd`.
    task = task_cls.with_overwrite(
        num_fewshot=num_fewshot,
        custom_subjects=None,
        custom_hf_revision=None,
    )
    samples = list(task.iterate_samples(num_samples=2))
    assert len(samples) > 0
    for sample in samples:
        assert sample.id is not None
        assert isinstance(sample.subject, str)
        assert sample.messages


@pytest.mark.cpu_slow
def test_drop_tasks_smoke() -> None:
    _smoke_test_task(DropCompletion)
    _smoke_test_task(DropMC)
    _smoke_test_task(DropMC_OLMES)
    _smoke_test_task(DropCloze)


@pytest.mark.cpu_slow
def test_naturalqs_open_tasks_smoke() -> None:
    _smoke_test_task(NaturalQsOpen)
    _smoke_test_task(NaturalQsOpenCloze)
    _smoke_test_task(NaturalQsOpenMC)
    _smoke_test_task(NaturalQsOpenMC_OLMES)


@pytest.mark.cpu_slow
@pytest.mark.slow_download
def test_math_minerva_tasks_smoke() -> None:
    _smoke_test_task(MATHMinervaEvalHarness)
    _smoke_test_task(MATHMinerva)
    _smoke_test_task(MATH500Minerva)


@pytest.mark.cpu_slow
def test_humaneval_bpb_smoke() -> None:
    _smoke_test_task(HumanEvalBPB)
    _smoke_test_task(HumanEvalBPB_V2)


@pytest.mark.cpu_slow
def test_mbpp_bpb_smoke() -> None:
    _smoke_test_task(MBPPBPB)


@pytest.mark.cpu_slow
def test_squad2_bpb_smoke() -> None:
    _smoke_test_task(SQUAD2BPB)
