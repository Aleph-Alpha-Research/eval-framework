import pytest

from eval_framework.tasks.benchmarks.humaneval import HumanEvalBPB, HumanEvalBPB_V2
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
def test_naturalqs_open_tasks_smoke() -> None:
    _smoke_test_task(NaturalQsOpen)
    _smoke_test_task(NaturalQsOpenCloze)
    _smoke_test_task(NaturalQsOpenMC)
    _smoke_test_task(NaturalQsOpenMC_OLMES)


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
