import pytest

from eval_framework.tasks.benchmarks.csqa import CommonsenseQACloze, CommonsenseQAMC
from eval_framework.tasks.benchmarks.drop import DropMC, DropRC
from eval_framework.tasks.benchmarks.lab_bench import LabBenchCloze, LabBenchMC
from eval_framework.tasks.benchmarks.math_reasoning import MATH500Minerva, MATHMinerva, MATHMinervaBPB
from eval_framework.tasks.benchmarks.medqa import MedQACloze, MedQAMC
from eval_framework.tasks.benchmarks.naturalqs_open import (
    NaturalQsOpen,
    NaturalQsOpenCloze,
    NaturalQsOpenMC,
)
from eval_framework.tasks.benchmarks.social_iqa import SocialIQACloze, SocialIQAMC


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
    _smoke_test_task(CommonsenseQAMC)


@pytest.mark.cpu_slow
def test_drop_tasks_smoke() -> None:
    _smoke_test_task(DropRC)
    _smoke_test_task(DropMC)


@pytest.mark.cpu_slow
def test_lab_bench_tasks_smoke() -> None:
    _smoke_test_task(LabBenchCloze)
    _smoke_test_task(LabBenchMC)


@pytest.mark.cpu_slow
def test_naturalqs_open_tasks_smoke() -> None:
    _smoke_test_task(NaturalQsOpen)
    _smoke_test_task(NaturalQsOpenCloze)
    _smoke_test_task(NaturalQsOpenMC)


@pytest.mark.cpu_slow
def test_math_minerva_tasks_smoke() -> None:
    _smoke_test_task(MATHMinerva)
    _smoke_test_task(MATHMinervaBPB)
    _smoke_test_task(MATH500Minerva)


@pytest.mark.cpu_slow
def test_social_iqa_tasks_smoke() -> None:
    try:
        _smoke_test_task(SocialIQACloze)
        _smoke_test_task(SocialIQAMC)
    except RuntimeError as e:
        if "no longer supported" in str(e) or "loading script" in str(e).lower():
            pytest.skip("allenai/social_i_qa uses a dataset loading script not supported by this datasets version")
        raise


@pytest.mark.cpu_slow
def test_medqa_tasks_smoke() -> None:
    _smoke_test_task(MedQACloze)
    _smoke_test_task(MedQAMC)
