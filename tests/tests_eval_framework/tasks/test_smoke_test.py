from eval_framework.tasks.benchmarks.basic_skills import BasicSkillsCloze, BasicSkillsMC
from eval_framework.tasks.benchmarks.boolq import BoolQCloze, BoolQMC
from eval_framework.tasks.benchmarks.copycolors import CopyColorsCloze, CopyColorsMC
from eval_framework.tasks.benchmarks.coqa import CoQA, CoQAMC
from eval_framework.tasks.benchmarks.csqa import CommonsenseQACloze, CommonsenseQAMC
from eval_framework.tasks.benchmarks.drop import DropRC, DropMC
from eval_framework.tasks.benchmarks.jeopardy import JeopardyCompletion, JeopardyMC
from eval_framework.tasks.benchmarks.lab_bench import LabBenchCloze, LabBenchMC
from eval_framework.tasks.benchmarks.naturalqs_open import (
    NaturalQsOpen,
    NaturalQsOpenCloze,
    NaturalQsOpenMC,
)
from eval_framework.tasks.benchmarks.paloma import GenericPaloma
from eval_framework.tasks.benchmarks.popqa import PopQA
from eval_framework.tasks.benchmarks.qasper_yesno import QASPERYesNo, QASPERYesNoMC
from eval_framework.tasks.benchmarks.simpleqa import SimpleQACompletion
from eval_framework.tasks.benchmarks.simpletom import SimpleToMMC
from eval_framework.tasks.benchmarks.tydiqa import TyDiQASecondaryTask


def _smoke_test_task(task_cls) -> None:
    task = task_cls()
    samples = list(task.iterate_samples(num_samples=2))
    assert len(samples) > 0
    for sample in samples:
        assert sample.id is not None
        assert isinstance(sample.subject, str)
        assert sample.messages


def test_basic_skills_tasks_smoke() -> None:
    _smoke_test_task(BasicSkillsCloze)
    _smoke_test_task(BasicSkillsMC)


def test_boolq_tasks_smoke() -> None:
    _smoke_test_task(BoolQCloze)
    _smoke_test_task(BoolQMC)


def test_csqa_tasks_smoke() -> None:
    _smoke_test_task(CommonsenseQACloze)
    _smoke_test_task(CommonsenseQAMC)


def test_paloma_smoke() -> None:
    _smoke_test_task(GenericPaloma)


def test_tydiqa_secondary_task_smoke() -> None:
    _smoke_test_task(TyDiQASecondaryTask)


def test_coqa_tasks_smoke() -> None:
    _smoke_test_task(CoQA)
    _smoke_test_task(CoQAMC)


def test_drop_tasks_smoke() -> None:
    _smoke_test_task(DropRC)
    _smoke_test_task(DropMC)


def test_jeopardy_tasks_smoke() -> None:
    _smoke_test_task(JeopardyCompletion)
    _smoke_test_task(JeopardyMC)


def test_lab_bench_tasks_smoke() -> None:
    _smoke_test_task(LabBenchCloze)
    _smoke_test_task(LabBenchMC)


def test_copycolors_tasks_smoke() -> None:
    _smoke_test_task(CopyColorsCloze)
    _smoke_test_task(CopyColorsMC)


def test_naturalqs_open_tasks_smoke() -> None:
    _smoke_test_task(NaturalQsOpen)
    _smoke_test_task(NaturalQsOpenCloze)
    _smoke_test_task(NaturalQsOpenMC)


def test_simpleqa_smoke() -> None:
    _smoke_test_task(SimpleQACompletion)


def test_simpletom_smoke() -> None:
    _smoke_test_task(SimpleToMMC)


def test_qasper_yesno_tasks_smoke() -> None:
    _smoke_test_task(QASPERYesNo)
    _smoke_test_task(QASPERYesNoMC)


def test_popqa_smoke() -> None:
    _smoke_test_task(PopQA)

