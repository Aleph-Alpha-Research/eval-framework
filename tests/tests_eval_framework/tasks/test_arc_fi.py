import pytest

from eval_framework.tasks.benchmarks.arc_fi import ARC_FI
from tests.tests_eval_framework.utils import DatasetPatcher


class TestARC_FI:
    @pytest.fixture
    def arc_fi_task(self) -> ARC_FI:
        with DatasetPatcher(ARC_FI, num_fewshot=1) as patched_task:
            return patched_task

    def test_arc_fi_five_samples(self, arc_fi_task: ARC_FI) -> None:
        assert len(arc_fi_task.SUBJECTS) > 0
        arc_fi_task._load_dataset(arc_fi_task.SUBJECTS[0])
        item = next(iter(x for x in arc_fi_task.dataset["validation"] if len(x["choices"]["label"]) > 4))
        result = arc_fi_task._get_fewshot_target_text(item)
        gt = arc_fi_task._get_ground_truth(item)
        assert gt is not None
        assert gt in result

    def test_arc_fi_num_label(self, arc_fi_task: ARC_FI) -> None:
        assert len(arc_fi_task.SUBJECTS) > 0
        arc_fi_task._load_dataset(arc_fi_task.SUBJECTS[0])
        item = next(iter(x for x in arc_fi_task.dataset["validation"] if x["choices"]["label"][0] == "1"))
        result = arc_fi_task._get_fewshot_target_text(item)
        gt = arc_fi_task._get_ground_truth(item)
        assert gt is not None
        assert gt in result
