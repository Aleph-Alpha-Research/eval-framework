import pytest

from eval_framework.tasks.benchmarks.pawsx import PAWSX
from tests.utils import DatasetPatcher


class TestPawsx:
    @pytest.fixture
    def pawsx_task(self) -> PAWSX:
        with DatasetPatcher(PAWSX, num_fewshot=2) as patched_task:
            return patched_task

    def test_post_process_generated_completion(self, pawsx_task: PAWSX) -> None:
        # GIVEN completion with some special chars, THEN these are stripped.
        assert pawsx_task.post_process_generated_completion(" Yes.") == "Yes"
        assert pawsx_task.post_process_generated_completion("'No") == "No"
        assert pawsx_task.post_process_generated_completion("\nNo") == "No"
        assert pawsx_task.post_process_generated_completion("\nYes") == "Yes"

    def test_sample_fewshot_examples(self, pawsx_task: PAWSX) -> None:
        # THEN we can obtain a sample without errors
        for _ in pawsx_task.iterate_samples(1):
            pass
        # AND the shots are balanced (pos/neg)
        counts = {0: 0, 1: 0}
        for example in pawsx_task._sample_fewshot_examples({}):
            counts[example["label"]] += 1
        assert counts[0] == counts[1] == 1
