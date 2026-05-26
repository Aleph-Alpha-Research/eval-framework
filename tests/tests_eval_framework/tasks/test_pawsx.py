import pytest

from eval_framework.tasks.benchmarks.pawsx import PAWSX
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.utils import get_task_names_for_module, run_formatter_hash_test
from tests.tests_eval_framework.utils import DatasetPatcher


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


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("pawsx"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, num_fewshot=2)
