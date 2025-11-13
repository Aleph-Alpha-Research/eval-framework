import pytest

from eval_framework.tasks.benchmarks.humaneval import HumanEval, HumanEvalInstruct
from eval_framework.tasks.utils import run_python_code
from tests.tests_eval_framework.utils import DatasetPatcher


class TestHumanEvalCode:
    @pytest.fixture
    def human_eval_task(self) -> HumanEval:
        with DatasetPatcher(HumanEval, num_fewshot=0) as patched_task:
            return patched_task

    def test_code_is_executed(self, human_eval_task: HumanEval) -> None:
        assert len(human_eval_task.SUBJECTS) > 0
        human_eval_task._load_dataset(human_eval_task.SUBJECTS[0])
        i = 0
        for i, item in enumerate(human_eval_task.dataset[human_eval_task.SAMPLE_SPLIT][:10]):
            sample = human_eval_task._create_samples(item, i, human_eval_task.SUBJECTS[0])[0]
            formatted_code = human_eval_task.post_process_generated_completion(item["canonical_solution"], sample)
            assert run_python_code(formatted_code).endswith("True")
            formatted_code = human_eval_task.post_process_generated_completion("", sample)
            assert not run_python_code(formatted_code).endswith("True")
        assert i == 9


class TestHumanEvalInstructCode:
    @pytest.fixture
    def human_eval_task_inst(self) -> HumanEvalInstruct:
        with DatasetPatcher(HumanEvalInstruct, num_fewshot=0) as patched_task:
            return patched_task

    def test_code_is_executed(self, human_eval_task_inst: HumanEvalInstruct) -> None:
        assert len(human_eval_task_inst.SUBJECTS) > 0
        human_eval_task_inst._load_dataset(human_eval_task_inst.SUBJECTS[0])
        i = 0
        for i, item in enumerate(human_eval_task_inst.dataset[human_eval_task_inst.SAMPLE_SPLIT][:10]):
            sample = human_eval_task_inst._create_samples(item, i, human_eval_task_inst.SUBJECTS[0])[0]
            completion = item["canonical_solution"]
            formatted_code = human_eval_task_inst.post_process_generated_completion(completion, sample)
            assert run_python_code(formatted_code).endswith("True")
        assert i == 9
