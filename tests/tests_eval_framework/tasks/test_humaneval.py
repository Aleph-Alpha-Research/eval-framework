import pytest

from eval_framework.tasks.benchmarks.humaneval import HumanEval, HumanEval_OLMES, HumanEvalInstruct
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


class TestHumanEvalOLMES:
    @pytest.fixture
    def human_eval_olmes_task(self) -> HumanEval_OLMES:
        with DatasetPatcher(HumanEval_OLMES, num_fewshot=3) as patched_task:
            return patched_task

    def test_code_is_executed(self, human_eval_olmes_task: HumanEval_OLMES) -> None:
        assert len(human_eval_olmes_task.SUBJECTS) > 0
        human_eval_olmes_task._load_dataset(human_eval_olmes_task.SUBJECTS[0])
        i = 0
        for i, item in enumerate(human_eval_olmes_task.dataset[human_eval_olmes_task.SAMPLE_SPLIT][:10]):
            sample = human_eval_olmes_task._create_samples(item, i, human_eval_olmes_task.SUBJECTS[0])[0]
            formatted_code = human_eval_olmes_task.post_process_generated_completion(item["canonical_solution"], sample)
            assert run_python_code(formatted_code).endswith("True")
            formatted_code = human_eval_olmes_task.post_process_generated_completion("", sample)
            assert not run_python_code(formatted_code).endswith("True")
        assert i == 9

    def test_olmes_settings(self, human_eval_olmes_task: HumanEval_OLMES) -> None:
        assert human_eval_olmes_task.num_fewshot == 3
        assert human_eval_olmes_task.max_tokens == 1024
        assert "\nclass" in human_eval_olmes_task.stop_sequences
        assert "\nif" in human_eval_olmes_task.stop_sequences
        assert "\nprint" in human_eval_olmes_task.stop_sequences
        assert "\n#" in human_eval_olmes_task.stop_sequences
        assert "\n```" in human_eval_olmes_task.stop_sequences
        assert human_eval_olmes_task.SAMPLE_SPLIT == "test"
        assert human_eval_olmes_task.FEWSHOT_SPLIT == "test"

    def test_olmes_prompt_format(self, human_eval_olmes_task: HumanEval_OLMES) -> None:
        human_eval_olmes_task._load_dataset(human_eval_olmes_task.SUBJECTS[0])
        item = human_eval_olmes_task.dataset[human_eval_olmes_task.SAMPLE_SPLIT][0]
        instruction = human_eval_olmes_task._get_instruction_text(item)
        assert instruction.startswith("```python\n")
        assert instruction == "```python\n" + item["prompt"]

        fewshot_target = human_eval_olmes_task._get_fewshot_target_text(item)
        assert fewshot_target.endswith("```")
        assert fewshot_target == item["canonical_solution"] + "```"


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
