import pytest

from eval_framework.tasks.benchmarks.mbpp import _OLMES_FEWSHOT_EXAMPLES, MBPP_OLMES
from eval_framework.tasks.utils import run_python_code
from template_formatting.formatter import ConcatFormatter
from tests.tests_eval_framework.utils import DatasetPatcher


class TestMBPP_OLMES:
    @pytest.fixture
    def task(self) -> MBPP_OLMES:
        with DatasetPatcher(MBPP_OLMES, num_fewshot=3, num_samples=10) as patched_task:
            return patched_task

    def test_num_fewshot_must_be_3(self) -> None:
        with pytest.raises(AssertionError, match="MBPP_OLMES requires exactly 3 fewshot examples"):
            MBPP_OLMES(num_fewshot=1)

    def test_stop_sequences(self) -> None:
        task = MBPP_OLMES(num_fewshot=3)
        assert task.stop_sequences == ["```", '\n"""', "\nassert", "\n#"]

    def test_instruction_uses_evalplus_format(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]
        item["subject"] = task.SUBJECTS[0]
        instruction = task._get_instruction_text(item)

        expected_prefix = (
            "Please provide a self-contained Python script that solves the following problem"
            " in a markdown code block:\n```\n"
        )
        assert instruction.startswith(expected_prefix)
        assert instruction.endswith("\n```\n")
        assert item["test_list"][0] in instruction

    def test_instruction_contains_only_one_test(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]
        item["subject"] = task.SUBJECTS[0]
        instruction = task._get_instruction_text(item)

        for test in item["test_list"][1:]:
            assert test not in instruction

    def test_cue_text(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]
        cue = task._get_cue_text(item)
        assert cue == "Here is the completed function:\n\n```python\n"

    def test_fewshot_examples_are_hardcoded(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]

        examples = task._sample_fewshot_examples(item)
        assert len(examples) == 3
        assert examples[0]["text"] == _OLMES_FEWSHOT_EXAMPLES[0]["text"]
        assert examples[1]["text"] == _OLMES_FEWSHOT_EXAMPLES[1]["text"]
        assert examples[2]["text"] == _OLMES_FEWSHOT_EXAMPLES[2]["text"]

    def test_fewshot_examples_are_deterministic(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]

        examples_1 = task._sample_fewshot_examples(item)
        examples_2 = task._sample_fewshot_examples(item)
        assert examples_1 == examples_2

    def test_fewshot_target_is_code_with_newline(self) -> None:
        task = MBPP_OLMES(num_fewshot=3)
        for example in _OLMES_FEWSHOT_EXAMPLES:
            target = task._get_fewshot_target_text(example)
            assert target == example["code"] + "\n"
            assert "```" not in target

    def test_code_execution_with_canonical_solution(self, task: MBPP_OLMES) -> None:
        task._load_dataset(task.SUBJECTS[0])
        for i, item in enumerate(task.dataset[task.SAMPLE_SPLIT][:5]):
            item["subject"] = task.SUBJECTS[0]
            sample = task._create_samples(item, i, task.SUBJECTS[0])[0]
            code = task.post_process_generated_completion(item["code"], sample)
            result = run_python_code(code)
            assert result.endswith("True"), f"Item {i} failed: {result}"

    def test_prompt_format_matches_oe_eval(self, task: MBPP_OLMES) -> None:
        """Verify the assembled prompt has the expected structure with ConcatFormatter."""
        task._load_dataset(task.SUBJECTS[0])
        item = task.dataset[task.SAMPLE_SPLIT][0]
        item["subject"] = task.SUBJECTS[0]
        sample = task._create_samples(item, 0, task.SUBJECTS[0])[0]

        formatter = ConcatFormatter()
        formatted = formatter.format(sample.messages, output_mode="string")

        assert "Please provide a self-contained Python script" in formatted
        assert "Here is the completed function:" in formatted
        assert "```python" in formatted

        fewshot_count = formatted.count("Please provide a self-contained Python script")
        assert fewshot_count == 4, f"Expected 4 occurrences (3 fewshot + 1 eval), got {fewshot_count}"
