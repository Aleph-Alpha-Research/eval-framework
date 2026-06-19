import pytest

from eval_framework.tasks.benchmarks.squad import SQuAD2_MA
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import get_task_names_for_module, run_formatter_hash_test


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("squad"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


_ANSWERABLE = {
    "context": "Paris is the capital of France.",
    "question": "What is the capital of France?",
    "answers": {"text": ["Paris"], "answer_start": [0]},
}
_UNANSWERABLE = {
    "context": "Paris is the capital of France.",
    "question": "What is the capital of Spain?",
    "answers": {"text": [], "answer_start": []},
}


def test_squad2_ma_system_prompt_instructs_reject() -> None:
    system = SQuAD2_MA()._get_system_prompt_text(_ANSWERABLE)
    assert system is not None
    assert f"respond with '{SQuAD2_MA.UNANSWERABLE_STR}'" in system


def test_squad2_ma_instruction_is_context_question_answer_only() -> None:
    instruction = SQuAD2_MA()._get_instruction_text(_ANSWERABLE)
    assert instruction == ("Context:\nParis is the capital of France.\n\nQuestion:\nWhat is the capital of France?")


def test_squad2_ma_reject_ground_truth_matches_instruction() -> None:
    task = SQuAD2_MA()
    assert task._get_ground_truth(_ANSWERABLE) == [" Paris"]
    assert f" {task.UNANSWERABLE_STR}" in task._get_ground_truth(_UNANSWERABLE)
