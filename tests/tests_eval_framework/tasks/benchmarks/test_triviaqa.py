import pytest

from eval_framework.tasks.benchmarks.triviaqa import TriviaQAMA
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import get_task_names_for_module, run_formatter_hash_test


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("triviaqa"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


_ITEM = {
    "question": "What is the capital of France?",
    "answer": {"aliases": ["Paris", "Paris, France"]},
    "entity_pages": {"wiki_context": ["Paris is the capital of France.", "France is in Europe."]},
}


def test_triviaqa_ma_system_prompt_instructs_reject() -> None:
    system = TriviaQAMA()._get_system_prompt_text(_ITEM)
    assert system is not None
    assert f"respond with '{TriviaQAMA.UNANSWERABLE_STR}'" in system


def test_triviaqa_ma_instruction_is_context_question_only() -> None:
    instruction = TriviaQAMA()._get_instruction_text(_ITEM)
    assert instruction == (
        "Context:\nParis is the capital of France.\n\nFrance is in Europe.\n\n"
        "Question:\nWhat is the capital of France?\n"
    )


def test_triviaqa_ma_ground_truth_uses_aliases() -> None:
    assert TriviaQAMA()._get_ground_truth(_ITEM) == ["Paris", "Paris, France"]
