import pytest

from eval_framework.tasks.benchmarks.triviaqa import TriviaQA_MA
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import get_task_names_for_module, run_formatter_hash_test


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", get_task_names_for_module("triviaqa"))
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls)


@pytest.fixture
def item():
    return {
        "question": "What is the capital of France?",
        "answer": {"aliases": ["Paris", "Paris, France"]},
        "entity_pages": {"wiki_context": ["Paris is the capital of France.", "France is in Europe."]},
    }


def test_triviaqa_ma_system_prompt_instructs_reject(item) -> None:
    system = TriviaQA_MA()._get_system_prompt_text(item)
    assert system is not None
    assert f"respond with '{TriviaQA_MA.UNANSWERABLE_STR}'" in system


def test_triviaqa_ma_instruction_is_context_question_only(item) -> None:
    instruction = TriviaQA_MA()._get_instruction_text(item)
    assert instruction == (
        "Context:\nParis is the capital of France.\n\nFrance is in Europe.\n\n"
        "Question:\nWhat is the capital of France?\n"
    )


def test_triviaqa_ma_ground_truth_uses_aliases(item) -> None:
    assert TriviaQA_MA()._get_ground_truth(item) == ["Paris", "Paris, France"]
