import pytest

from eval_framework.tasks.benchmarks.arc_de import ArcDeReader
from eval_framework.tasks.registry import Registry
from eval_framework.tasks.task_names import register_arc_de_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import run_formatter_hash_test

# Registry for this test suite only holding arc_de tasks
_arc_de_registry = Registry()
register_arc_de_tasks(registry=_arc_de_registry)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("task_name", _arc_de_registry.task_names())
def test_formatter_hash(task_name: str, formatter_cls: type[BaseFormatter]) -> None:
    run_formatter_hash_test(task_name, formatter_cls, registry=_arc_de_registry)


@pytest.mark.parametrize("answer_key", ["2", "B"])
def test_reader_maps_fields_and_normalises_answer_key(answer_key: str) -> None:
    # Given an ARC-DE row whose correct answer is the 2nd option, keyed as a number or a letter
    # (the dataset uses both forms)
    row = {
        "question_de": "Welche Farbe hat der Himmel?",
        "choices_de": {"text": ["Grün", "Blau", "Rot"]},
        "answerKey": answer_key,
    }

    # When reading it into choice fields
    fields = ArcDeReader().read(row)

    # Then the German question and answer texts carry over, and the key becomes a 0-based index
    assert fields.raw_question == "Welche Farbe hat der Himmel?"
    assert fields.choices == ["Grün", "Blau", "Rot"]
    assert fields.correct_index == 1
