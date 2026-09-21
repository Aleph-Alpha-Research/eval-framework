import pytest

from eval_framework.benchmarks.arc_de import ARC_DE_BENCHMARK, ArcDeReader
from eval_framework.contract import Benchmark
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter
from tests.tests_eval_framework.tasks.benchmarks.utils import assert_benchmark_formatter_hash


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("benchmark", [ARC_DE_BENCHMARK], ids=lambda b: b.id())
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls)


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
