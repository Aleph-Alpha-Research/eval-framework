from collections.abc import Sequence

import pytest

from eval_framework.metrics.completion.csv_format import CSVFormat, extract_csv_from_text
from eval_framework.shared.types import Completion

REQ_NUM = 3

CITY_CSV = """
Name,Country,Population
Moscow,Russia,12615279
London,United Kingdom,9304016
Saint Petersburg,Russia,5384342
Berlin,Germany,3748148
Madrid,Spain,3265038"""

CITY_CSV_SEMICOLON = """
Name;Country
Moscow;Russia
London;United Kingdom
Saint Petersburg;Russia
Berlin;Germany
Madrid;Spain"""

CSV_WITH_INTRO = """Sure here's your CSV:
'''
Name,Country,Population
Moscow,Russia,12615279
London,United Kingdom,9304016
Saint Petersburg,Russia,5384342
Berlin,Germany,3748148
Madrid,Spain,3265038"""

NO_CSV_TEXT = """Sure here's your CSV:
This is not a csv."""


def assert_consistent_column_count(csv_lines: Sequence[str] | None, delimiter: str | None, count: int) -> None:
    assert csv_lines
    counts = [len(csv_line.split(delimiter)) for csv_line in csv_lines]
    assert set(counts) == set([count])


def test_extract_csv_from_text_comma() -> None:
    text = """
Here is some intro text.
name,age,city
John,30,New York
Jane,25,Los Angeles
This is some outro text.
Some,sentences,in
Outro text have the same pattern
    """
    csv_lines, delimiter = extract_csv_from_text(text)
    assert csv_lines
    assert (
        "\n".join(csv_lines)
        == """name,age,city
John,30,New York
Jane,25,Los Angeles"""
    )
    assert delimiter == ","
    assert_consistent_column_count(csv_lines, delimiter, 3)


def test_extract_csv_from_text_tab() -> None:
    text = """
Here is some intro text.
name	age	city
John	30	New York
Jane	25	Los Angeles
This is some outro text.
Some	sentences	in
Outro text have the same pattern
    """
    csv_lines, delimiter = extract_csv_from_text(text)
    assert csv_lines
    assert (
        "\n".join(csv_lines)
        == """name	age	city
John	30	New York
Jane	25	Los Angeles"""
    )
    assert delimiter == "	"
    assert_consistent_column_count(csv_lines, delimiter, 3)


def test_extract_csv_from_text_spaces() -> None:
    text = """
Here is some intro text.
name age city
John 30 NewYork
Jane 25 LosAngeles
This is some outro text.
Some sentences in
Outro text have the same pattern
    """
    csv_lines, delimiter = extract_csv_from_text(text)
    assert csv_lines
    assert (
        "\n".join(csv_lines)
        == """name age city
John 30 NewYork
Jane 25 LosAngeles"""
    )
    assert delimiter == " "
    assert_consistent_column_count(csv_lines, delimiter, 3)


def test_extract_csv_from_text_comma_ignores_mixed_numbers_of_commas() -> None:
    text = """
Here is some intro text.
name,age,city
John,30,New York
Jane,25,Los Angeles
This is some outro text.
Some,sentences,in
another,line here
something with, commas,
hey, there
Outro text have the same pattern
    """
    csv_lines, delimiter = extract_csv_from_text(text)
    assert csv_lines
    assert (
        "\n".join(csv_lines)
        == """name,age,city
John,30,New York
Jane,25,Los Angeles"""
    )
    assert delimiter == ","
    assert_consistent_column_count(csv_lines, delimiter, 3)


test_setups = [
    (f"{CITY_CSV}", [True, True, True]),
    (f"{CITY_CSV_SEMICOLON}", [True, False, False]),
    (f"{NO_CSV_TEXT}", [False, False, False]),
]


@pytest.mark.parametrize("sample_csv, asserts", test_setups)
def test_csv_format_for_target(sample_csv: str, asserts: list[bool]) -> None:
    completion = Completion(
        id=0,
        subject="en",
        ground_truth='{"columns": ["Name", "Country", "Population"], "separator": "comma"}',
        prompt="",
        prompt_sequence_positions=None,
        messages=None,
        completion=sample_csv,
        raw_completion="",
        raw_completion_sequence_positions=None,
    )

    metric = CSVFormat()
    results = metric.calculate(completion)
    assert len(results) == 3
    for result in results:
        match result.metric_name:
            case "CSV Format/has_csv":
                assert result.value == asserts[0]
            case "CSV Format/is_separator_respected":
                assert result.value == asserts[1]
            case "CSV Format/is_column_count_respected":
                assert result.value == asserts[2]
