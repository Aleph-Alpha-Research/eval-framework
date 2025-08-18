import json
from typing import Any, Mapping

import pytest

from eval_framework.metrics.completion_metrics.json_format import JsonFormat, get_json_object, remove_comments
from eval_framework.shared.types import Completion

CITY_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "properties": {
        "cities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "country": {"type": "string"},
                    "population": {"type": "integer", "minimum": 0},
                },
                "required": ["name", "country", "population"],
            },
        }
    },
    "required": ["cities"],
}
INVALID_SCHEMA = {"title": "my json", "description": "my json", "type": "invalid_type"}
CITY_JSON = json.dumps(
    {
        "cities": [
            {"name": "Moscow", "country": "Russia", "population": 12615279},
            {"name": "London", "country": "United Kingdom", "population": 9304016},
            {"name": "Saint Petersburg", "country": "Russia", "population": 5384342},
            {"name": "Berlin", "country": "Germany", "population": 3748148},
            {"name": "Madrid", "country": "Spain", "population": 3265038},
        ]
    }
)

OTHER_JSON = json.dumps({"knock knock": "who's there?"})


@pytest.mark.parametrize(
    "answer, schema, is_just_json, is_valid_json, fulfills_schema",
    [
        (f"{CITY_JSON}", CITY_SCHEMA, True, True, True),
        (f"```{CITY_JSON}```", CITY_SCHEMA, True, True, True),
        (
            (
                "Sure! Here's the JSON object that holds the five largest cities in Europe and "
                f"their population:\n```{CITY_JSON}```"
            ),
            CITY_SCHEMA,
            False,
            True,
            True,
        ),
        (f"{OTHER_JSON}", CITY_SCHEMA, True, True, False),
        ("no json here", CITY_SCHEMA, False, False, False),
        ("no json here", None, False, False, None),
        (f"{CITY_JSON}", INVALID_SCHEMA, True, True, None),
        ("{this is no json}", None, False, False, None),
    ],
)
def test_do_evaluate_single_output(
    answer: str,
    schema: Mapping[str, Any] | None,
    is_just_json: bool,
    is_valid_json: bool,
    fulfills_schema: bool,
) -> None:
    completion = Completion(
        id=0,
        subject="en",
        ground_truth=json.dumps({"json_schema": schema}),
        prompt="",
        prompt_sequence_positions=None,
        messages=None,
        completion=answer,
        raw_completion=answer,
        raw_completion_sequence_positions=None,
    )

    metric = JsonFormat()
    results = metric.calculate(completion)
    assert len(results) == 3
    for result in results:
        match result.metric_name:
            case "json_format/is_just_json":
                assert result.value == is_just_json
            case "json_format/is_valid_json":
                assert result.value == is_valid_json
            case "json_format/fulfills_schema":
                assert result.value == fulfills_schema


@pytest.mark.parametrize(
    "text, expected",
    [
        ("non-json stuff", None),
        ('non-json stuff {"key": "value"} non-json stuff', '{"key": "value"}'),
        (
            'non-json stuff {"key": "value with \\n new line"} non-json stuff',
            '{"key": "value with \\n new line"}',
        ),
        (
            'non-json stuff {"nested": {"inner": "json"}} non-json stuff',
            '{"nested": {"inner": "json"}}',
        ),
        (
            'non-json stuff {"nested": [{"inner": "json"}]} non-json stuff',
            '{"nested": [{"inner": "json"}]}',
        ),
        (
            'non-json stuff [{"nested": "json"}] non-json stuff',
            '[{"nested": "json"}]',
        ),
    ],
)
def test_get_json_object(text: str, expected: str) -> None:
    if expected:
        assert expected == get_json_object(text)
    else:
        with pytest.raises(RuntimeError):
            get_json_object(text)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("line  // with comment\n    // just comment\nline", "line  \nline"),
    ],
)
def test_remove_comments(text: str, expected: str) -> None:
    assert expected == remove_comments(text)
