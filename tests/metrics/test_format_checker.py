import pytest

from eval_framework.metrics.completion_metrics.format_checker import CheckJsonFormat, CheckPostScriptFormat
from eval_framework.shared.types import Completion


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion='{"name": "John Doe",\n'
                '"age": 35,\n'
                '"email": "john.doe@example.com",\n'
                '"hobbies": ["reading", "hiking", "photography"],\n'
                '"address": {"street": "123 Main St", "city": "Anytown"}}',
                raw_completion='{"name": "John Doe",\n'
                '"age": 35,\n'
                '"email": "john.doe@example.com",\n'
                '"hobbies": ["reading", "hiking", "photography"],\n'
                '"address": {"street": "123 Main St", "city": "Anytown"}}',
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_json",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion='```json\n{"name": "John Doe",\n'
                '"age": 35,\n'
                '"email": "john.doe@example.com",\n'
                '"hobbies": ["reading", "hiking", "photography"],\n'
                '"address": {"street": "123 Main St", "city": "Anytown"}}\n```',
                raw_completion='```json\n{"name": "John Doe",\n'
                '"age": 35,\n'
                '"email": "john.doe@example.com",\n'
                '"hobbies": ["reading", "hiking", "photography"],\n'
                '"address": {"street": "123 Main St", "city": "Anytown"}}\n```',
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_json_embedded",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion='{name: "John Doe",\n'
                '"age": 35,\n'
                '"email": "john.doe@example.com",\n'
                '"hobbies": ["reading", "hiking", "photography"],\n'
                '"address": {"street": "123 Main St", "city": "Anytown"}}',
                raw_completion='{name: "John Doe",\n'
                '"age": 35,\n'
                '"email": "john.doe@example.com",\n'
                '"hobbies": ["reading", "hiking", "photography"],\n'
                '"address": {"street": "123 Main St", "city": "Anytown"}}',
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_json",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="This is a json-file:\n"
                '```json\n{"name": "John Doe",\n'
                '"age": 35,\n'
                '"email": "john.doe@example.com",\n'
                '"hobbies": ["reading", "hiking", "photography"],\n'
                '"address": {"street": "123 Main St", "city": "Anytown"}}\n```',
                raw_completion="This is a json-file:\n"
                '```json\n{"name": "John Doe",\n'
                '"age": 35,\n'
                '"email": "john.doe@example.com",\n'
                '"hobbies": ["reading", "hiking", "photography"],\n'
                '"address": {"street": "123 Main St", "city": "Anytown"}}\n```',
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_json_embedded",
        ),
    ],
)
def test_check_json_format(response: Completion, expected_value: float) -> None:
    metric = CheckJsonFormat()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "JSON Format"
    assert results[0].higher_is_better is True


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="This is a postscript P.P.S. ",
                raw_completion="This is a postscript P.P.S. ",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_postscript_pps",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="This is a postscript P.S. X",
                raw_completion="This is a postscript P.S. X",
                raw_completion_sequence_positions=None,
            ),
            1.0,
            id="valid_postscript_ps",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="This is a postscript P. S. X",
                raw_completion="This is a postscript P. S. X",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_postscript_ps",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="This is a postscript P.P. S. X",
                raw_completion="This is a postscript P.P. S. X",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="invalid_postscript_pps",
        ),
    ],
)
def test_check_postscript_format(response: Completion, expected_value: float) -> None:
    metric = CheckPostScriptFormat()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Postscript Format"
    assert results[0].higher_is_better is True
