import json

import pytest

from eval_framework.metrics.completion_metrics.length_control import LengthControl, LengthRequirementUnit
from eval_framework.shared.types import Completion

UNIT_STRINGS = {
    "words": {"text": "Hello", "sep": " "},
    "sentences": {"text": "Hello I am a sentence.", "sep": " "},
    "paragraphs": {
        "text": "Hello I am a paragraph. I contain two sentences.",
        "sep": "\n\n",
    },
}


@pytest.mark.parametrize(
    "unit",
    [
        LengthRequirementUnit.WORDS.value,
        LengthRequirementUnit.SENTENCES.value,
        LengthRequirementUnit.PARAGRAPHS.value,
    ],
)
@pytest.mark.parametrize(
    "count, count_requirement, expected_fulfills_length_requirement, expected_absolute_normalized_distance_to_target, "
    "expected_normalized_distance_to_target",
    [
        (50, 100, False, 0.5, -0.5),
        (100, 100, True, 0, 0.0),
        (4, 5, False, 1 / 5, -1 / 5),
        (5, 6, True, 1 / 6, -1 / 6),
        (6, 5, False, 1 / 5, 1 / 5),
        (5, 4, False, 1 / 4, 1 / 4),
        (4, 6, False, 1 / 3, -1 / 3),
    ],
)
def test_length_control_for_target(
    unit: str,
    count: int,
    count_requirement: int,
    expected_fulfills_length_requirement: bool,
    expected_absolute_normalized_distance_to_target: float,
    expected_normalized_distance_to_target: float,
) -> None:
    completion = Completion(
        id=0,
        subject="en",
        ground_truth=json.dumps({"count": count_requirement, "unit": unit, "type": "target"}),
        prompt="",
        prompt_sequence_positions=None,
        messages=None,
        # including two newlines at the beginning that should be ignored
        completion="\n\n " + UNIT_STRINGS[unit]["sep"].join([UNIT_STRINGS[unit]["text"]] * count),
        raw_completion="\n\n " + UNIT_STRINGS[unit]["sep"].join([UNIT_STRINGS[unit]["text"]] * count),
        raw_completion_sequence_positions=None,
    )

    metric = LengthControl()
    results = metric.calculate(completion)
    assert len(results) == 3
    for result in results:
        assert result.value is not None
        match result.metric_name:
            case "length_control/normalized_distance_to_target":
                assert round(result.value, 8) == round(expected_normalized_distance_to_target, 8)
            case "length_control/absolute_normalized_distance_to_target":
                assert round(result.value, 8) == round(expected_absolute_normalized_distance_to_target, 8)
            case "length_control/fulfills_length_requirement":
                assert result.value == expected_fulfills_length_requirement


@pytest.mark.parametrize(
    "unit",
    [
        LengthRequirementUnit.WORDS.value,
        LengthRequirementUnit.SENTENCES.value,
        LengthRequirementUnit.PARAGRAPHS.value,
    ],
)
@pytest.mark.parametrize(
    "count, min_count_requirement, expected_fulfills_length_requirement, "
    "expected_absolute_normalized_distance_to_target, expected_normalized_distance_to_target",
    [
        (50, 100, False, 0.5, -0.5),
        (100, 100, True, 0, 0.0),
        (4, 5, False, 1 / 5, -1 / 5),
        (5, 6, False, 1 / 6, -1 / 6),
        (6, 5, True, 1 / 5, 1 / 5),
        (5, 4, True, 1 / 4, 1 / 4),
    ],
)
def test_length_control_for_min(
    unit: str,
    count: int,
    min_count_requirement: int,
    expected_fulfills_length_requirement: bool,
    expected_absolute_normalized_distance_to_target: float,
    expected_normalized_distance_to_target: float,
) -> None:
    completion = Completion(
        id=0,
        subject="en",
        ground_truth=json.dumps({"count": min_count_requirement, "unit": unit, "type": "minimum"}),
        prompt="",
        prompt_sequence_positions=None,
        messages=None,
        completion=UNIT_STRINGS[unit]["sep"].join([UNIT_STRINGS[unit]["text"]] * count),
        raw_completion=UNIT_STRINGS[unit]["sep"].join([UNIT_STRINGS[unit]["text"]] * count),
        raw_completion_sequence_positions=None,
    )

    metric = LengthControl()
    results = metric.calculate(completion)
    assert len(results) == 3
    for result in results:
        assert result.value is not None
        match result.metric_name:
            case "length_control/normalized_distance_to_target":
                assert round(result.value, 8) == round(expected_normalized_distance_to_target, 8)
            case "length_control/absolute_normalized_distance_to_target":
                assert round(result.value, 8) == round(expected_absolute_normalized_distance_to_target, 8)
            case "length_control/fulfills_length_requirement":
                assert result.value == expected_fulfills_length_requirement


@pytest.mark.parametrize(
    "unit",
    [
        LengthRequirementUnit.WORDS.value,
        LengthRequirementUnit.SENTENCES.value,
        LengthRequirementUnit.PARAGRAPHS.value,
    ],
)
@pytest.mark.parametrize(
    "count, max_count_requirement, expected_fulfills_length_requirement, "
    "expected_absolute_normalized_distance_to_target, expected_normalized_distance_to_target",
    [
        (50, 100, True, 0.5, -0.5),
        (100, 100, True, 0, 0.0),
        (4, 5, True, 1 / 5, -1 / 5),
        (5, 6, True, 1 / 6, -1 / 6),
        (6, 5, False, 1 / 5, 1 / 5),
        (5, 4, False, 1 / 4, 1 / 4),
    ],
)
def test_length_control_for_max(
    unit: str,
    count: int,
    max_count_requirement: int,
    expected_fulfills_length_requirement: bool,
    expected_absolute_normalized_distance_to_target: float,
    expected_normalized_distance_to_target: float,
) -> None:
    completion = Completion(
        id=0,
        subject="en",
        ground_truth=json.dumps({"count": max_count_requirement, "unit": unit, "type": "maximum"}),
        prompt="",
        prompt_sequence_positions=None,
        messages=None,
        completion=UNIT_STRINGS[unit]["sep"].join([UNIT_STRINGS[unit]["text"]] * count),
        raw_completion=UNIT_STRINGS[unit]["sep"].join([UNIT_STRINGS[unit]["text"]] * count),
        raw_completion_sequence_positions=None,
    )

    metric = LengthControl()
    results = metric.calculate(completion)
    assert len(results) == 3
    for result in results:
        assert result.value is not None
        match result.metric_name:
            case "length_control/normalized_distance_to_target":
                assert round(result.value, 8) == round(expected_normalized_distance_to_target, 8)
            case "length_control/absolute_normalized_distance_to_target":
                assert round(result.value, 8) == round(expected_absolute_normalized_distance_to_target, 8)
            case "length_control/fulfills_length_requirement":
                assert result.value == expected_fulfills_length_requirement
