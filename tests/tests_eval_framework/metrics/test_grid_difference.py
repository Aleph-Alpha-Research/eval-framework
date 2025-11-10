import pytest

import eval_framework.tasks.benchmarks.sphyr as sphyr
from eval_framework.metrics.completion.grid_difference import GridDifference
from eval_framework.shared.types import Completion
from template_formatting.formatter import Message, Role
from tests_eval_framework.utils import assert_hash_string

GRID = """0 0 L L L 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0
0 0 0 V 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0
0 0 0 1 0 0 0 0 0 0
0 0 1 1 1 0 0 0 0 0
0 0 1 1 1 0 0 0 0 0
0 0 1 1 1 0 0 0 0 0
S S S S S S S S S 0"""


class TestGridDifference:
    @pytest.fixture
    def grid_difference_metric(self) -> GridDifference:
        return GridDifference()

    def test_extract_grid_from_prompt(self, grid_difference_metric: GridDifference) -> None:
        EXPECTED_CONCAT_PROMPT = (
            sphyr.SYSTEM_PROMPT
            + "\n\n"
            + sphyr.PROMPT_TEMPLATE.format(GRID=GRID, FILL_INSTRUCTION=sphyr.EASY_FILL_INSTRUCTION)
        )
        grid = grid_difference_metric.extract_grid_from_prompt(EXPECTED_CONCAT_PROMPT)
        assert_hash_string(
            task_name=grid_difference_metric.__class__.__name__,
            suffix_key="EXTRACTED_GRID",
            tested_string=grid,
        )

    calculate_score_params = [
        (0, 3, 1.0),  # 0 differences in output, 3 in ground truth
        (1, 3, (1 - (1 / 3))),  # 1 difference in output, 3 in ground truth
        (2, 3, (1 - (2 / 3))),  # 2 differences in output, 3 in ground truth
        (3, 3, 0.0),  # 3 differences in output, 3 in ground truth
        (5, 3, (1 - (5 / 3))),  # 5 differences in output, 3 in ground truth
    ]

    @pytest.mark.parametrize(
        "output_ground_truth_differences_count, input_ground_truth_differences_count, expected_score",
        calculate_score_params,
    )
    def test_calculate_score(
        self,
        output_ground_truth_differences_count: int,
        input_ground_truth_differences_count: int,
        expected_score: float,
        grid_difference_metric: GridDifference,
    ) -> None:
        score = grid_difference_metric.calculate_score(
            output_ground_truth_difference_count=output_ground_truth_differences_count,
            input_ground_truth_difference_count=input_ground_truth_differences_count,
        )
        assert score == expected_score, f"Expected score {expected_score}, got {score}"

    count_differences_params = [
        (["S", "1", "1", "1"], ["S", "1", "1", "1"], 0),  # No differences
        (["S", "1", "1", "1"], ["0", "1", "1", "1"], 1),  # One difference
        (["S", "1", "1", "1"], ["L", "V", "S", "1"], 3),  # One difference at the start
    ]

    @pytest.mark.parametrize("list_1, list_2, expected_difference_count", count_differences_params)
    def test_count_differences(
        self,
        list_1: list[str],
        list_2: list[str],
        expected_difference_count: int,
        grid_difference_metric: GridDifference,
    ) -> None:
        difference_count = grid_difference_metric.count_differences(list_1, list_2)
        assert difference_count == expected_difference_count, (
            f"Expected {expected_difference_count} difference, got {difference_count}"
        )

    calculate_params = [
        (
            "Below is the input grid with masked regions:\n\nL L L\nV V V\nS S S\n\nPlease output the completed grid",
            "L L L\n0 1 0\nS S S",
            "L L L\n0 1 0\nS S S",
            True,
            1.0,
            1.0,
        ),  # No differences
        (
            "Below is the input grid with masked regions:\n\nL L L\nV V V\nS S S\n\nPlease output the completed grid",
            "L L L\n0 1 0\nS S S",
            "L L L\n1 1 0\nS S S",
            False,
            (1 - (1 / 3)),
            (1 - (1 / 3)),
        ),  # 1 difference
        (
            "Below is the input grid with masked regions:\n\nL L L\nV V V\nS S S\n\nPlease output the completed grid",
            "L L L\n0 1 0\nS S S",
            "L L L\n1 0 0\nS S S",
            False,
            (1 - (2 / 3)),
            (1 - (2 / 3)),
        ),  # 2 differences
        (
            "Below is the input grid with masked regions:\n\nL L L\nV V V\nS S S\n\nPlease output the completed grid",
            "L L L\n0 1 0\nS S S",
            "L 1 L\n0 1 0\nS S S",
            False,
            (1 - (1 / 3)),
            (1 - (1 / 3)),
        ),  # Correct, but changed existing correct value, so 1 difference
        (
            "Below is the input grid with masked regions:\n\nL L L\nV V V\nS S S\n\nPlease output the completed grid",
            "L L L\n0 1 0\nS S S",
            "L 1 S\n0 0 1\nS S S",
            False,
            -(1 / 3),
            0.0,
        ),  # Not correct and changed 2 existing correct value, so 3 difference
    ]

    @pytest.mark.parametrize(
        "prompt, ground_truth, completion, expected_exact_match, expected_score, expected_normalized_score",
        calculate_params,
    )
    def test_calculate(
        self,
        prompt: str,
        ground_truth: str,
        completion: str,
        expected_exact_match: bool,
        expected_score: float,
        expected_normalized_score: float,
        grid_difference_metric: GridDifference,
    ) -> None:
        messages = [
            Message(role=Role.SYSTEM, content=""),
            Message(role=Role.USER, content=prompt),
        ]
        mock_completion = Completion(
            prompt=prompt,
            prompt_sequence_positions=None,
            messages=messages,
            raw_completion=completion,
            raw_completion_sequence_positions=None,
            id=0,
            subject="",
            ground_truth=ground_truth,
            completion=completion,
            error=None,
        )

        results = grid_difference_metric.calculate(mock_completion)
        assert len(results) == 3, (
            "Expected three result from the metric calculation: exact_match, score, and normalized_score"
        )
        # exact_match
        assert results[0].value == float(expected_exact_match), f"Expected exact match to be {expected_exact_match}"
        # score
        assert results[1].value == pytest.approx(expected_score), f"Expected score to be approximately {expected_score}"
        # normalized_score
        assert results[2].value == pytest.approx(expected_normalized_score), (
            f"Expected normalized score to be approximately {expected_normalized_score}"
        )
