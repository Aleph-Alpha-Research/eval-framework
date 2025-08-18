import re

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class GridDifference(BaseMetric[Completion]):
    NAME = "grid_difference"

    def count_differences(self, character_list_1: list[str], character_list_2: list[str]) -> int:
        count = 0
        for character_1, character_2 in zip(character_list_1, character_list_2):
            if character_1 != character_2:
                count += 1
        return count

    def calculate_score(
        self, output_ground_truth_difference_count: int, input_ground_truth_difference_count: int
    ) -> float:
        if output_ground_truth_difference_count == 0 and input_ground_truth_difference_count == 0:
            return 1.0
        score = 1.0 - (float(output_ground_truth_difference_count) / float(input_ground_truth_difference_count))
        return score

    def extract_grid_from_prompt(self, prompt: str) -> str:
        # Extract grid between known markers
        start_marker = "Below is the input grid with masked regions:"
        end_marker = "Please output the completed grid"

        # Use regex with DOTALL flag to match across newlines
        match = re.search(re.escape(start_marker) + r"(.*?)" + re.escape(end_marker), prompt, re.DOTALL)

        if match:
            grid = match.group(1).strip()
            return grid

        return ""

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        input_grid = self.extract_grid_from_prompt(prompt=response.last_user_instruction).split()
        output_grid = response.completion.split()

        assert response.ground_truth_list[0], "Ground truth list is empty or not provided in the response."
        ground_truth_grid = response.ground_truth_list[0].split()

        input_ground_truth_differences_count = self.count_differences(input_grid, ground_truth_grid)
        output_ground_truth_differences_count = self.count_differences(output_grid, ground_truth_grid)

        exact_match = True
        score = 1.0
        normalized_score = 1.0
        if output_ground_truth_differences_count != 0:
            exact_match = False
            score = self.calculate_score(
                output_ground_truth_differences_count,
                input_ground_truth_differences_count,
            )
            normalized_score = max(score, 0.0)

        return [
            MetricResult(
                metric_name=f"{self.NAME}_exact_match",
                value=float(exact_match),
                higher_is_better=True,
                error=response.error,
            ),
            MetricResult(metric_name=f"{self.NAME}_score", value=score, higher_is_better=True, error=response.error),
            MetricResult(
                metric_name=f"{self.NAME}_normalized_score",
                value=normalized_score,
                higher_is_better=True,
                error=response.error,
            ),
        ]
