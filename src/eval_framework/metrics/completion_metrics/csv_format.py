import json

from pydantic import BaseModel, Field

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion

SEPARATOR_MAP = {"comma": ",", "semicolon": ";", "space": " ", "tab": "\t"}


class CSVFormatEvaluation(BaseModel):
    implicit: bool = Field(default=False)
    has_csv: bool = Field(default=False)
    is_separator_respected: bool = Field(default=False)
    is_column_count_respected: bool = Field(default=False)


class CSVFormat(BaseMetric[Completion]):
    NAME = "CSV Format"
    KEYS = ["has_csv", "is_separator_respected", "is_column_count_respected"]

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [
                MetricResult(metric_name=f"{self.NAME}/{k}", value=None, higher_is_better=True, error=response.error)
                for k in self.KEYS
            ]

        if response.completion == "":
            return [
                MetricResult(metric_name=f"{self.NAME}/{k}", value=0.0, higher_is_better=True, error=response.error)
                for k in self.KEYS
            ]

        grading = evaluate_csv_format(response)

        results = []
        for key in self.KEYS:
            result = MetricResult(
                metric_name=f"{self.NAME}/{key}",
                value=float(getattr(grading, key)),
                higher_is_better=True,
                error=response.error,
            )
            results.append(result)
        return results


def extract_csv_from_text(text: str, min_rows: int = 2, min_columns: int = 2) -> tuple[list[str] | None, str | None]:
    lines = text.split("\n")
    delimiters = set(SEPARATOR_MAP.values())
    best_delimiter = None
    csv_lines: list[str] = []

    # Iterate over lines to find potential delimiters and consistent substring counts
    for i, line in enumerate(lines):
        for delimiter in delimiters:
            substrings = line.split(delimiter)
            if len(substrings) < min_columns:
                continue

            current_csv_lines = [line]
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                next_substrings = next_line.split(delimiter)
                if len(next_substrings) != len(substrings):
                    break
                current_csv_lines.append(next_line)
            if len(current_csv_lines) >= min_rows and len(current_csv_lines) > len(csv_lines):
                best_delimiter = delimiter
                csv_lines = current_csv_lines

    if not csv_lines:
        return None, None

    return csv_lines, best_delimiter


def evaluate_csv_format(response: Completion) -> CSVFormatEvaluation:
    expected_output = json.loads(str(response.ground_truth))

    expected_separator_code = expected_output["separator"]
    csv_lines, separator = extract_csv_from_text(response.completion)

    if not csv_lines:
        return CSVFormatEvaluation(has_csv=False, implicit=not expected_separator_code)

    csv_format_evaluation = CSVFormatEvaluation(has_csv=True, implicit=not expected_separator_code)

    if not expected_separator_code:
        csv_format_evaluation.is_separator_respected = separator in SEPARATOR_MAP.values()
    else:
        csv_format_evaluation.is_separator_respected = separator == SEPARATOR_MAP.get(expected_separator_code)

    expected_column_count = len(expected_output["columns"])
    column_counts = [len(csv_lines.split(separator)) for csv_lines in csv_lines]

    csv_format_evaluation.is_column_count_respected = all(
        column_count == expected_column_count for column_count in column_counts
    )

    return csv_format_evaluation
