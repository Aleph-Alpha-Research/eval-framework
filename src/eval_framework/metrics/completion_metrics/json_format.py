import json
from typing import Any, Mapping

import jsonschema  # type: ignore
from pydantic import BaseModel

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion


class JsonFormatEvaluation(BaseModel):
    is_just_json: bool = False
    is_valid_json: bool = False
    fulfills_schema: bool | None = None
    json_parsing_error: str | None = None
    schema_validation_error: str | None = None


class JsonFormat(BaseMetric[Completion]):
    NAME = "JSON Format"

    def calculate(self, response: Completion) -> list[MetricResult]:
        keys = [
            "is_just_json",
            "is_valid_json",
            "fulfills_schema",
        ]

        if response.error is not None:
            return [
                MetricResult(metric_name=f"{self.NAME}/{k}", value=None, higher_is_better=True, error=response.error)
                for k in keys
            ]

        if response.completion == "":
            return [
                MetricResult(metric_name=f"{self.NAME}/{k}", value=0.0, higher_is_better=True, error=response.error)
                for k in keys
            ]

        json_dict, grading = self._extract_and_parse_json(response.completion)
        schema = json.loads(str(response.ground_truth))["json_schema"]
        if schema and json_dict is None:
            grading.fulfills_schema = False
        if schema and json_dict is not None:
            grading = self._validate_json_against_schema(json_dict, schema, grading)

        results = []
        for key in keys:
            result = MetricResult(
                metric_name=f"{self.NAME}/{key}",
                value=float(getattr(grading, key)) if getattr(grading, key) is not None else None,
                higher_is_better=True,
                error=response.error,
                code_execution_trace=(grading.json_parsing_error or "") + (grading.schema_validation_error or ""),
            )
            results.append(result)
        return results

    @staticmethod
    def _validate_json_against_schema(
        json_obj: object, schema: Mapping[str, Any], evaluation_result: JsonFormatEvaluation
    ) -> JsonFormatEvaluation:
        evaluation_result = evaluation_result.model_copy(deep=True)
        try:
            jsonschema.validate(json_obj, schema)
            evaluation_result.fulfills_schema = True
        except jsonschema.exceptions.ValidationError as e:
            evaluation_result.fulfills_schema = False
            evaluation_result.schema_validation_error = type(e).__name__
        except jsonschema.exceptions.SchemaError as e:
            evaluation_result.schema_validation_error = type(e).__name__
        return evaluation_result

    @staticmethod
    def _extract_and_parse_json(completion: str) -> tuple[object, JsonFormatEvaluation]:
        evaluation_result = JsonFormatEvaluation()
        json_dict = None
        try:
            json_dict = json.loads(remove_comments(completion.strip("`")))
            evaluation_result.is_just_json = True
            evaluation_result.is_valid_json = True
        except Exception as _:
            try:
                json_string = remove_comments(get_json_object(completion))
                json_dict = json.loads(json_string)
                evaluation_result.is_valid_json = True
            except Exception as e:
                evaluation_result.json_parsing_error = type(e).__name__
        return json_dict, evaluation_result


def get_json_object(text: str) -> str:
    """
    Extract the first valid JSON object or array from text.

    This function handles nested brackets properly by using a bracket counting
    approach to find complete JSON structures, rather than using regex which
    can incorrectly match outer brackets containing non-JSON content.
    """

    def find_json_at_position(text: str, start_pos: int, open_char: str, close_char: str) -> str | None:
        """Find a complete JSON object/array starting at the given position."""
        if start_pos >= len(text) or text[start_pos] != open_char:
            return None

        bracket_count = 0
        in_string = False
        escaped = False

        for i in range(start_pos, len(text)):
            char = text[i]

            if escaped:
                escaped = False
                continue

            if char == "\\" and in_string:
                escaped = True
                continue

            if char == '"' and not escaped:
                in_string = not in_string
                continue

            if not in_string:
                if char == open_char:
                    bracket_count += 1
                elif char == close_char:
                    bracket_count -= 1
                    if bracket_count == 0:
                        # Found complete JSON structure
                        candidate = text[start_pos : i + 1]
                        # Test if it's valid JSON
                        try:
                            json.loads(candidate)
                            return candidate
                        except json.JSONDecodeError:
                            return None

        return None

    # Look for JSON objects {} and arrays []
    json_candidates = []

    # Search for objects starting with {
    for i in range(len(text)):
        if text[i] == "{":
            candidate = find_json_at_position(text, i, "{", "}")
            if candidate:
                json_candidates.append(candidate)

    # Search for arrays starting with [
    for i in range(len(text)):
        if text[i] == "[":
            candidate = find_json_at_position(text, i, "[", "]")
            if candidate:
                json_candidates.append(candidate)

    if not json_candidates:
        raise RuntimeError(f"No valid JSON object found in {text}.")

    # Return the longest valid JSON (most likely to be the main content)
    return max(json_candidates, key=len)


def remove_comments(text: str, comment_indicator: str = "//") -> str:
    lines = text.splitlines()
    lines = [line.split(comment_indicator)[0] for line in lines]
    return "\n".join([line for line in lines if line.strip()])
