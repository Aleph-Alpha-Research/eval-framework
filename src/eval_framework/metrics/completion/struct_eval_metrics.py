import csv
import io
import json
import tomllib
from typing import Any

import xmltodict
import yaml
from lxml import etree

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, extract_context_metric


class StructMetricContext(BaseMetricContext):
    output_type: str
    paths: list[str]


class StructMetric(BaseMetric[Completion]):
    NAME = "StructMetric"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, StructMetricContext)

        output_type = context.output_type

        try:
            match output_type.lower():
                case "json":
                    result = json.loads(response.completion)
                case "yaml":
                    result = list(yaml.safe_load_all(response.completion))
                    if isinstance(result, list) and len(result) == 1:
                        result = result[0]
                    else:
                        raise yaml.YAMLError("Multiple documents found in YAML")
                case "toml":
                    result = tomllib.loads(response.completion)
                case "xml":
                    result = xmltodict.parse(response.completion)
                case "csv":
                    csv_output = csv.DictReader(io.StringIO(response.completion))
                    # Check for unclosed quotes
                    if response.completion.count('"') % 2 != 0:
                        raise csv.Error("Unclosed quote in CSV")
                    if not csv_output.fieldnames:
                        raise csv.Error("CSV has no headers")
                    result = {"csv_headers": csv_output.fieldnames, "csv_rows": list(csv_output)}
                case _:
                    raise ValueError(f"Unsupported format: {output_type}")
            valid_format = 1.0
        except (json.JSONDecodeError, yaml.YAMLError, tomllib.TOMLDecodeError, csv.Error, Exception):
            valid_format = 0.0

        has_required_fields = 0.0
        if valid_format == 1:
            # assert "paths" in response.eval_kwargs, "Paths must be provided in eval_kwargs"
            assert context.paths is not None, "Paths must be provided in context"
            paths = context.paths
            assert isinstance(paths, list), "Paths must be a list of strings"
            valid_paths = 0
            for path in paths:
                if path_exists(result, path):
                    valid_paths += 1
            has_required_fields = valid_paths / len(paths) if paths else 1.0

        return [
            MetricResult(
                metric_name=f"{self.NAME}/valid_format",
                value=valid_format,
                higher_is_better=True,
            ),
            MetricResult(
                metric_name=f"{self.NAME}/has_keywords",
                value=has_required_fields,
                higher_is_better=True,
            ),
        ]


def is_valid_html(html: str) -> bool:
    parser = etree.HTMLParser(recover=False)
    try:
        etree.fromstring(html.encode("utf-8"), parser)
    except etree.XMLSyntaxError:
        return False
    return len(parser.error_log) == 0


class RenderableStructMetricContext(BaseMetricContext):
    output_type: str
    keywords: list[str]


class RenderableStructMetric(StructMetric):
    NAME = "RenderableStructMetric"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, RenderableStructMetricContext)

        output_type = context.output_type

        valid_format = 0.0
        match output_type.lower():
            case "html":
                valid_format = float(is_valid_html(response.completion))
            case _:
                raise ValueError(f"Unsupported format for RenderableStructMetric: {output_type}")

        assert context.keywords is not None, "Keywords must be provided in context"
        keywords = context.keywords
        assert isinstance(keywords, list), "Keywords must be a list of strings"
        has_keywords = 1.0
        if keywords:
            has_keywords = sum(1 for keyword in keywords if keyword.lower() in response.completion.lower()) / len(
                keywords
            )

        return [
            MetricResult(
                metric_name=f"{self.NAME}/valid_format",
                value=valid_format,
                higher_is_better=True,
            ),
            MetricResult(
                metric_name=f"{self.NAME}/has_keywords",
                value=has_keywords,
                higher_is_better=True,
            ),
        ]


# adapted from: https://github.com/TIGER-AI-Lab/StructEval/blob/main/structeval/eval_engine/eval_utils.py
def tokenize_path(path: str) -> list[str]:
    """
    Tokenize a dot-notation path, handling back-ticks and array indices.

    Args:
        path: The path string (e.g. "users.0.name" or "users[0].name")

    Returns:
        List of path tokens
    """
    # Special‑case: treat CSV header paths as a single token
    if path.startswith("csv::"):
        return [path]

    tokens, buf, in_bt = [], "", False
    i, n = 0, len(path)

    while i < n:
        ch = path[i]

        # Toggle back-tick state
        if ch == "`":
            in_bt = not in_bt
            i += 1
            continue

        # Dot delimiter (when not inside back-ticks)
        if ch == "." and not in_bt:
            if buf:
                tokens.append(buf)
                buf = ""
            i += 1
            continue

        # Bracket "[index]" treated as separate token
        if ch == "[" and not in_bt:
            if buf:
                tokens.append(buf)
                buf = ""
            j = path.find("]", i)
            if j == -1:
                raise ValueError(f"Unclosed '[' in path: {path}")
            tokens.append(path[i : j + 1])  # e.g. "[0]"
            i = j + 1
            continue

        # Regular character
        buf += ch
        i += 1

    if buf:
        tokens.append(buf)
    return tokens


# adapted from: https://github.com/TIGER-AI-Lab/StructEval/blob/main/structeval/eval_engine/eval_utils.py
def path_exists(data: Any, path: str) -> bool:
    """
    Check if a path exists in a structured data object.

    Args:
        data: The structured data to check
        path: The path to check (dot notation)

    Returns:
        True if path exists, False otherwise
    """
    tokens = tokenize_path(path)

    def walk(node: Any, toks: list[str]) -> bool:
        if not toks:
            return True
        tok, *rest = toks

        # CSV header rule (root level only)
        if isinstance(node, dict) and "csv_headers" in node and tok.startswith("csv::"):
            header = tok[5:]
            return header in node["csv_headers"] and not rest  # must be terminal

        # Wildcard
        if tok == "*":
            if isinstance(node, list):
                return any(walk(item, rest) for item in node)
            return False

        # Fixed index [n]
        if tok.startswith("[") and tok.endswith("]"):
            try:
                idx = int(tok[1:-1])
            except ValueError:
                return False
            return isinstance(node, list) and 0 <= idx < len(node) and walk(node[idx], rest)

        # Dict key handling (JSON/YAML/TOML/XML)
        if isinstance(node, dict):
            # 1️⃣ Literal key match (works for "@id" too)
            if tok in node:
                return walk(node[tok], rest)

            # 2️⃣ XML attribute fallback: "@id" → "id"
            if tok.startswith("@"):
                attr = tok[1:]
                if attr in node:
                    return walk(node[attr], rest)

        return False

    return walk(data, tokens)
