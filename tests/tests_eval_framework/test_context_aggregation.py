"""Aggregators see each sample's context fields as ``context.*`` DataFrame columns, joined in memory from the
responses, so they can group across samples by item fields (e.g. per question and model) without anything
being added to ``Result`` or written to disk."""

import json
from pathlib import Path
from statistics import median
from typing import Any, ClassVar

import pandas as pd
import pytest
import wandb

from eval_framework.evaluation_generator import EvaluationGenerator, context_columns, flatten_context
from eval_framework.metrics.aggregators.aggregators import Aggregator, IdentifierMean, Identity, PassAtK
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.result_processors.base import Result
from eval_framework.result_processors.result_processor import ResultsFileProcessor
from eval_framework.shared.types import BaseMetricContext, Completion, Loglikelihood
from eval_framework.tasks.eval_config import EvalConfig
from tests.tests_eval_framework.conftest import MockLLM

# One row per (question, model, correct response) and that response's score: several rows per (question, model),
# a different number per pair.
_SCORES: dict[tuple[str, str], list[float]] = {
    ("q1", "model-a"): [0.2, 0.9, 0.4],
    ("q1", "model-b"): [0.6, 0.1, 0.3, 0.9],
    ("q2", "model-a"): [0.8, 0.2],
    ("q2", "model-b"): [1.0, 0.0, 0.7, 0.3, 0.6],
}
# The score must be the mean of the per-(question, model) medians ...
_EXPECTED = sum(median(scores) for scores in _SCORES.values()) / len(_SCORES)
# ... not the row-level mean, which is what grouping per sample would give.
_ROW_MEAN = sum(s for scores in _SCORES.values() for s in scores) / sum(len(s) for s in _SCORES.values())


class _ItemContext(BaseMetricContext):
    question_id: str
    model: str
    score: float
    test_cases: list[str] = []  # bulky, non-scalar: must not become a column


class _OtherContext(BaseMetricContext):
    language: str
    weights: dict[str, float] = {}


class _MedianPerQuestionAndModel(Aggregator):
    """Collapses all rows of one (question, model) into their median, whatever sample they came from."""

    name = "MedianPerQuestionAndModel"
    _KEYS: ClassVar[list[str]] = ["context.question_id", "context.model"]

    def __call__(self, response_df: pd.DataFrame, identifier_columns: list[str], **kwargs: Any) -> pd.DataFrame:
        if response_df[self._KEYS].isna().any(axis=None):
            raise ValueError("rows without a question_id or model in their context")
        group = [c for c in identifier_columns if c != "problem"] + self._KEYS
        other_cols = [c for c in response_df.columns if c not in group and c != "value"]
        return response_df.groupby(group).agg({"value": "median", **{c: "first" for c in other_cols}}).reset_index()


class _ScoreFromContext(BaseMetric[Completion | Loglikelihood]):
    """Stands in for e.g. BPB: one score per row, read from the item. Counts calls, to tell a resumed run's reuse
    of results.jsonl from a recomputation."""

    NAME = "Score"
    AGGREGATORS = [_MedianPerQuestionAndModel()]
    calls = 0

    def calculate(self, response: Completion | Loglikelihood) -> list[MetricResult]:
        type(self).calls += 1
        assert isinstance(response.context, BaseMetricContext)
        score = response.context.model_dump()["score"]
        return [MetricResult(metric_name=self.NAME, value=score, higher_is_better=True)]


def _completion(id: int, context: BaseMetricContext | list[BaseMetricContext] | None, subject: str = "s") -> Completion:
    return Completion(
        id=id,
        subject=subject,
        ground_truth=None,
        messages=None,
        prompt="prompt",
        prompt_num_tokens=1,
        completion="completion",
        raw_completion="completion",
        raw_completion_num_tokens=1,
        context=context,
    )


def _stub_responses() -> list[Completion]:
    contexts = [
        _ItemContext(question_id=q, model=m, score=s, test_cases=["assert x"])
        for (q, m), scores in _SCORES.items()
        for s in scores
    ]
    return [_completion(i, c) for i, c in enumerate(contexts)]


def _config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(output_dir=tmp_path, num_fewshot=0, num_samples=2, task_name="HumanEval_OLMES", llm_class=MockLLM)


def _run_eval(tmp_path: Path, output_dir: Path) -> dict[str, float | None]:
    evaluator = EvaluationGenerator(_config(tmp_path), ResultsFileProcessor(output_dir))
    evaluator.metrics = [_ScoreFromContext]
    evaluator.run_eval()
    return json.loads((output_dir / "aggregated_results.json").read_text())


def test_median_per_question_and_model_then_mean(tmp_path: Path, wandb_run: wandb.Run) -> None:
    output_dir = tmp_path / "out"
    processor = ResultsFileProcessor(output_dir)
    processor.save_metadata({"llm_name": "llm"})
    processor.save_responses(list(_stub_responses()))

    aggregated = _run_eval(tmp_path, output_dir)

    assert aggregated["MedianPerQuestionAndModel _ScoreFromContext.Score"] == pytest.approx(_EXPECTED)
    assert aggregated["MedianPerQuestionAndModel _ScoreFromContext.Score (micro)"] == pytest.approx(_EXPECTED)
    assert abs(_EXPECTED - _ROW_MEAN) > 1e-6  # the test would notice per-row grouping
    # Nothing about the context reaches results.jsonl.
    assert all("context" not in json.loads(line) for line in (output_dir / "results.jsonl").read_text().splitlines())


def test_a_run_resumed_from_saved_results_gets_the_columns_from_the_responses(
    tmp_path: Path, wandb_run: wandb.Run
) -> None:
    output_dir = tmp_path / "out"
    processor = ResultsFileProcessor(output_dir)
    processor.save_metadata({"llm_name": "llm"})
    processor.save_responses(list(_stub_responses()))
    first = _run_eval(tmp_path, output_dir)
    calls_after_first_run = _ScoreFromContext.calls

    resumed = _run_eval(tmp_path, output_dir)

    assert _ScoreFromContext.calls == calls_after_first_run  # results came from results.jsonl, not recomputed
    assert resumed == first


def test_existing_aggregators_give_the_same_numbers_with_and_without_context(tmp_path: Path) -> None:
    class _Metric(BaseMetric[Completion | Loglikelihood]):
        NAME = "ExactMatch"
        AGGREGATORS = [PassAtK(k=1), PassAtK(k=2), IdentifierMean(), Identity()]

        def calculate(self, response: Completion | Loglikelihood) -> list[MetricResult]:
            return []

    config = _config(tmp_path).model_copy(update={"repeats": 2})
    evaluator = EvaluationGenerator(config, ResultsFileProcessor(tmp_path))
    evaluator.metrics = [_Metric]
    rows = [("s1", "k1", 1.0), ("s1", "k1", 0.0), ("s1", "k2", 1.0), ("s1", "k2", 1.0), ("s2", "k1", 0.0)]
    results = [
        Result(
            id=i,
            subject=subject,
            num_fewshot=0,
            llm_name="llm",
            task_name="t",
            metric_class_name="_Metric",
            metric_name="ExactMatch",
            key=key,
            value=value,
            higher_is_better=True,
        )
        for i, (subject, key, value) in enumerate(rows)
    ]
    # Some rows with context (string and number fields, two context types), some without.
    with_context = context_columns(
        [
            _completion(0, _ItemContext(question_id="q1", model="a", score=0.5), subject="s1"),
            _completion(1, [_ItemContext(question_id="q2", model="b", score=0.1), _OtherContext(language="de")], "s1"),
            _completion(4, _OtherContext(language="en"), subject="s2"),
        ]
    )

    assert evaluator._aggregate_results_with_aggregators(
        results, with_context
    ) == evaluator._aggregate_results_with_aggregators(results, {})


def test_flatten_a_single_context_keeps_its_scalar_fields() -> None:
    context = _ItemContext(question_id="q1", model="a", score=0.5, test_cases=["assert x"])

    assert flatten_context(context) == {"question_id": "q1", "model": "a", "score": 0.5}


def test_flatten_a_list_of_contexts_merges_their_fields() -> None:
    contexts: list[BaseMetricContext] = [
        _ItemContext(question_id="q1", model="a", score=0.5),
        _OtherContext(language="de", weights={"a": 1.0}),
    ]

    assert flatten_context(contexts) == {"question_id": "q1", "model": "a", "score": 0.5, "language": "de"}


def test_flatten_rejects_a_field_defined_by_two_contexts_in_a_list() -> None:
    contexts: list[BaseMetricContext] = [_OtherContext(language="de"), _OtherContext(language="en")]

    with pytest.raises(ValueError, match="'language' is defined by more than one context"):
        flatten_context(contexts)


def test_flatten_leaves_out_lists_dicts_nested_models_and_none() -> None:
    class _Nested(BaseMetricContext):
        inner: _OtherContext
        maybe: str | None = None
        flag: bool = True
        count: int = 3

    context = _Nested(inner=_OtherContext(language="de"))

    assert flatten_context(context) == {"flag": True, "count": 3}
    assert flatten_context(None) == {}


def test_flatten_a_context_read_back_from_json(tmp_path: Path) -> None:
    # Read back from output.jsonl, a context is a plain BaseMetricContext whose fields are extras.
    processor = ResultsFileProcessor(tmp_path)
    processor.save_responses(
        [_completion(0, [_ItemContext(question_id="q1", model="a", score=0.5), _OtherContext(language="de")])]
    )

    [loaded] = processor.load_responses()

    assert isinstance(loaded.context, list)
    assert type(loaded.context[0]) is BaseMetricContext
    assert flatten_context(loaded.context) == {"question_id": "q1", "model": "a", "score": 0.5, "language": "de"}
