from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion


class PhoenixIdentityContext(BaseMetricContext):
    """Context for PhoenixIdentityMetric — carries expected keywords per question."""

    required_any: list[str]
    """At least one of these strings must appear in the model response (case-insensitive)."""


class PhoenixIdentityMetric(BaseMetric[Completion]):
    """Checks whether a model response contains at least one expected identity keyword.

    Scores 1.0 if any keyword in ``required_any`` appears in the completion
    (case-insensitive substring match), 0.0 otherwise.
    """

    NAME = "phoenix_identity"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        assert isinstance(response.context, PhoenixIdentityContext), (
            f"Expected PhoenixIdentityContext, got {type(response.context)}"
        )

        completion_lower = response.completion.lower()
        is_correct = any(kw.lower() in completion_lower for kw in response.context.required_any)
        return [MetricResult(metric_name=self.NAME, value=float(is_correct), higher_is_better=True)]
