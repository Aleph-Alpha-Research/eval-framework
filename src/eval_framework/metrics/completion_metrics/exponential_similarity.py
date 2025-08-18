from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion, Error


class ExponentialSimilarity(BaseMetric[Completion]):
    NAME = "ExponentialSimilarity"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        ground_truths = [gt for gt in response.ground_truth_list if gt is not None]
        if not ground_truths:
            return [MetricResult(metric_name=self.NAME, value=0.0, higher_is_better=True, error=response.error)]

        try:
            # Try to calculate exponential similarity for each ground truth
            similarities = []
            for gt in ground_truths:
                try:
                    gt_float = float(gt)
                    completion_float = float(response.completion)
                    similarities.append(calculate_exponential_similarity(gt_float, completion_float))
                except (ValueError, TypeError):
                    # Skip this ground truth if conversion fails
                    continue

            # If we have any valid similarities, return the max
            if similarities:
                return [
                    MetricResult(
                        metric_name=self.NAME, value=max(similarities), higher_is_better=True, error=response.error
                    )
                ]
            else:
                # If all conversions failed, return an error
                error = Error(
                    error_class="ValueError",
                    message="Could not convert ground truth or completion to float",
                    traceback="",
                )
                return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=error)]
        except Exception as e:
            error = Error(error_class=e.__class__.__name__, message=str(e), traceback="")
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=error)]


def calculate_exponential_similarity(p_true: float, p_pred: float) -> float:
    """
    Compute the exponential similarity (SpaceDigest version) between
    the gold percentage and predicted value.

    Parameters:
    - p_true (float): The gold/reference percentage.
    - p_pred (float): The predicted scalar.
    - d (float): Base of the exponent. Default is 2.
    - c (float): Coefficient in exponent. Default is 10.

    Returns:
    - float: Similarity score between 0 and 1.
    """
    d = 2
    c = 10

    return d ** (-c * abs(p_true / 100 - p_pred / 100))
