import logging
import math

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from eval_framework.metrics.base import BaseMetric
from eval_framework.metrics.efficiency.bytes_per_sequence_position import (
    BytesCompletion,
    BytesLoglikelihood,
    SequencePositionsCompletion,
    SequencePositionsLoglikelihood,
)
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric
from eval_framework.result_processors.base import Result, ResultProcessor
from eval_framework.shared.types import Completion, Loglikelihood
from eval_framework.tasks.base import ResponseType
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.registry import registry
from eval_framework.utils.constants import RED, RESET
from eval_framework.utils.tqdm_handler import get_disable_bar_flag, safe_tqdm_write

logger = logging.getLogger(__name__)


def needs_std_err(metric_name: str) -> bool:
    """Standard error is only computed for non-efficiency metrics."""
    return not ("SequencePositions" in metric_name or "Bytes" in metric_name)


def error_free_ratio(df: pd.DataFrame) -> float:
    """Fraction of rows in ``df`` whose ``error`` column is null."""
    return float(len(df[df["error"].isnull()]) / len(df))


def mean_over_key_subject(df: pd.DataFrame) -> float:
    """Mean of per-(key, subject) means, giving every key/subject group equal weight.

    ``df`` is expected to be error-free and contain ``key``, ``subject`` and ``value`` columns.
    """
    key_subject_mean = df.groupby(["key", "subject"]).mean()
    return float(key_subject_mean[["value"]].mean()["value"])


def mean_treating_errors_as_zero(df: pd.DataFrame) -> float:
    """Like :func:`mean_over_key_subject` but errored samples contribute a value of ``0.0``.

    ``df`` must contain ``key``, ``subject``, ``value`` and ``error`` columns. Only rows with an
    error have their (``None``) value replaced by ``0.0``; genuinely missing values stay ``NaN``.
    """
    with_errors = df[["key", "subject", "value", "error"]].copy()
    error_mask = with_errors["error"].notna()
    with_errors.loc[error_mask, "value"] = with_errors.loc[error_mask, "value"].fillna(0.0)
    key_subject_mean = with_errors.groupby(["key", "subject"])["value"].mean()
    return float(key_subject_mean.mean())


def std_and_num_samples(df: pd.DataFrame) -> tuple[float, int]:
    """Return the mean per-(key, subject) standard deviation and the sample count of ``df``.

    ``df`` is expected to be error-free and contain ``key``, ``subject`` and ``value`` columns.
    """
    key_subject_std = df.groupby(["key", "subject"]).std()
    std = float(key_subject_std[["value"]].mean()["value"])
    return std, len(df)


def select_metrics(response_type: ResponseType, task_metrics: list[type[BaseMetric]]) -> list[type[BaseMetric]]:
    """Return the task metrics plus the efficiency metrics matching the response type."""
    if response_type == ResponseType.COMPLETION:
        return task_metrics + [BytesCompletion, SequencePositionsCompletion]
    elif response_type == ResponseType.LOGLIKELIHOODS:
        return task_metrics + [BytesLoglikelihood, SequencePositionsLoglikelihood]
    raise ValueError(f"Unsupported response type: {response_type!r}")


class EvaluationGenerator:
    def __init__(self, config: EvalConfig, result_processor: ResultProcessor) -> None:
        logger.info("EvaluationGenerator initialized")

        self.few_shot = config.num_fewshot
        self.config = config
        self.num_samples = config.num_samples
        self.max_tokens = config.max_tokens
        self.result_processor = result_processor
        self.save_intermediate_results = config.save_intermediate_results

        eval_ = registry()[config.task_name]
        self.metrics = select_metrics(eval_.response_type(), eval_.metrics())
        self.task_name = eval_.display_name()

    def _find_metric_class(self, metric_class_name: str) -> type[BaseMetric]:
        """Return the metric class in ``self.metrics`` whose name matches ``metric_class_name``."""
        for metric_class in self.metrics:
            if metric_class.__name__ == metric_class_name:
                return metric_class
        raise ValueError(f"Metric class {metric_class_name!r} not found in metrics list.")

    def _run_metric_calculators(self, responses: list[Completion | Loglikelihood]) -> list[Result]:
        results: list[Result] = self.result_processor.load_metrics_results()
        llm_name = self.result_processor.load_metadata()["llm_name"]

        subject_result_id_existing = set()
        for result in results:
            subject_result_id_existing.add(f"{result.subject}_{result.id}_{result.metric_class_name}")

        """
        we have three dimensions: subject, metric, sample_id
        we wanna average over sample_id
        and also over all subjects by averaging over the averages
        dict[metric, dict[subject, dict[sample_id, list[result]]]]
        """
        llm_judge = None
        for metric_class in self.metrics:
            metric: BaseMetric
            if issubclass(metric_class, BaseLLMJudgeMetric):
                if llm_judge is None:
                    assert self.config.llm_judge_class is not None, "The llm_judge_class must be defined in the config."
                    llm_judge = self.config.llm_judge_class(**self.config.judge_model_args)
                metric = metric_class(
                    llm_judge=llm_judge,
                    randomize_order=self.config.randomize_judge_order,
                )
            else:
                metric = metric_class()
            metric.fail_on_error = self.config.fail_on_error

            logger.info(f"Starting calculation of {metric.NAME}")
            safe_tqdm_write(f"INFO: Calculating {metric.NAME}")
            for response in tqdm(responses, desc=f"Calculating {metric.NAME}", disable=get_disable_bar_flag()):
                if f"{response.subject}_{response.id}_{metric.__class__.__name__}" in subject_result_id_existing:
                    continue

                subject = response.subject
                metric_results = metric.calculate(response)
                for metric_result in metric_results:
                    if "/" in metric_result.metric_name:
                        metric_name, key = metric_result.metric_name.split("/")
                    else:
                        metric_name = metric_result.metric_name
                        key = None
                    completion = response.completion if isinstance(response, Completion) else str(response.ground_truth)

                    result = Result(
                        id=response.id,
                        metric_class_name=metric.__class__.__name__,
                        metric_name=metric_name,
                        num_fewshot=self.few_shot,
                        key=key,
                        subject=subject,
                        llm_name=llm_name,
                        task_name=self.task_name,
                        value=metric_result.value,
                        higher_is_better=metric_result.higher_is_better,
                        prompt=response.prompt,
                        response=completion,
                        llm_judge_prompt=metric_result.llm_judge_prompt,
                        llm_judge_response=metric_result.llm_judge_response,
                        code_execution_trace=metric_result.code_execution_trace,
                        error=metric_result.error,
                    )
                    results.append(result)
                    if self.save_intermediate_results:
                        self.result_processor.save_metrics_result(result)

            logger.info(f"Completed calculation of {metric.NAME}")
            safe_tqdm_write(f"INFO: Completed {metric.NAME}")

        if not self.save_intermediate_results:
            self.result_processor.save_metrics_results(results)
        return results

    def _aggregate_results(self, results: list[Result]) -> dict[str, float | None]:
        data = pd.DataFrame(
            [
                {
                    "metric_name": r.metric_name,
                    "subject": r.subject,
                    "key": r.key,
                    "value": r.value,
                    "error": r.error,
                }
                for r in results
            ]
        )
        if len(data) == 0:
            return {}
        data.fillna({"key": ""}, inplace=True)
        metrics = sorted(data["metric_name"].unique())
        aggregated_results: dict[str, float | None] = {}

        for metric in metrics:
            # filter for metric
            data_subset = data[data["metric_name"] == metric][["subject", "key", "value", "error"]]

            # filter and count errors
            mask = data_subset["error"].isnull()
            data_subset_error_free = data_subset.loc[mask, ["subject", "key", "value"]]

            metric_error_free_ratio = error_free_ratio(data_subset)
            aggregated_results[f"ErrorFreeRatio {metric}"] = metric_error_free_ratio

            # aggregate by key and subject first to have equal weights for all key / subject combinations
            key_subject_mean = data_subset_error_free.groupby(["key", "subject"]).mean()
            aggregated_results[f"Average {metric}"] = mean_over_key_subject(data_subset_error_free)

            if metric_error_free_ratio < 1.0:
                # Treat error samples (with value=None) as 0 for the "including errors" average
                aggregated_results[f"Average {metric} (including Errors)"] = mean_treating_errors_as_zero(data_subset)

            std_err_mean_sum_of_squares = 0.0
            std_err_mean_total_num_samples = 0.0
            std_err_mean_num_subjects = 0

            for column in ["key", "subject"]:
                if len(data_subset[column].unique()) > 1:
                    for name, _group in key_subject_mean.groupby([column]):
                        mask = data_subset[column] == name[0]
                        group = data_subset.loc[mask, ["subject", "key", "value", "error"]]
                        group_error_free = group[group["error"].isnull()][["subject", "key", "value"]]
                        group_error_free_ratio = error_free_ratio(group)
                        aggregated_results[f"ErrorFreeRatio {metric} - {name[0]}"] = group_error_free_ratio

                        value = mean_over_key_subject(group_error_free)
                        aggregated_results[f"Average {metric} - {name[0]}"] = value if not math.isnan(value) else None

                        if group_error_free_ratio < 1.0:
                            # Treat error samples (with value=None) as 0 for the "including errors" average
                            value_with_errors = mean_treating_errors_as_zero(group)
                            aggregated_results[f"Average {metric} (including Errors) - {name[0]}"] = (
                                value_with_errors if not math.isnan(value_with_errors) else None
                            )

                        if needs_std_err(metric):
                            # calculate standard error for selected  metrics
                            std, num_samples = std_and_num_samples(group_error_free)

                            if math.isnan(std) or num_samples == 0:
                                aggregated_results[f"StdErr {metric} - {name[0]}"] = None
                            else:
                                aggregated_results[f"StdErr {metric} - {name[0]}"] = std / np.sqrt(num_samples)
                            aggregated_results[f"NumSamples {metric} - {name[0]}"] = num_samples

                            std_err_mean_sum_of_squares += std**2 / num_samples
                            std_err_mean_total_num_samples += num_samples
                            std_err_mean_num_subjects += 1

            if needs_std_err(metric):
                # calculate standard error for selected  metrics
                if std_err_mean_total_num_samples > 0:
                    # calculate the standard error of the mean (SEM) for the aggregated results (eg. add in quadrature)
                    # SEM = sqrt(sum(variance_i * n_i) / i)
                    # where variance_i is the variance of each group and i is the number of groups
                    # (the combined mean is also not weighted by the number of samples)
                    if math.isnan(std) or std_err_mean_total_num_samples == 0:
                        aggregated_results[f"StdErr {metric}"] = None
                    else:
                        aggregated_results[f"StdErr {metric}"] = np.sqrt(
                            std_err_mean_sum_of_squares / std_err_mean_num_subjects
                        )
                    aggregated_results[f"NumSamples {metric}"] = std_err_mean_total_num_samples
                else:
                    # if there are no sub-groups to combine, calculate the SEM here directly
                    std, num_samples = std_and_num_samples(data_subset_error_free)
                    if math.isnan(std) or num_samples == 0:
                        aggregated_results[f"StdErr {metric}"] = None
                    else:
                        aggregated_results[f"StdErr {metric}"] = std / np.sqrt(num_samples)
                    aggregated_results[f"NumSamples {metric}"] = num_samples

        if (
            "Average Bytes" in aggregated_results
            and "Average SequencePositions" in aggregated_results
            and aggregated_results["Average Bytes"]
            and aggregated_results["Average SequencePositions"]
        ):
            aggregated_results["Average Bytes per Sequence Position"] = (
                aggregated_results["Average Bytes"] / aggregated_results["Average SequencePositions"]
            )

        return aggregated_results

    def _aggregate_results_with_aggregators(self, results: list[Result]) -> dict[str, float | None]:
        data = pd.DataFrame(
            [
                {
                    "metric_name": r.metric_name,
                    "metric_class_name": r.metric_class_name,
                    "subject": r.subject,
                    "key": r.key,
                    "value": r.value,
                    "error": r.error,
                    "prompt": r.prompt,
                }
                for r in results
            ]
        )
        if len(data) == 0:
            return {}
        data = data.fillna({"key": ""})
        aggregated_results: dict[str, float | None] = {}
        data = data.loc[data.error.isnull()]

        for (metric_name, current_metric_class), metric_group in data.groupby(["metric_name", "metric_class_name"]):
            # The reason we groupby over both metric_name and metric_class_name is because we want to aggregate
            # results for a single metric. Two metric classes can implement the same metric name. We want to separate
            # those cases. We cannot group over only metric_class_name because each metric class can implement
            # multiple metrics with different names.
            current_metric = self._find_metric_class(current_metric_class)

            for aggregator in current_metric.AGGREGATORS:
                aggregated_results[f"{aggregator.name} {current_metric_class}.{metric_name}"] = (
                    aggregator(metric_group, ["prompt"])  # Compute the aggregator, grouped by the prompt...
                    .groupby(["key", "subject"])  # ... then group by key, subject...
                    .agg({"value": "mean"})["value"]  # ...and average scores over each key, subject group...
                    .mean()  # ...and lastly average the scores across all groups giving equal weight to every
                    .item()  # key, subject group.
                )

        # Loop to additionally compute per-subject/per-key breakdown metric scores, e.g. for only subject="algebra"
        for (key, subject, metric_name, current_metric_class), ksm_group in data.groupby(
            ["key", "subject", "metric_name", "metric_class_name"]
        ):
            current_metric = self._find_metric_class(current_metric_class)

            for aggregator in current_metric.AGGREGATORS:
                save_string = (
                    f"{aggregator.name} {metric_name} - {subject}"
                    if not key
                    else f"{aggregator.name} {metric_name} - {key} - {subject}"
                )
                aggregated_results[save_string] = aggregator(ksm_group, ["prompt"])["value"].mean().mean().item()

        return aggregated_results

    def run_eval(self) -> list[Result]:
        """Runs evaluation using saved completions."""
        logger.info("Running evaluation...")
        responses = self.result_processor.load_responses()
        if not responses:
            raise ValueError("No saved completions found. Run 'run_completions' first.")

        metrics_results = self._run_metric_calculators(responses)
        del responses
        aggregated_results = self._aggregate_results(metrics_results)
        results_with_aggregators = self._aggregate_results_with_aggregators(metrics_results)
        aggregated_results.update(results_with_aggregators)

        wandb.log(aggregated_results)
        self.result_processor.save_aggregated_results(aggregated_results)
        logger.info(aggregated_results)
        logger.info(f"{RED}[ Evaluation completed and results saved! ]{RESET}")
        return metrics_results
