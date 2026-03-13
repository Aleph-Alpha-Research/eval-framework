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
from eval_framework.tasks.registry import get_task
from eval_framework.utils.constants import RED, RESET
from eval_framework.utils.tqdm_handler import get_disable_bar_flag, safe_tqdm_write

logger = logging.getLogger(__name__)


class EvaluationGenerator:
    def __init__(self, config: EvalConfig, result_processor: ResultProcessor) -> None:
        logger.info("EvaluationGenerator initialized")

        self.few_shot = config.num_fewshot
        self.config = config
        self.num_samples = config.num_samples
        self.max_tokens = config.max_tokens
        self.result_processor = result_processor
        self.save_intermediate_results = config.save_intermediate_results

        task_class = get_task(config.task_name)
        if task_class.RESPONSE_TYPE == ResponseType.COMPLETION:
            self.metrics = task_class.METRICS + [BytesCompletion, SequencePositionsCompletion]
        elif task_class.RESPONSE_TYPE == ResponseType.LOGLIKELIHOODS:
            self.metrics = task_class.METRICS + [BytesLoglikelihood, SequencePositionsLoglikelihood]
        else:
            raise NotImplementedError

        self.task_name = task_class.NAME

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
        data = pd.DataFrame([r.model_dump() for r in results])
        if len(data) == 0:
            return {}
        data.fillna({"key": ""}, inplace=True)
        metrics = sorted(data["metric_name"].unique())
        aggregated_results: dict[str, float | None] = {}

        for metric in metrics:
            # filter for metric
            data_subset = data[data["metric_name"] == metric][["subject", "key", "value", "error"]]

            # filter and count errors
            total_count = len(data_subset)

            mask = data_subset["error"].isnull()
            data_subset_error_free = data_subset.loc[mask, ["subject", "key", "value"]]

            error_free_ratio = float(len(data_subset_error_free) / total_count)
            aggregated_results[f"ErrorFreeRatio {metric}"] = error_free_ratio

            # aggregate by key and subject first to have equal weights for all key / subject combinations
            key_subject_mean = data_subset_error_free.groupby(["key", "subject"]).mean()
            aggregated_results[f"Average {metric}"] = float(key_subject_mean[["value"]].mean()["value"])

            if error_free_ratio < 1.0:
                # Treat error samples (with value=None) as 0 for the "including errors" average
                data_subset_with_errors = data_subset[["key", "subject", "value", "error"]].copy()
                # Only fill value with 0 where there's an error (not for all None values)
                error_mask = data_subset_with_errors["error"].notna()
                data_subset_with_errors.loc[error_mask, "value"] = data_subset_with_errors.loc[
                    error_mask, "value"
                ].fillna(0.0)
                key_subject_mean_with_errors = data_subset_with_errors.groupby(["key", "subject"])["value"].mean()
                aggregated_results[f"Average {metric} (including Errors)"] = float(key_subject_mean_with_errors.mean())

            std_err_mean_sum_of_squares = 0.0
            std_err_mean_total_num_samples = 0.0
            std_err_mean_num_subjects = 0

            for column in ["key", "subject"]:
                if len(data_subset[column].unique()) > 1:
                    for name, _group in key_subject_mean.groupby([column]):
                        mask = data_subset[column] == name[0]
                        group = data_subset.loc[mask, ["subject", "key", "value", "error"]]
                        # group = data_subset[data[column] == name][["subject", "key", "value", "error"]]
                        group_total_count = len(group)
                        group_error_free = group[group["error"].isnull()][["subject", "key", "value"]]
                        group_error_free_ratio = float(len(group_error_free) / group_total_count)
                        aggregated_results[f"ErrorFreeRatio {metric} - {name[0]}"] = group_error_free_ratio

                        group_key_subject_mean = group_error_free.groupby(["key", "subject"]).mean()
                        value = float(group_key_subject_mean[["value"]].mean()["value"])
                        aggregated_results[f"Average {metric} - {name[0]}"] = value if not math.isnan(value) else None

                        if group_error_free_ratio < 1.0:
                            # Treat error samples (with value=None) as 0 for the "including errors" average
                            group_with_errors = group[["key", "subject", "value", "error"]].copy()
                            # Only fill value with 0 where there's an error (not for all None values)
                            error_mask = group_with_errors["error"].notna()
                            group_with_errors.loc[error_mask, "value"] = group_with_errors.loc[
                                error_mask, "value"
                            ].fillna(0.0)
                            group_key_subject_mean_with_errors = group_with_errors.groupby(["key", "subject"])[
                                "value"
                            ].mean()
                            value_with_errors = float(group_key_subject_mean_with_errors.mean())
                            aggregated_results[f"Average {metric} (including Errors) - {name[0]}"] = (
                                value_with_errors if not math.isnan(value_with_errors) else None
                            )

                        if not ("SequencePositions" in metric or "Bytes" in metric):
                            # calculate standard error for selected  metrics
                            group_key_subject_std = group_error_free.groupby(["key", "subject"]).std()
                            std = float(group_key_subject_std[["value"]].mean()["value"])
                            num_samples = len(group_error_free)

                            if math.isnan(std) or num_samples == 0:
                                aggregated_results[f"StdErr {metric} - {name[0]}"] = None
                            else:
                                aggregated_results[f"StdErr {metric} - {name[0]}"] = std / np.sqrt(num_samples)
                            aggregated_results[f"NumSamples {metric} - {name[0]}"] = num_samples

                            std_err_mean_sum_of_squares += std**2 / num_samples
                            std_err_mean_total_num_samples += num_samples
                            std_err_mean_num_subjects += 1

            if not ("SequencePositions" in metric or "Bytes" in metric):
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
                    key_subject_std = data_subset_error_free.groupby(["key", "subject"]).std()
                    std = float(key_subject_std[["value"]].mean()["value"])
                    num_samples = len(data_subset_error_free)
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
        data = pd.DataFrame([r.model_dump() for r in results])
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
            current_metric = None
            # now loop over the self.metrics list and find the metric class that matches the current_metric_class
            for metric_class in self.metrics:
                if metric_class.__name__ == current_metric_class:
                    current_metric = metric_class
                    break
            if current_metric is None:
                raise ValueError(f"Metric {metric_name} not found in metrics list")

            for aggregator in current_metric.AGGREGATORS:
                aggregated_results[f"{aggregator.name} {current_metric_class}.{metric_name}"] = (
                    aggregator(metric_group, ["prompt"])  # Compute the aggregator, grouped by the prompt...
                    .groupby(["key", "subject"])  # ... then group by key, subject...
                    .agg({"value": "mean"})["value"]  # ...and average scores over each key, subject group...
                    .mean()  # ...and lastly average the scores across all groups giving equal weight to every
                    .item()  # key, subject group.
                )

        # Loop to additionally compute per-subject/per-key breakdown metric scores, e.g. for only subject="algebra"
        for (key, subject, metric_name), ksm_group in data.groupby(["key", "subject", "metric_name"]):
            current_metric_class = ksm_group["metric_class_name"].unique().item()
            current_metric = None
            # now loop over the self.metrics list and find the metric class that matches the current_metric_class
            for metric_class in self.metrics:
                if metric_class.__name__ == current_metric_class:
                    current_metric = metric_class
                    break

            if current_metric is None:
                raise ValueError(f"Metric {metric_name} not found in metrics list. This should never happen.")

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
        aggregated_results = self._aggregate_results(metrics_results)
        results_with_aggregators = self._aggregate_results_with_aggregators(metrics_results)
        aggregated_results.update(results_with_aggregators)

        wandb.log(aggregated_results)
        self.result_processor.save_aggregated_results(aggregated_results)
        logger.info(aggregated_results)
        logger.info(f"{RED}[ Evaluation completed and results saved! ]{RESET}")
        return metrics_results
