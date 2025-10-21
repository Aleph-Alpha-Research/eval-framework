import logging
import math

import numpy as np
import pandas as pd
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
from eval_framework.result_processors.result_processor import ResultsWandbProcessor
from eval_framework.shared.types import Completion, Loglikelihood
from eval_framework.tasks.base import ResponseType
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.registry import get_task
from eval_framework.utils.constants import RED, RESET

logger = logging.getLogger(__name__)


class EvaluationGenerator:
    def __init__(
        self,
        config: EvalConfig,
        result_processor: ResultProcessor,
        wandb_processor: ResultsWandbProcessor | None = None,
    ) -> None:
        logger.info("EvaluationGenerator initialized")

        self.config = config
        self.few_shot = config.num_fewshot
        self.num_samples = config.num_samples
        self.max_tokens = config.max_tokens
        self.save_intermediate_results = config.save_intermediate_results
        self.result_processor = result_processor
        self.wandb_processor = wandb_processor

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
                metric = metric_class(llm_judge=llm_judge)
            else:
                metric = metric_class()

            logger.info(f"Starting calculation of {metric.NAME}")
            tqdm.write(f"INFO: Calculating {metric.NAME}")
            for response in tqdm(responses, desc=f"Calculating {metric.NAME}"):
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
            tqdm.write(f"INFO: Completed {metric.NAME}")

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

            mask = data["error"].isnull()
            data_subset_error_free = data_subset.loc[mask, ["subject", "key", "value"]]
            # data_subset_error_free = data_subset[data_subset["error"].isnull()][["subject", "key", "value"]]

            aggregated_results[f"ErrorFreeRatio {metric}"] = float(len(data_subset_error_free) / total_count)

            # aggregate by key and subject first to have equal weights for all key / subject combinations
            key_subject_mean = data_subset_error_free.groupby(["key", "subject"]).mean()
            aggregated_results[f"Average {metric}"] = float(key_subject_mean[["value"]].mean()["value"])

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
                        aggregated_results[f"ErrorFreeRatio {metric} - {name[0]}"] = float(
                            len(group_error_free) / group_total_count
                        )

                        group_key_subject_mean = group_error_free.groupby(["key", "subject"]).mean()
                        value = float(group_key_subject_mean[["value"]].mean()["value"])
                        aggregated_results[f"Average {metric} - {name[0]}"] = value if not math.isnan(value) else None

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

    def run_eval(self) -> list[Result]:
        """Runs evaluation using saved completions."""
        logger.info("Running evaluation...")
        responses = self.result_processor.load_responses()
        if not responses:
            raise ValueError("No saved completions found. Run 'run_completions' first.")

        metrics_results = self._run_metric_calculators(responses)
        aggregated_results = self._aggregate_results(metrics_results)

        self.result_processor.save_aggregated_results(aggregated_results)
        if self.wandb_processor is not None:
            self.wandb_processor.save_aggregated_results(aggregated_results)
        logger.info(aggregated_results)
        logger.info(f"{RED}[ Evaluation completed and results saved! ]{RESET}")
        return metrics_results
