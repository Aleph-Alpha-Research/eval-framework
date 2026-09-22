import logging
import math
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, cast

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from eval_framework.metrics.base import BaseMetric
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric
from eval_framework.metrics.loglikelihood.bpb_variants_common import aggregate_prior_bpb_metrics, select_ground_truth
from eval_framework.metrics.loglikelihood.bpb_variants_estimators import summarize_all
from eval_framework.result_processors.base import Result, ResultProcessor
from eval_framework.shared.types import Completion, Loglikelihood
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.registry import registry
from eval_framework.utils.constants import RED, RESET
from eval_framework.utils.tqdm_handler import get_disable_bar_flag

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

        eval_ = registry()[config.task_name]
        self.metrics = eval_.metrics()
        self.task_name = eval_.display_name()

    def _results_for(
        self,
        metric: BaseMetric[Completion | Loglikelihood],
        response: Completion | Loglikelihood,
        llm_name: str,
    ) -> list[Result]:
        """Compute one response's results. Runs in a worker thread, so must not mutate shared state."""
        results: list[Result] = []
        for metric_result in metric.calculate(response):
            if "/" in metric_result.metric_name:
                metric_name, key = metric_result.metric_name.split("/")
            else:
                metric_name = metric_result.metric_name
                key = None

            results.append(
                Result(
                    id=response.id,
                    metric_class_name=metric.__class__.__name__,
                    metric_name=metric_name,
                    num_fewshot=self.few_shot,
                    key=key,
                    subject=response.subject,
                    llm_name=llm_name,
                    task_name=self.task_name,
                    value=metric_result.value,
                    higher_is_better=metric_result.higher_is_better,
                    llm_judge_prompt=metric_result.llm_judge_prompt,
                    llm_judge_response=metric_result.llm_judge_response,
                    code_execution_trace=metric_result.code_execution_trace,
                    error=metric_result.error,
                )
            )
        return results

    def _collect(self, batch: list[Result], results: list[Result]) -> None:
        """Accumulate and persist one response's results. Main thread only, so no locking needed."""
        for result in batch:
            results.append(result)
            if self.save_intermediate_results:
                self.result_processor.save_metrics_result(result)

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
            raw_metric: BaseMetric[Any]
            if issubclass(metric_class, BaseLLMJudgeMetric):
                if llm_judge is None:
                    llm_judge = self.config.llm_judge()
                raw_metric = metric_class(
                    llm_judge=llm_judge,
                    randomize_order=self.config.randomize_judge_order,
                )
            else:
                raw_metric = metric_class()
            metric = cast(BaseMetric[Completion | Loglikelihood], raw_metric)
            metric.fail_on_error = self.config.fail_on_error

            metric.prepare(responses)
            logger.info(f"Starting calculation of {metric.NAME}")

            pending = [
                response
                for response in responses
                if f"{response.subject}_{response.id}_{metric.__class__.__name__}" not in subject_result_id_existing
            ]
            compute = partial(self._results_for, metric, llm_name=llm_name)
            desc = f"Calculating {metric.NAME}"

            if metric.MAX_WORKERS > 1:
                with ThreadPoolExecutor(max_workers=metric.MAX_WORKERS) as executor:
                    # map yields in submission order, so results match a serial run.
                    batches: Iterator[list[Result]] = executor.map(compute, pending)
                    for batch in tqdm(batches, total=len(pending), desc=desc, disable=get_disable_bar_flag()):
                        self._collect(batch, results)
            else:
                for response in tqdm(pending, desc=desc, disable=get_disable_bar_flag()):
                    self._collect(compute(response), results)

            logger.info(f"Completed calculation of {metric.NAME}")

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
            total_count = len(data_subset)

            mask = data_subset["error"].isnull()
            data_subset_error_free = data_subset.loc[mask, ["subject", "key", "value"]]

            error_free_ratio = float(len(data_subset_error_free) / total_count)
            aggregated_results[f"ErrorFreeRatio {metric}"] = error_free_ratio

            # Default average is weighted by key and subject (macro average).
            # aggregate by key and subject first to have equal weights for all key / subject combinations
            key_subject_mean = data_subset_error_free.groupby(["key", "subject"]).mean()
            aggregated_results[f"Average {metric}"] = float(key_subject_mean[["value"]].mean()["value"])
            # Additionally, report the plain mean over all samples (micro average or per-sample average).
            aggregated_results[f"Average {metric} (micro)"] = float(data_subset_error_free["value"].mean())

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
                        aggregated_results[f"Average {metric} - {name[0]}"] = value

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
                            aggregated_results[f"Average {metric} (including Errors) - {name[0]}"] = value_with_errors

                        if not ("SequencePositions" in metric or "Bytes" in metric):
                            # calculate standard error for selected  metrics
                            group_key_subject_std = group_error_free.groupby(["key", "subject"]).std()
                            std = float(group_key_subject_std[["value"]].mean()["value"])
                            num_samples = len(group_error_free)

                            if math.isnan(std) or num_samples == 0:
                                aggregated_results[f"StdErr {metric} - {name[0]}"] = float("nan")
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
                        aggregated_results[f"StdErr {metric}"] = float("nan")
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
                        aggregated_results[f"StdErr {metric}"] = float("nan")
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
                    # Repeats of one problem carry consecutive ids (see `repeat_samples`).
                    "problem": r.id // self.config.repeats,
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
            current_metric = None
            # now loop over the self.metrics list and find the metric class that matches the current_metric_class
            for metric_class in self.metrics:
                if metric_class.__name__ == current_metric_class:
                    current_metric = metric_class
                    break
            if current_metric is None:
                raise ValueError(f"Metric {metric_name} not found in metrics list")

            for aggregator in current_metric.AGGREGATORS:
                # Compute the aggregator per problem (collapsing the repeats of each problem into one score).
                per_problem = aggregator(metric_group, ["subject", "problem"])
                # Macro average: mean per key/subject group, then mean over groups, giving equal weight to every group.
                aggregated_results[f"{aggregator.name} {current_metric_class}.{metric_name}"] = (
                    per_problem.groupby(["key", "subject"]).agg({"value": "mean"})["value"].mean().item()
                )
                # Micro average: plain mean over all problems.
                aggregated_results[f"{aggregator.name} {current_metric_class}.{metric_name} (micro)"] = (
                    per_problem["value"].mean().item()
                )

        # Loop to additionally compute per-subject/per-key breakdown metric scores, e.g. for only subject="algebra"
        for (key, subject, metric_name, current_metric_class), ksm_group in data.groupby(
            ["key", "subject", "metric_name", "metric_class_name"]
        ):
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
                aggregated_results[save_string] = (
                    aggregator(ksm_group, ["subject", "problem"])["value"].mean().mean().item()
                )

        return aggregated_results

    @staticmethod
    def _flatten_corpus_summary(summary: dict, subject: str | None = None) -> dict[str, float | None]:
        """Flatten ``summarize_all()`` output into aggregated-result keys."""
        scope = "" if subject is None else f" - {subject}"
        out: dict[str, float | None] = {}
        scalar_map = {
            "corpus_bpb": f"Corpus BPB{scope}",
            "bits_per_answer": f"Corpus BitsPerAnswer{scope}",
            "mean_bytes": f"Corpus mean_bytes{scope}",
            "median_bytes": f"Corpus median_bytes{scope}",
            "token_corpus_bpb": f"Corpus token BPB{scope}",
            "space_stripped_corpus_bpb": f"Corpus BPB space_stripped{scope}",
            "effective_length": f"Corpus effective_length BPB{scope}",
        }
        for summary_key, label in scalar_map.items():
            if summary_key in summary and isinstance(summary[summary_key], (int, float)):
                out[label] = float(summary[summary_key])

        for ls_key in ("ls_bpb_task_q", "ls_bpb_common_q"):
            if ls_key in summary and isinstance(summary[ls_key], (int, float)):
                out[f"Corpus {ls_key}{scope}"] = float(summary[ls_key])

        for fit_key in ("ols", "huber", "ols_tokens"):
            fit = summary.get(fit_key)
            if isinstance(fit, dict):
                for sub_key, sub_val in fit.items():
                    if isinstance(sub_val, (int, float)) and sub_val == sub_val:
                        out[f"Corpus {fit_key}_{sub_key}{scope}"] = float(sub_val)

        for method in ("bpb_at_nstar_ols", "bpb_at_nstar_huber"):
            nested = summary.get(method)
            if isinstance(nested, dict):
                for nstar, val in nested.items():
                    if isinstance(val, (int, float)) and val == val:
                        out[f"Corpus {method} {nstar}{scope}"] = float(val)

        return out

    @staticmethod
    def _aggregate_corpus_bpb_metrics(
        results: list[Result], leading_space_by_item: dict[tuple[int, str], float] | None = None
    ) -> dict[str, float | None]:
        """Corpus BPB and related estimators from the ``BitsPerByte_*`` companion fields.

        ``leading_space_by_item`` maps (id, subject) to 1.0 when the ground truth begins with a
        space (else 0.0), feeding the space-stripped corpus BPB. Result rows no longer carry the
        response text, so leading-space comes in from the loglikelihood responses.
        """
        leading_space_by_item = leading_space_by_item or {}
        sidecar_names = {"BitsPerByte_bits", "BitsPerByte_bytes", "BitsPerByte_tokens"}
        rows: list[dict] = []
        for result in results:
            if result.error is not None or result.value is None:
                continue
            if result.metric_name not in sidecar_names:
                continue
            rows.append(
                {
                    "id": result.id,
                    "subject": result.subject,
                    "key": result.key or "",
                    "metric_name": result.metric_name,
                    "value": result.value,
                }
            )

        if not rows:
            return {}

        data = pd.DataFrame(rows)
        pivot = data.pivot_table(
            index=["id", "subject", "key"],
            columns="metric_name",
            values="value",
            aggfunc="first",
        )
        required = ["BitsPerByte_bits", "BitsPerByte_bytes"]
        if not all(col in pivot.columns for col in required):
            return {}

        pivot = pivot.dropna(subset=required)
        if len(pivot) == 0:
            return {}

        aggregated: dict[str, float | None] = {}

        def summarize_slice(frame: pd.DataFrame, subject: str | None = None) -> None:
            bits = frame["BitsPerByte_bits"].to_numpy(dtype=float)
            nbytes = frame["BitsPerByte_bytes"].to_numpy(dtype=float)
            tokens = None
            if "BitsPerByte_tokens" in frame.columns:
                tok = frame["BitsPerByte_tokens"].to_numpy(dtype=float)
                if np.all(np.isfinite(tok)):
                    tokens = tok
            leading_space = None
            if leading_space_by_item:
                ls_vals = []
                for idx in frame.index:
                    if subject is None:
                        item_id, item_subject = idx[0], idx[1]
                    else:
                        item_id, item_subject = idx[0], subject
                    ls_vals.append(leading_space_by_item.get((item_id, item_subject), 0.0))
                leading_space = np.asarray(ls_vals, dtype=float)
            summary = summarize_all(bits, nbytes, tokens=tokens, leading_space=leading_space)
            aggregated.update(EvaluationGenerator._flatten_corpus_summary(summary, subject=subject))

        summarize_slice(pivot)

        for subject in sorted(pivot.index.get_level_values("subject").unique()):
            subject_frame = pivot.xs(subject, level="subject")
            if len(subject_frame) == 0:
                continue
            summarize_slice(subject_frame, subject=subject)

        return aggregated

    def run_eval(self) -> list[Result]:
        """Runs evaluation using saved completions."""
        logger.info("Running evaluation...")
        responses = self.result_processor.load_responses()
        if not responses:
            raise ValueError("No saved completions found. Run 'run_completions' first.")

        metrics_results = self._run_metric_calculators(responses)
        loglikelihood_responses = [r for r in responses if isinstance(r, Loglikelihood)]
        # Leading-space flag per item, used by the space-stripped corpus BPB.
        leading_space_by_item: dict[tuple[int, str], float] = {}
        for response in loglikelihood_responses:
            ground_truth = select_ground_truth(response)
            if ground_truth is not None:
                leading_space_by_item[(response.id, response.subject)] = 1.0 if ground_truth.startswith(" ") else 0.0
        del responses
        aggregated_results = self._aggregate_results(metrics_results)
        results_with_aggregators = self._aggregate_results_with_aggregators(metrics_results)
        aggregated_results.update(results_with_aggregators)
        aggregated_results.update(self._aggregate_corpus_bpb_metrics(metrics_results, leading_space_by_item))
        aggregated_results.update(aggregate_prior_bpb_metrics(loglikelihood_responses))

        wandb.log(aggregated_results)
        self.result_processor.save_aggregated_results(aggregated_results)
        logger.info(aggregated_results)
        logger.info(f"{RED}[ Evaluation completed and results saved! ]{RESET}")
        return metrics_results
