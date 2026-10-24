import logging
from collections.abc import Callable, Iterator
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


STDERR_EXCLUDED = ("SequencePositions", "Bytes")


def _results_frame(results: list[Result], **extra: Callable[[Result], Any]) -> pd.DataFrame:
    rows = [
        {
            "metric_name": r.metric_name,
            "metric_class_name": r.metric_class_name,
            "subject": r.subject,
            "key": r.key or "",
            "value": r.value,
            "error": r.error,
            **{name: fn(r) for name, fn in extra.items()},
        }
        for r in results
    ]
    return pd.DataFrame(rows)


def _scope_statistics(frame: pd.DataFrame, metric: str, scope: str = "", micro: bool = False) -> dict[str, float]:
    """Macro statistics over (key, subject) cells for one metric within one scope."""
    error_free = frame.loc[frame["error"].isnull()]
    cells = error_free.groupby(["key", "subject"])["value"].agg(["mean", "std"])
    out = {
        f"ErrorFreeRatio {metric}{scope}": frame["error"].isnull().mean(),
        f"Average {metric}{scope}": cells["mean"].mean(),
        "_std": cells["std"].mean(),
        "_n": len(error_free),
    }
    if micro:
        out[f"Average {metric} (micro)"] = error_free["value"].mean()
    if frame["error"].notna().any():
        value = frame["value"].mask(frame["error"].notna() & frame["value"].isna(), 0.0)
        out[f"Average {metric} (including Errors){scope}"] = (
            value.groupby([frame["key"], frame["subject"]]).mean().mean()
        )
    return out


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

    def _compute_sample_results(
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

    def _compute_metrics(self, responses: list[Completion | Loglikelihood]) -> list[Result]:
        results: list[Result] = self.result_processor.load_metrics_results()
        llm_name = self.result_processor.load_metadata()["llm_name"]

        subject_result_id_existing = set()
        for result in results:
            subject_result_id_existing.add(f"{result.subject}_{result.id}_{result.metric_class_name}")

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
            compute = partial(self._compute_sample_results, metric, llm_name=llm_name)
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

    @staticmethod
    def _summarize_samples(results: list[Result]) -> dict[str, float | None]:
        data = _results_frame(results)
        if data.empty:
            return {}
        out: dict[str, float] = {}
        for metric, frame in data.groupby("metric_name"):
            with_stderr = not any(s in metric for s in STDERR_EXCLUDED)
            overall = _scope_statistics(frame, metric, micro=True)
            slices = []
            for column in ("key", "subject"):
                if frame[column].nunique() > 1:
                    for name, group in frame.groupby(column):
                        stats = _scope_statistics(group, metric, f" - {name}")
                        slices.append(stats)
                        if with_stderr:
                            stats[f"StdErr {metric} - {name}"] = stats["_std"] / np.sqrt(stats["_n"])
                            stats[f"NumSamples {metric} - {name}"] = stats["_n"]
                        out.update(stats)
            if with_stderr:
                if slices:
                    out[f"StdErr {metric}"] = np.sqrt(np.mean([s["_std"] ** 2 / s["_n"] for s in slices]))
                    out[f"NumSamples {metric}"] = overall["_n"]
                else:
                    out[f"StdErr {metric}"] = overall["_std"] / np.sqrt(overall["_n"])
                    out[f"NumSamples {metric}"] = overall["_n"]
            out.update(overall)
        summary: dict[str, float | None] = {k: float(v) for k, v in out.items() if not k.startswith("_")}
        nbytes, npos = summary.get("Average Bytes"), summary.get("Average SequencePositions")
        if nbytes and npos:
            summary["Average Bytes per Sequence Position"] = nbytes / npos
        return summary

    def _summarize_problems(self, results: list[Result]) -> dict[str, float | None]:
        # Repeats of one problem carry consecutive ids (see `repeat_samples`).
        data = _results_frame(results, problem=lambda r: r.id // self.config.repeats)
        if data.empty:
            return {}
        data = data.loc[data["error"].isnull()]
        metric_by_class = {cls.__name__: cls for cls in self.metrics}
        out: dict[str, float | None] = {}
        for (metric, cls_name), frame in data.groupby(["metric_name", "metric_class_name"]):
            for aggregator in metric_by_class[cls_name].AGGREGATORS:
                per_problem = aggregator(frame, ["key", "subject", "problem"])
                by_cell = per_problem.groupby(["key", "subject"])["value"].mean()
                out[f"{aggregator.name} {cls_name}.{metric}"] = float(by_cell.mean())
                out[f"{aggregator.name} {cls_name}.{metric} (micro)"] = float(per_problem["value"].mean())
                for (key, subject), value in by_cell.items():
                    label = f"{key} - {subject}" if key else subject
                    out[f"{aggregator.name} {metric} - {label}"] = float(value)
        return out

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

        metrics_results = self._compute_metrics(responses)
        loglikelihood_responses = [r for r in responses if isinstance(r, Loglikelihood)]
        # Leading-space flag per item, used by the space-stripped corpus BPB.
        leading_space_by_item: dict[tuple[int, str], float] = {}
        for response in loglikelihood_responses:
            ground_truth = select_ground_truth(response)
            if ground_truth is not None:
                leading_space_by_item[(response.id, response.subject)] = 1.0 if ground_truth.startswith(" ") else 0.0
        del responses
        aggregated_results = self._summarize_samples(metrics_results)
        results_with_aggregators = self._summarize_problems(metrics_results)
        aggregated_results.update(results_with_aggregators)
        aggregated_results.update(self._aggregate_corpus_bpb_metrics(metrics_results, leading_space_by_item))
        aggregated_results.update(aggregate_prior_bpb_metrics(loglikelihood_responses))

        wandb.log(aggregated_results)
        self.result_processor.save_aggregated_results(aggregated_results)
        logger.info(aggregated_results)
        logger.info(f"{RED}[ Evaluation completed and results saved! ]{RESET}")
        return metrics_results
