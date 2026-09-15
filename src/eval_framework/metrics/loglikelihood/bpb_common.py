"""Shared logic for ``BitsPerByteLoglikelihood``."""

from __future__ import annotations

import math
import os
from typing import Literal, get_args

import numpy as np

from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.loglikelihood.bpb_estimators import PrefixItem, cumulative_cost, prior_bpb
from eval_framework.shared.types import Error, Loglikelihood

AliasRule = Literal["first", "shortest", "best"]

# First matching alias in ground_truth order (main-branch default).
_DEFAULT_ALIAS_RULE: AliasRule = "first"

STANDARD_METRIC_NAMES = (
    "BitsPerByte",
    "BitsPerByte_bits",
    "BitsPerByte_bytes",
    "BitsPerByte_tokens",
    "BitsPerByte_nAliases",
)

PREFIX_METRIC_NAMES = (
    "PrefixBPB",
    "PrefixBPB_rate",
    "PrefixBPB_entryCost",
    "PrefixBPB_contentBytes",
)


def _resolve_alias_rule() -> AliasRule:
    raw = os.environ.get("BPB_ALIAS_RULE")
    if raw is None:
        return _DEFAULT_ALIAS_RULE
    rule = raw.strip().lower()
    if rule not in get_args(AliasRule):
        raise ValueError(f"Invalid BPB_ALIAS_RULE={raw!r}; expected one of {get_args(AliasRule)}")
    return rule  # type: ignore[return-value]


ALIAS_RULE: AliasRule = _resolve_alias_rule()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} must be an int, got {raw!r}") from e


# Prefix byte budgets (override with PREFIX_K_* env vars).
K_ENTRY = _int_env("PREFIX_K_ENTRY", 1)
K_RATE = _int_env("PREFIX_K_RATE", 32)
K0 = _int_env("PREFIX_K0", 2)

# Geometric prior mean for Prior BPB (default 111; override with PRIOR_BPB_MU).
MU_PRIOR = float(_int_env("PRIOR_BPB_MU", 111))


def list_gold_candidates(response: Loglikelihood) -> list[str]:
    return [gt for gt in (response.ground_truth_list or []) if gt is not None and gt in response.loglikelihoods]


def select_gold(response: Loglikelihood, candidates: list[str] | None = None) -> str | None:
    if candidates is None:
        candidates = list_gold_candidates(response)
    if not candidates:
        return None
    if ALIAS_RULE == "first":
        return candidates[0]
    if ALIAS_RULE == "shortest":
        return min(candidates, key=lambda gt: len(gt.encode("utf-8")))
    return max(candidates, key=lambda gt: float(response.loglikelihoods[gt]))


def standard_bpb_error(message: str, response: Loglikelihood) -> list[MetricResult]:
    err = response.error or Error(error_class="ValueError", message=message, traceback="")
    return [
        MetricResult(metric_name=name, value=None, higher_is_better=False, error=err)
        for name in STANDARD_METRIC_NAMES
    ]


def compute_standard_bpb_results(
    response: Loglikelihood,
    answer_text: str,
    candidates: list[str],
) -> list[MetricResult]:
    log_p_x = float(response.loglikelihoods[answer_text])
    num_bytes = len(answer_text.encode("utf-8"))
    if num_bytes == 0:
        return standard_bpb_error("Ground-truth answer has zero UTF-8 bytes", response)

    seq_positions = getattr(response, "loglikelihoods_sequence_positions", None) or {}
    num_tokens = seq_positions.get(answer_text)
    bits = -log_p_x / math.log(2)

    def ok(name: str, value: float | None) -> MetricResult:
        return MetricResult(metric_name=name, value=value, higher_is_better=False, error=response.error)

    return [
        ok("BitsPerByte", bits / num_bytes),
        ok("BitsPerByte_bits", bits),
        ok("BitsPerByte_bytes", float(num_bytes)),
        ok("BitsPerByte_tokens", float(num_tokens) if num_tokens is not None else None),
        ok("BitsPerByte_nAliases", float(len(candidates))),
    ]


def compute_prefix_bpb_results(response: Loglikelihood, gold: str) -> list[MetricResult]:
    per_token = (response.loglikelihoods_per_token or {}).get(gold)
    if per_token is None or not per_token.bits:
        return []

    bits_arr = np.asarray(per_token.bits, dtype=float)
    byte_lens_arr = np.asarray(per_token.byte_lens, dtype=float)
    offset = 1 if gold.startswith(" ") else 0
    content_bytes = int(round(float(byte_lens_arr.sum()))) - offset
    if content_bytes <= 0:
        err = response.error or Error(error_class="ValueError", message="Gold has zero content bytes", traceback="")
        return [MetricResult(metric_name=n, value=None, higher_is_better=False, error=err) for n in PREFIX_METRIC_NAMES]

    name_for = {K_ENTRY: "PrefixBPB", K_RATE: "PrefixBPB_rate", K0: "PrefixBPB_entryCost"}

    def cost(k: int) -> float:
        return cumulative_cost(bits_arr, byte_lens_arr, offset + k)

    def estimable(k: int, value: float | None) -> MetricResult:
        if content_bytes >= k and value is not None:
            return MetricResult(metric_name=name_for[k], value=value, higher_is_better=False, error=None)
        err = Error(
            error_class="NotEstimable",
            message=f"gold has {content_bytes} content bytes < K={k}",
            traceback="",
        )
        return MetricResult(metric_name=name_for[k], value=None, higher_is_better=False, error=err)

    return [
        estimable(K_ENTRY, cost(K_ENTRY) / K_ENTRY),
        estimable(K_RATE, cost(K_RATE) / K_RATE),
        estimable(K0, cost(K0)),
        MetricResult(
            metric_name="PrefixBPB_contentBytes",
            value=float(content_bytes),
            higher_is_better=False,
            error=None,
        ),
    ]


def build_prefix_item(response: Loglikelihood, gold: str) -> PrefixItem | None:
    per_token = (response.loglikelihoods_per_token or {}).get(gold)
    if per_token is None or not per_token.bits:
        return None
    offset = 1 if gold.startswith(" ") else 0
    item = PrefixItem(
        np.asarray(per_token.bits, dtype=float),
        np.asarray(per_token.byte_lens, dtype=float),
        offset=offset,
    )
    if item.content_bytes <= 0:
        return None
    return item


def collect_prefix_items(responses: list[Loglikelihood]) -> list[PrefixItem]:
    items: list[PrefixItem] = []
    for response in responses:
        if response.error is not None:
            continue
        gold = select_gold(response)
        if gold is None:
            continue
        item = build_prefix_item(response, gold)
        if item is not None:
            items.append(item)
    return items


def compute_prior_bpb_for_items(items: list[PrefixItem], mu: float | None = None) -> dict[str, float | None]:
    if mu is None:
        mu = MU_PRIOR
    if not items:
        return {"value": None, "prior_mass_in_support": None}
    result = prior_bpb(items, prior="geometric", mu=mu)
    value = result.get("value")
    return {
        "value": float(value) if isinstance(value, (int, float)) and value == value else None,
        "prior_mass_in_support": float(result["prior_mass_in_support"])
        if isinstance(result.get("prior_mass_in_support"), (int, float))
        else None,
    }


def aggregate_prior_bpb_metrics(responses: list[Loglikelihood]) -> dict[str, float | None]:
    """Prior BPB from per-token logprobs (geometric prior, ``MU_PRIOR`` default 111).

    Also emits ``Prior BPB prior_mass_in_support``: share of prior weight on byte
    positions k where at least one gold has length >= k (see ``prior_bpb`` in
    bpb_estimators.py). Near 1 on long-gold tasks; lower on short-gold tasks with
    large mu (e.g. letter labels).
    """
    aggregated: dict[str, float | None] = {}

    def add_slice(items: list[PrefixItem], subject: str | None = None) -> None:
        scope = "" if subject is None else f" - {subject}"
        summary = compute_prior_bpb_for_items(items)
        if summary["prior_mass_in_support"] is not None:
            aggregated[f"Prior BPB prior_mass_in_support{scope}"] = summary["prior_mass_in_support"]
        if summary["value"] is not None:
            aggregated[f"Prior BPB{scope}"] = summary["value"]

    add_slice(collect_prefix_items(responses))

    subjects = sorted({response.subject for response in responses})
    for subject in subjects:
        subject_responses = [response for response in responses if response.subject == subject]
        add_slice(collect_prefix_items(subject_responses), subject=subject)

    return aggregated


def compute_all_bpb_results(response: Loglikelihood) -> list[MetricResult]:
    if response.error:
        return standard_bpb_error("upstream error", response)

    candidates = list_gold_candidates(response)
    answer_text = select_gold(response, candidates)
    if answer_text is None:
        return standard_bpb_error("No ground-truth answer found in loglikelihoods", response)

    results = compute_standard_bpb_results(response, answer_text, candidates)
    results.extend(compute_prefix_bpb_results(response, answer_text))
    return results
