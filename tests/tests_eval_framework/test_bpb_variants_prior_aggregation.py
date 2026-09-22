import math

import eval_framework.metrics.loglikelihood.bpb_variants_common as bpb_common
from eval_framework.metrics.loglikelihood.bpb_variants_common import (
    MU_PRIOR,
    aggregate_prior_bpb_metrics,
    collect_prefix_items,
)
from eval_framework.shared.types import Loglikelihood, PerTokenScores


def _response(ground_truth: str, bits: list[float], byte_lens: list[int]) -> Loglikelihood:
    return Loglikelihood(
        id=1,
        subject="subj",
        ground_truth=ground_truth,
        prompt="p",
        prompt_num_tokens=None,
        loglikelihoods={ground_truth: -math.log(2) * sum(bits)},
        loglikelihoods_num_tokens={ground_truth: len(bits)},
        loglikelihoods_per_token={ground_truth: PerTokenScores(bits=bits, byte_lens=byte_lens)},
    )


def test_mu_prior_default_is_111():
    assert MU_PRIOR == 111.0


def test_collect_prefix_items_skips_missing_per_token():
    with_token = _response("abc", [1.0, 1.0, 1.0], [1, 1, 1])
    without_token = Loglikelihood(
        id=2,
        subject="subj",
        ground_truth="x",
        prompt="p",
        prompt_num_tokens=None,
        loglikelihoods={"x": -0.5},
        loglikelihoods_num_tokens={"x": 1},
    )
    items = collect_prefix_items([with_token, without_token])
    assert len(items) == 1


def test_aggregate_prior_bpb_flat_cost():
    # 1 bit/byte => Prior BPB = 1.0
    resp = _response("abcdef", [1.0] * 6, [1] * 6)
    aggregated = aggregate_prior_bpb_metrics([resp])
    assert aggregated["Prior BPB"] == 1.0
    assert aggregated["Prior BPB - subj"] == 1.0
    assert aggregated["Prior BPB prior_mass_in_support"] == 1.0
    assert aggregated["Prior BPB prior_mass_in_support - subj"] == 1.0


def test_aggregate_prior_bpb_respects_mu_override(monkeypatch):
    monkeypatch.setattr(bpb_common, "MU_PRIOR", 60.0)
    resp = _response("abcdef", [1.0] * 6, [1] * 6)
    aggregated = aggregate_prior_bpb_metrics([resp])
    assert aggregated["Prior BPB"] == 1.0
