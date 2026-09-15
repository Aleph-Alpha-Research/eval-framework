import math

import eval_framework.metrics.loglikelihood.bpb_common as bpb_common
from eval_framework.metrics.loglikelihood.prefix_bpb import PrefixBitsPerByte
from eval_framework.shared.types import Loglikelihood, PerTokenScores


def _resp(gold: str, bits: list[float], byte_lens: list[int], others: dict | None = None) -> Loglikelihood:
    ll = {gold: -math.log(2) * sum(bits)}
    per_tok = {gold: PerTokenScores(bits=bits, byte_lens=byte_lens)}
    for k, v in (others or {}).items():
        ll[k] = v
    return Loglikelihood(
        id=1,
        subject="s",
        ground_truth=gold,
        prompt="p",
        prompt_sequence_positions=None,
        loglikelihoods=ll,
        loglikelihoods_sequence_positions={gold: len(bits)},
        loglikelihoods_per_token=per_tok,
    )


def _by_name(results):
    return {r.metric_name for r in results}, {r.metric_name: r for r in results}


def test_emits_headline_and_sidecars():
    resp = _resp(" hello", [2.0, 3.0, 4.0], [2, 2, 2])  # offset=1, content=5 bytes
    results = PrefixBitsPerByte().calculate(resp)
    names, by = _by_name(results)
    assert names == {"PrefixBPB", "PrefixBPB_rate", "PrefixBPB_entryCost", "PrefixBPB_contentBytes"}
    assert by["PrefixBPB_contentBytes"].value == 5.0


def test_headline_equals_cost_over_k_entry(monkeypatch):
    # 1 bit/byte => PrefixBPB = 1.0 at K_ENTRY
    monkeypatch.setattr(bpb_common, "K_ENTRY", 1)
    monkeypatch.setattr(bpb_common, "K_RATE", 4)
    resp = _resp("abcdef", [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1, 1, 1, 1, 1, 1])  # no offset
    _, by = _by_name(PrefixBitsPerByte().calculate(resp))
    assert by["PrefixBPB"].value == 1.0
    assert by["PrefixBPB_rate"].value == 1.0
    assert by["PrefixBPB_entryCost"].value == 2.0  # cost of first K0=2 bytes


def test_not_estimable_when_gold_too_short(monkeypatch):
    monkeypatch.setattr(bpb_common, "K_RATE", 32)
    resp = _resp(" A", [5.0], [2])  # offset=1 -> 1 content byte
    _, by = _by_name(PrefixBitsPerByte().calculate(resp))
    # K_ENTRY=1 is estimable
    assert by["PrefixBPB"].value == 5.0
    # K=32 not estimable -> NotEstimable
    assert by["PrefixBPB_rate"].value is None
    assert by["PrefixBPB_rate"].error is not None
    assert by["PrefixBPB_rate"].error.error_class == "NotEstimable"


def test_errors_without_per_token():
    resp = Loglikelihood(
        id=1,
        subject="s",
        ground_truth="A",
        prompt="p",
        prompt_sequence_positions=None,
        loglikelihoods={"A": -0.5},
        loglikelihoods_sequence_positions={"A": 1},
    )
    results = PrefixBitsPerByte().calculate(resp)
    assert all(r.value is None and r.error is not None for r in results)


def test_gold_selected_by_best_rule(monkeypatch):
    import eval_framework.metrics.loglikelihood.bpb_common as bpb_common

    monkeypatch.setattr(bpb_common, "ALIAS_RULE", "best")
    # best alias has higher ll but no per-token scores
    resp = _resp("worse", [10.0, 10.0], [3, 2], others={"betteralias": -0.01})
    resp.ground_truth = ["worse", "betteralias"]
    results = PrefixBitsPerByte().calculate(resp)
    assert all(r.error is not None for r in results)
