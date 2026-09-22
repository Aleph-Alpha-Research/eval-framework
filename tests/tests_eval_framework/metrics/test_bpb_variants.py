import math

import eval_framework.metrics.loglikelihood.bpb_variants_common as bpb_common
from eval_framework.metrics.loglikelihood.bpb_variants import BitsPerByteVariantsLoglikelihood
from eval_framework.shared.types import Loglikelihood, PerTokenScores


def _response(
    ground_truth: str,
    bits: list[float] | None = None,
    byte_lens: list[int] | None = None,
    *,
    log_p: float | None = None,
) -> Loglikelihood:
    if bits is not None:
        log_p = -math.log(2) * sum(bits)
    assert log_p is not None
    per_tok = {}
    if bits is not None and byte_lens is not None:
        per_tok = {ground_truth: PerTokenScores(bits=bits, byte_lens=byte_lens)}
    return Loglikelihood(
        id=1,
        subject="s",
        ground_truth=ground_truth,
        prompt="p",
        prompt_num_tokens=None,
        loglikelihoods={ground_truth: log_p},
        loglikelihoods_num_tokens={ground_truth: len(bits) if bits else 1},
        loglikelihoods_per_token=per_tok,
    )


def _by_name(results):
    return {r.metric_name: r for r in results}


def test_headline_matches_legacy_formula():
    ground_truth = " Paris"
    log_p = -2.5
    resp = Loglikelihood(
        id=1,
        subject="s",
        ground_truth=ground_truth,
        prompt="p",
        prompt_num_tokens=None,
        loglikelihoods={ground_truth: log_p},
        loglikelihoods_num_tokens={ground_truth: 1},
    )
    by = _by_name(BitsPerByteVariantsLoglikelihood().calculate(resp))
    num_bytes = len(ground_truth.encode("utf-8"))
    expected = -log_p / (num_bytes * math.log(2))
    assert by["BitsPerByte"].value == expected
    assert by["BitsPerByte_bits"].value == -log_p / math.log(2)
    assert by["BitsPerByte_bytes"].value == float(num_bytes)


def test_emits_prefix_when_per_token_present(monkeypatch):
    monkeypatch.setattr(bpb_common, "K_ENTRY", 1)
    monkeypatch.setattr(bpb_common, "K_RATE", 4)
    resp = _response("abcdef", [1.0] * 6, [1] * 6)
    names = {r.metric_name for r in BitsPerByteVariantsLoglikelihood().calculate(resp)}
    assert "PrefixBPB" in names
    assert "PrefixBPB_rate" in names


def test_omits_prefix_without_per_token():
    resp = Loglikelihood(
        id=1,
        subject="s",
        ground_truth="A",
        prompt="p",
        prompt_num_tokens=None,
        loglikelihoods={"A": -0.5},
        loglikelihoods_num_tokens={"A": 1},
    )
    names = {r.metric_name for r in BitsPerByteVariantsLoglikelihood().calculate(resp)}
    assert names == {
        "BitsPerByte",
        "BitsPerByte_bits",
        "BitsPerByte_bytes",
        "BitsPerByte_tokens",
        "BitsPerByte_nAliases",
    }


def test_first_alias_default(monkeypatch):
    monkeypatch.setattr(bpb_common, "ALIAS_RULE", "first")
    resp = Loglikelihood(
        id=1,
        subject="s",
        ground_truth=["first", "second"],
        prompt="p",
        prompt_num_tokens=None,
        loglikelihoods={"first": -1.0, "second": -0.1},
        loglikelihoods_num_tokens={"first": 1, "second": 1},
    )
    by = _by_name(BitsPerByteVariantsLoglikelihood().calculate(resp))
    assert by["BitsPerByte_bytes"].value == float(len(b"first"))
