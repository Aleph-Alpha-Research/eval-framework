"""The constituent BPB metrics reproduce ``BitsPerByteVariantsLoglikelihood`` exactly.

Listing ``BitsPerByteLoglikelihood`` + ``CorpusBitsPerByteLoglikelihood`` +
``PrefixBitsPerByteLoglikelihood`` must emit the same metric names and values (and the same
error/non-error status) as the old bundled variants class, for every response shape.
"""

from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.loglikelihood.bpb_constituents import (
    CorpusBitsPerByteLoglikelihood,
    PrefixBitsPerByteLoglikelihood,
    PriorBitsPerByteLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bpb_variants import BitsPerByteVariantsLoglikelihood
from eval_framework.shared.types import Loglikelihood, TokenLogprob


def _response(
    ground_truth: str,
    *,
    tokens: list[tuple[str, float]] | None = None,
    loglikelihood: float | None = None,
    extra_loglikelihoods: dict[str, float] | None = None,
) -> Loglikelihood:
    token_logprobs: dict[str, list[TokenLogprob]] = {}
    if tokens is not None:
        token_logprobs = {ground_truth: [TokenLogprob(logprob=lp, decoded_token=d) for d, lp in tokens]}
        loglikelihood = sum(lp for _, lp in tokens)
        num_tokens = len(tokens)
    else:
        assert loglikelihood is not None
        num_tokens = 1
    loglikelihoods = {ground_truth: loglikelihood}
    loglikelihoods.update(extra_loglikelihoods or {})
    return Loglikelihood(
        sample_id=1,
        result_id=1,
        subject="s",
        ground_truth=ground_truth,
        prompt="p",
        prompt_num_tokens=None,
        loglikelihoods=loglikelihoods,
        loglikelihoods_num_tokens={ground_truth: num_tokens},
        token_logprobs=token_logprobs,
    )


def _as_map(results: list[MetricResult]) -> dict[str, tuple[float | None, bool]]:
    out: dict[str, tuple[float | None, bool]] = {}
    for r in results:
        assert r.metric_name not in out, f"duplicate metric name {r.metric_name!r}"
        out[r.metric_name] = (r.value, r.error is not None)
    return out


def _constituent_results(response: Loglikelihood) -> list[MetricResult]:
    return [
        *BitsPerByteLoglikelihood().calculate(response),
        *CorpusBitsPerByteLoglikelihood().calculate(response),
        *PrefixBitsPerByteLoglikelihood().calculate(response),
    ]


def _assert_equivalent(response: Loglikelihood) -> dict[str, tuple[float | None, bool]]:
    variants = _as_map(BitsPerByteVariantsLoglikelihood().calculate(response))
    constituents = _as_map(_constituent_results(response))
    assert constituents == variants
    return constituents


def test_equivalent_with_per_token_data() -> None:
    names = _assert_equivalent(_response(" hello", tokens=[(" hel", -2.0), ("lo", -3.0)]))
    assert "PrefixBPB" in names
    assert "BitsPerByte" in names
    assert "BitsPerByte_bits" in names


def test_equivalent_without_per_token_data() -> None:
    names = _assert_equivalent(_response(" Paris", loglikelihood=-2.5))
    assert not any(n.startswith("PrefixBPB") for n in names)
    assert set(names) == {
        "BitsPerByte",
        "BitsPerByte_bits",
        "BitsPerByte_bytes",
        "BitsPerByte_tokens",
        "BitsPerByte_nAliases",
    }


def test_equivalent_when_no_ground_truth_found() -> None:
    response = _response(" Paris", loglikelihood=-2.5)
    response.loglikelihoods = {" London": -1.0}  # ground truth not among the scored continuations
    names = _assert_equivalent(response)
    assert all(is_error for _, is_error in names.values())


def test_equivalent_on_zero_byte_ground_truth() -> None:
    names = _assert_equivalent(_response("", loglikelihood=-1.0))
    assert all(is_error for _, is_error in names.values())


def test_prior_metric_emits_nothing_per_response() -> None:
    response = _response(" hello", tokens=[(" hel", -2.0), ("lo", -3.0)])
    assert PriorBitsPerByteLoglikelihood().calculate(response) == []
