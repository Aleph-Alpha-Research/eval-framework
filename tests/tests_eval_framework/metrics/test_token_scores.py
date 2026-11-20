import math

import pytest

from eval_framework.metrics.loglikelihood.token_scores import per_token_scores, per_token_scores_for
from eval_framework.shared.types import Loglikelihood, PerTokenScores, TokenLogprob


def _tokens(*tokens_and_logprobs: tuple[str, float]) -> list[TokenLogprob]:
    return [TokenLogprob(logprob=logprob, decoded_token=token) for token, logprob in tokens_and_logprobs]


@pytest.mark.parametrize(
    ("tokens", "continuation", "expected_byte_lens"),
    [
        pytest.param([(" A", -0.5)], " A", [2], id="token_ends_at_prompt_boundary"),
        pytest.param([(" A", -0.5)], "A", [1], id="token_straddles_prompt_boundary"),
        pytest.param([(" New", -1.5), (" York", -0.25)], " New York", [4, 5], id="multi_token_continuation"),
        pytest.param([(" é", -0.5)], "é", [2], id="multibyte_token_straddles_prompt_boundary"),
        pytest.param([], "", [], id="empty_continuation"),
    ],
)
def test_per_token_scores_count_only_bytes_inside_the_continuation(
    tokens: list[tuple[str, float]], continuation: str, expected_byte_lens: list[int]
) -> None:
    """Byte lengths sum to the continuation, even when the first token starts in the prompt.

    Continuation ``A`` is scored as the token `` A``. Prefix / Prior BPB take
    ``sum(byte_lens) - offset`` as the content bytes, so the prompt-side space must not
    be counted.
    """
    scores = per_token_scores(continuation, _tokens(*tokens))

    assert scores.byte_lens == expected_byte_lens
    assert sum(scores.byte_lens) == len(continuation.encode("utf-8"))


def test_per_token_scores_bits_are_surprisal_in_base_two() -> None:
    scores = per_token_scores(" New York", _tokens((" New", -1.5), (" York", -0.25)))

    assert scores.bits == [1.5 / math.log(2), 0.25 / math.log(2)]


def _response(**kwargs: object) -> Loglikelihood:
    base: dict[str, object] = dict(
        sample_id=1,
        result_id=1,
        subject="s",
        ground_truth=" A",
        prompt="p",
        prompt_num_tokens=None,
        loglikelihoods={" A": -0.5},
        loglikelihoods_num_tokens={" A": 1},
    )
    base.update(kwargs)
    return Loglikelihood(**base)  # type: ignore[arg-type]


def test_per_token_scores_for_prefers_raw_token_logprobs() -> None:
    response = _response(
        token_logprobs={" A": [TokenLogprob(logprob=-0.5, decoded_token=" A")]},
        loglikelihoods_per_token={" A": PerTokenScores(bits=[99.0], byte_lens=[99])},
    )

    scores = per_token_scores_for(response, " A")

    assert scores is not None
    assert scores.bits == [0.5 / math.log(2)]
    assert scores.byte_lens == [2]


def test_per_token_scores_for_falls_back_to_legacy_field() -> None:
    response = _response(loglikelihoods_per_token={" A": PerTokenScores(bits=[1.0], byte_lens=[2])})

    scores = per_token_scores_for(response, " A")

    assert scores is not None
    assert scores.bits == [1.0]
    assert scores.byte_lens == [2]


def test_per_token_scores_for_returns_none_without_per_token_data() -> None:
    assert per_token_scores_for(_response(), " A") is None
