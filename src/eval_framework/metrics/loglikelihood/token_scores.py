"""Derive per-token BPB inputs from raw token logprobs.

The LLM adapter forwards :class:`TokenLogprob` (natural-log probability and decoded
text) for the tokens covering each scored continuation. Prefix and Prior BPB need
surprisal in bits and UTF-8 byte lengths, which this module computes. Keeping the
arithmetic here, rather than in the vLLM client, is the point of the refactor: the
adapter should not know these numbers feed BPB.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from eval_framework.shared.types import BaseLoglikelihood, PerTokenScores, TokenLogprob

_LN2 = math.log(2)


def per_token_scores(continuation: str, tokens: Sequence[TokenLogprob]) -> PerTokenScores:
    """Surprisal in bits and UTF-8 byte length for each token covering ``continuation``.

    ``bits[j] = -log2 p(token j)``, taken from the whole token. ``byte_lens`` sums to the
    continuation's UTF-8 length. The first token may start inside the prompt (continuation
    ``A`` scored as token `` A``); only the bytes that fall inside the continuation are
    counted, so the first token's prompt-side bytes are removed.
    """
    byte_lens = [len(token.decoded_token.encode("utf-8")) for token in tokens]
    if byte_lens:
        # Excess is a prefix of the first token: covered text is the continuation with
        # that token's prompt-side bytes still attached.
        byte_lens[0] -= sum(byte_lens) - len(continuation.encode("utf-8"))
    return PerTokenScores(
        bits=[float(-token.logprob / _LN2) for token in tokens],
        byte_lens=byte_lens,
    )


def per_token_scores_for(response: BaseLoglikelihood, continuation: str) -> PerTokenScores | None:
    """Per-token scores for ``continuation``, or ``None`` when the backend reported none.

    Prefers the raw ``token_logprobs`` and derives bits/byte lengths from them. Falls back
    to the deprecated ``loglikelihoods_per_token`` so old adapters and saved runs keep
    working. This is the branch point between the new and old producers.
    """
    tokens = response.token_logprobs.get(continuation)
    if tokens is not None:
        return per_token_scores(continuation, tokens)
    return response.loglikelihoods_per_token.get(continuation)
