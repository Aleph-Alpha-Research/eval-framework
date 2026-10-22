import math

import pytest
from pydantic import ValidationError

from eval_framework.metrics.loglikelihood.bpb_variants_common import aggregate_prior_bpb_metrics
from eval_framework.shared.types import Loglikelihood, MaskedTokenScores


def _valid_scores() -> dict:
    return {
        "bits": [1.0, 2.0, 3.0],
        "byte_lens": [2, 0, 3],
        "token_indices": [2, 3, 7],
        "byte_spans": [(3, 5), (12, 16)],
        "requested_bytes": 6,
        "total_input_tokens": 10,
    }


def test_masked_scores_allow_shared_bytes_and_excluded_boundaries() -> None:
    scores = MaskedTokenScores(**_valid_scores())
    assert sum(scores.byte_lens) == 5 < scores.requested_bytes
    assert MaskedTokenScores.model_validate_json(scores.model_dump_json()) == scores


@pytest.mark.parametrize(
    "overrides",
    [
        {"bits": []},
        {"bits": [1.0]},
        {"bits": [1.0, float("nan"), 3.0]},
        {"bits": [1.0, float("inf"), 3.0]},
        {"bits": [1.0, -0.1, 3.0]},
        {"byte_lens": [2, -1, 3]},
        {"byte_lens": [0, 0, 0]},
        {"byte_lens": [4, 0, 3]},
        {"token_indices": [2, 3]},
        {"token_indices": [2, 2, 7]},
        {"token_indices": [7, 3, 2]},
        {"token_indices": [-1, 3, 7]},
        {"token_indices": [2, 3, 10]},
        {"byte_spans": [(-1, 1), (12, 16)]},
        {"byte_spans": [(3, 3), (12, 18)]},
        {"byte_spans": [(3, 7), (5, 7)]},
        {"byte_spans": [(12, 16), (3, 5)]},
        {"byte_spans": [(3, 5)]},
        {"byte_spans": []},
    ],
)
def test_masked_scores_reject_invalid_audit_data(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        MaskedTokenScores(**(_valid_scores() | overrides))


def test_masked_scores_are_not_used_for_completion_prefix_estimators() -> None:
    scores = MaskedTokenScores(**_valid_scores())
    response = Loglikelihood(
        id=0,
        subject="trajectory",
        ground_truth="full trajectory text",
        prompt="full trajectory text",
        prompt_num_tokens=scores.total_input_tokens,
        loglikelihoods={"full trajectory text": -math.log(2) * sum(scores.bits)},
        loglikelihoods_num_tokens={"full trajectory text": len(scores.bits)},
        masked_loglikelihoods={"full trajectory text": scores},
    )
    assert response.loglikelihoods_per_token == {}
    assert aggregate_prior_bpb_metrics([response]) == {}


def test_legacy_loglikelihoods_do_not_require_masked_scores() -> None:
    response = Loglikelihood(
        id=0,
        subject="legacy",
        ground_truth="answer",
        prompt="question",
        prompt_num_tokens=None,
        loglikelihoods={"answer": -1.0},
        loglikelihoods_num_tokens={"answer": 1},
    )
    assert response.masked_loglikelihoods == {}
