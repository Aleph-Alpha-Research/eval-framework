import pytest

from eval_framework.exceptions import LogicError
from eval_framework.metrics.completion.bleu import BLEU, ResponseToOriginalBLEU
from eval_framework.shared.types import Completion
from template_formatting.formatter import Message, Role


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="wow, a cute dog!",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="wow, an ugly cat!",
                raw_completion="wow, an ugly cat!",
                raw_completion_sequence_positions=None,
            ),
            17.965205598154213,
            id="bleu_somewhat_different",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="wow, a cute dog!",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="wow, a cute dog!",
                raw_completion="wow, a cute dog!",
                raw_completion_sequence_positions=None,
            ),
            pytest.approx(100.0),
            id="bleu_exact_match",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="holy moly, what kind of a rocket is that?",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="oh this is a cute dog!",
                raw_completion="oh this is a cute dog!",
                raw_completion_sequence_positions=None,
            ),
            4.410363736106611,
            id="bleu_completely_different",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="some ground truth",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="",
                raw_completion="",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="bleu_mising_completion_is_maximally_different",
        ),
    ],
)
def test_bleu(response: Completion, expected_value: float) -> None:
    metric = BLEU()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "BLEU"
    assert results[0].higher_is_better is True


@pytest.mark.parametrize(
    "completion",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth="",
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="A",
                raw_completion="A",
                raw_completion_sequence_positions=None,
            ),
            id="bleu_missing_ground_truth",
        ),
    ],
)
def test_bleu_errors(completion: Completion) -> None:
    metric = BLEU()
    with pytest.raises(LogicError):
        metric.calculate(completion)


def test_response_to_original_bleu() -> None:
    completion_without_gt = Completion(
        id=1,
        subject="test",
        ground_truth=None,
        prompt="test",
        prompt_sequence_positions=None,
        messages=[Message(role=Role.USER, content="wow, a cute dog!")],
        completion="wow, an ugly cat!",
        raw_completion="wow, an ugly cat!",
        raw_completion_sequence_positions=None,
    )

    completion_with_gt = completion_without_gt.model_copy()
    completion_with_gt.ground_truth = "wow, a cute dog!"

    metric_rto = ResponseToOriginalBLEU()
    metric_bleu = BLEU()

    value_rto = metric_rto.calculate(completion_without_gt)[0].value
    value_bleu = metric_bleu.calculate(completion_with_gt)[0].value
    assert value_rto is not None
    assert value_bleu is not None
    assert value_rto * 100 == value_bleu
