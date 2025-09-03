import pytest

from eval_framework.metrics.completion.text_counter import (
    ParagraphCounter,
    ParagraphCounterMetricContext,
    ResponseToOriginalLengthRatio,
    SentenceCounter,
    SentenceCounterMetricContext,
    WordCounter,
    WordCounterMetricContext,
)
from eval_framework.shared.types import Completion
from template_formatting.formatter import Message, Role


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="This is a test.",
                raw_completion="This is a test.",
                raw_completion_sequence_positions=None,
                context=WordCounterMetricContext(comparison="less than", word_count=5),
            ),
            1.0,
            id="word_counter_less_than_valid",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="This is a longer test with more words.",
                raw_completion="This is a longer test with more words.",
                raw_completion_sequence_positions=None,
                context=WordCounterMetricContext(comparison="less than", word_count=5),
            ),
            0.0,
            id="word_counter_less_than_invalid",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Short test.",
                raw_completion="Short test.",
                raw_completion_sequence_positions=None,
                context=WordCounterMetricContext(comparison="at least", word_count=2),
            ),
            1.0,
            id="word_counter_at_least_valid",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="One.",
                raw_completion="One.",
                raw_completion_sequence_positions=None,
                context=WordCounterMetricContext(comparison="at least", word_count=2),
            ),
            0.0,
            id="word_counter_at_least_invalid",
        ),
    ],
)
def test_word_counter(response: Completion, expected_value: float) -> None:
    metric = WordCounter()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Word Count"
    assert results[0].higher_is_better is True


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Sentence one. This is a second sentence.",
                raw_completion="Sentence one. This is a second sentence.",
                raw_completion_sequence_positions=None,
                context=SentenceCounterMetricContext(comparison="at least", sentence_count=2),
            ),
            1.0,
            id="sentence_counter_at_least_valid",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="One sentence.",
                raw_completion="One sentence.",
                raw_completion_sequence_positions=None,
                context=SentenceCounterMetricContext(comparison="at least", sentence_count=2),
            ),
            0.0,
            id="sentence_counter_at_least_invalid",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Mr. Jones spent 3.5 hours on www.reddit.com.",
                raw_completion="Mr. Jones spent 3.5 hours on www.reddit.com.",
                raw_completion_sequence_positions=None,
                context=SentenceCounterMetricContext(comparison="less than", sentence_count=2),
            ),
            1.0,
            id="sentence_counter_less_than_valid",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Sentence one. Sentence two. Sentence three.",
                raw_completion="Sentence one. Sentence two. Sentence three.",
                raw_completion_sequence_positions=None,
                context=SentenceCounterMetricContext(comparison="less than", sentence_count=2),
            ),
            0.0,
            id="sentence_counter_less_than_invalid",
        ),
    ],
)
def test_sentence_counter(response: Completion, expected_value: float) -> None:
    metric = SentenceCounter()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Sentence Count"
    assert results[0].higher_is_better is True


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Paragraph one.\n\nParagraph two.",
                raw_completion="Paragraph one.\n\nParagraph two.",
                raw_completion_sequence_positions=None,
                context=ParagraphCounterMetricContext(comparison="at least", paragraph_count=2),
            ),
            1.0,
            id="paragraph_counter_at_least_valid",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Single paragraph\nwith some\nnew lines.",
                raw_completion="Single paragraph\nwith some\nnew lines.",
                raw_completion_sequence_positions=None,
                context=ParagraphCounterMetricContext(comparison="less than", paragraph_count=2),
            ),
            1.0,
            id="paragraph_counter_less_than_valid",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Para 1\n\nPara 2\n\nPara 3",
                raw_completion="Para 1\n\nPara 2\n\nPara 3",
                raw_completion_sequence_positions=None,
                context=ParagraphCounterMetricContext(comparison="less than", paragraph_count=3),
            ),
            0.0,
            id="paragraph_counter_less_than_invalid",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=None,
                completion="Para 1\n\nPara 2",
                raw_completion="Para 1\n\nPara 2",
                raw_completion_sequence_positions=None,
                context=ParagraphCounterMetricContext(comparison="at least", paragraph_count=3),
            ),
            0.0,
            id="paragraph_counter_at_least_invalid",
        ),
    ],
)
def test_paragraph_counter(response: Completion, expected_value: float) -> None:
    metric = ParagraphCounter()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Paragraph Count"
    assert results[0].higher_is_better is True


@pytest.mark.parametrize(
    "response,expected_value",
    [
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=[Message(role=Role.USER, content="1234")],
                completion="1",
                raw_completion="1",
                raw_completion_sequence_positions=None,
            ),
            0.25,
            id="short_response",
        ),
        pytest.param(
            Completion(
                id=1,
                subject="test",
                ground_truth=None,
                prompt="test",
                prompt_sequence_positions=None,
                messages=[Message(role=Role.USER, content="1234")],
                completion="",
                raw_completion="",
                raw_completion_sequence_positions=None,
            ),
            0.0,
            id="no_response",
        ),
    ],
)
def test_response_to_original_length_ratio(response: Completion, expected_value: float) -> None:
    metric = ResponseToOriginalLengthRatio()
    results = metric.calculate(response)
    assert len(results) == 1
    assert results[0].value == pytest.approx(expected_value)
    assert results[0].metric_name == "Response to Original Length Ratio"
    assert results[0].higher_is_better is False


def test_response_to_original_length_ratio_error() -> None:
    response = Completion(
        id=1,
        subject="test",
        ground_truth=None,
        prompt="test",
        prompt_sequence_positions=None,
        messages=[],
        completion="123",
        raw_completion="123",
        raw_completion_sequence_positions=None,
    )
    metric = ResponseToOriginalLengthRatio()
    results = metric.calculate(response)
    assert len(results) == 0
