import pytest

from eval_framework.exceptions import LogicError
from eval_framework.metrics.completion.rouge_1 import ROUGE_1
from eval_framework.metrics.completion.rouge_2 import ROUGE_2
from eval_framework.metrics.completion.rouge_geometric_mean import ROUGE_GEOMETRIC_MEAN
from eval_framework.metrics.completion.rouge_l import ROUGE_L
from eval_framework.shared.types import Completion


class TestROUGEMetrics:
    @pytest.fixture
    def completion_perfect(self) -> Completion:
        return Completion(
            id=1,
            subject="test",
            ground_truth="The capital of France is Paris.",
            prompt="What is the capital of France?",
            prompt_sequence_positions=None,
            messages=None,
            completion="The capital of France is Paris.",
            raw_completion="The capital of France is Paris.",
            raw_completion_sequence_positions=None,
            context=None,
        )

    @pytest.fixture
    def completion_partial(self) -> Completion:
        return Completion(
            id=2,
            subject="test",
            ground_truth="The Eiffel Tower is located in Paris, France and was completed in 1889.",
            prompt="Describe the Eiffel Tower.",
            prompt_sequence_positions=None,
            messages=None,
            completion="The Eiffel Tower stands in Paris and was built in 1889.",
            raw_completion="The Eiffel Tower stands in Paris and was built in 1889.",
            raw_completion_sequence_positions=None,
            context=None,
        )

    def test_perfect_match(self, completion_perfect: Completion) -> None:
        r1 = ROUGE_1().calculate(completion_perfect)[0].value
        r2 = ROUGE_2().calculate(completion_perfect)[0].value
        rl = ROUGE_L().calculate(completion_perfect)[0].value

        assert r1 == 1.0
        assert r2 == 1.0
        assert rl == 1.0

    def test_partial_match(self, completion_partial: Completion) -> None:
        """
        Tests a realistic partial match where the completion captures
        the main information but uses different phrasing
        """
        r1 = ROUGE_1().calculate(completion_partial)[0].value
        r2 = ROUGE_2().calculate(completion_partial)[0].value
        rl = ROUGE_L().calculate(completion_partial)[0].value
        assert r1 is not None
        assert r2 is not None
        assert rl is not None

        # These values represent realistic partial matches
        assert 0.5 < r1 < 1.0  # Higher as it matches individual words
        assert 0.3 < r2 < 0.8  # Lower as bigram matches are harder
        assert 0.4 < rl < 0.9  # Moderate as it finds longest common subsequence

    def test_multiple_ground_truths(self) -> None:
        completion = Completion(
            id=3,
            subject="test",
            ground_truth=[
                "Python is a programming language.",
                "Python is a popular coding language.",
                "Python is widely used in programming.",
            ],
            prompt="What is Python?",
            prompt_sequence_positions=None,
            messages=None,
            completion="Python is a programming language.",
            raw_completion="Python is a programming language.",
            raw_completion_sequence_positions=None,
            context=None,
        )

        # Should match perfectly with first ground truth
        r1 = ROUGE_1().calculate(completion)[0].value
        r2 = ROUGE_2().calculate(completion)[0].value
        rl = ROUGE_L().calculate(completion)[0].value

        assert r1 == 1.0
        assert r2 == 1.0
        assert rl == 1.0

    def test_realistic_qa_example(self) -> None:
        completion = Completion(
            id=4,
            subject="test",
            ground_truth="Albert Einstein published his theory of special relativity in 1905.",
            prompt="When did Einstein publish special relativity?",
            prompt_sequence_positions=None,
            messages=None,
            completion="Einstein published special relativity theory in 1905.",
            raw_completion="Einstein published special relativity theory in 1905.",
            raw_completion_sequence_positions=None,
            context=None,
        )

        r1 = ROUGE_1().calculate(completion)[0].value
        r2 = ROUGE_2().calculate(completion)[0].value
        rl = ROUGE_L().calculate(completion)[0].value

        # These scores reflect real-world variation in expressing the same information
        assert 0.8235294117647058 == r1  # High unigram overlap
        assert 0.4 == r2  # Moderate bigram overlap
        assert 0.7058823529411764 == rl  # Good subsequence match

    def test_empty_ground_truth(self) -> None:
        completion_none_ground_truth = Completion(
            id=5,
            subject="test",
            ground_truth=None,
            prompt="test",
            prompt_sequence_positions=None,
            messages=None,
            completion="test",
            raw_completion="test",
            raw_completion_sequence_positions=None,
            context=None,
        )

        with pytest.raises(LogicError):
            ROUGE_1().calculate(completion_none_ground_truth)
            ROUGE_2().calculate(completion_none_ground_truth)
            ROUGE_L().calculate(completion_none_ground_truth)

    def test_empty_completion(self) -> None:
        completion_empty = Completion(
            id=6,
            subject="test",
            ground_truth="test",
            prompt="test",
            prompt_sequence_positions=None,
            messages=None,
            completion="",
            raw_completion="",
            raw_completion_sequence_positions=None,
            context=None,
        )
        assert ROUGE_1().calculate(completion_empty)[0].value == 0.0
        assert ROUGE_2().calculate(completion_empty)[0].value == 0.0
        assert ROUGE_L().calculate(completion_empty)[0].value == 0.0

    def test_geometric_mean(self) -> None:
        completion = Completion(
            id=4,
            subject="test",
            ground_truth="Albert Einstein published his theory of special relativity in 1905.",
            prompt="When did Einstein publish special relativity?",
            prompt_sequence_positions=None,
            messages=None,
            completion="Einstein published special relativity theory in 1905.",
            raw_completion="Einstein published special relativity theory in 1905.",
            raw_completion_sequence_positions=None,
            context=None,
        )

        r1 = ROUGE_1().calculate(completion)[0].value
        r2 = ROUGE_2().calculate(completion)[0].value
        rl = ROUGE_L().calculate(completion)[0].value

        assert r1 is not None, "ROUGE_1 should not be None"
        assert r2 is not None, "ROUGE_2 should not be None"
        assert rl is not None, "ROUGE_L should not be None"

        geometric_mean = (r1 * r2 * rl) ** (1 / 3)
        rouge_geometric_mean = ROUGE_GEOMETRIC_MEAN().calculate(completion)[0].value

        assert geometric_mean == rouge_geometric_mean == 0.6149273508447368
