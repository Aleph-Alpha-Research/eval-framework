from collections.abc import Sequence
from typing import Any

import pytest

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.llm_judge_mtbench_pair import (
    PAIR_JUDGE_PROMPTS_LIST,
    MTBenchJudgePair,
    MTBenchJudgePairMetricContext,
    generate_pair_judge_prompts,
)
from eval_framework.metrics.llm.llm_judge_mtbench_single import (
    SINGLE_JUDGE_PROMPTS_LIST,
    MTBenchJudgeSingle,
    MTBenchJudgeSingleMetricContext,
)
from eval_framework.shared.types import Completion, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import Message


class FakeLLMJudge(BaseLLM):
    def __init__(self) -> None:
        pass

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        *_: Any,
    ) -> list[RawCompletion]:
        return [
            RawCompletion(
                prompt="prompt",
                completion="Rating: [[5]]",
                prompt_sequence_positions=None,
                completion_sequence_positions=None,
            )
        ] * len(messages)

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        raise NotImplementedError


class ErrorFakeLLMJudge(BaseLLM):
    def __init__(self) -> None:
        pass

    def generate_from_messages(
        self,
        *_: Any,
    ) -> list[RawCompletion]:
        raise RuntimeError("Intentional error for testing.")

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        raise NotImplementedError


@pytest.fixture
def context_no_reference() -> MTBenchJudgeSingleMetricContext:
    return MTBenchJudgeSingleMetricContext(
        category="testing",
        reference=None,
    )


@pytest.fixture
def context_with_reference() -> MTBenchJudgeSingleMetricContext:
    return MTBenchJudgeSingleMetricContext(
        category="math",
        reference="42",
    )


@pytest.fixture
def context_no_reference_pair() -> MTBenchJudgePairMetricContext:
    return MTBenchJudgePairMetricContext(
        category="testing",
        answer=["42"],
        reference=None,
    )


@pytest.fixture
def context_with_reference_pair() -> MTBenchJudgePairMetricContext:
    return MTBenchJudgePairMetricContext(
        category="math",
        answer=["42"],
        reference="42",
    )


@pytest.fixture
def example_completion(request) -> Completion:  # type: ignore
    # ignoring the type here since request is a special fixture
    # Get the context fixture by name from the request parameter
    context = request.getfixturevalue(request.param)
    return Completion(
        context=context,
        id=1,
        subject="math",
        ground_truth="42",
        prompt="What is 6 multiplied by 7?",
        prompt_sequence_positions=None,
        messages=[],
        completion="The answer is 42.",
        raw_completion="The answer is 42.",
        raw_completion_sequence_positions=None,
    )


@pytest.mark.parametrize("example_completion", ["context_with_reference", "context_no_reference"], indirect=True)
@pytest.mark.parametrize("llm_judge,should_error", [[FakeLLMJudge(), False], [ErrorFakeLLMJudge(), True]])
def test_llm_judge_mtbench_single_evaluate_prompt(
    example_completion: Completion, llm_judge: BaseLLM, should_error: bool
) -> None:
    metric = MTBenchJudgeSingle(llm_judge)
    result: list[MetricResult] = metric.calculate(example_completion)
    assert len(result) == 1
    singular_result = result[0]
    assert (singular_result.error is not None) == should_error
    if not (should_error):
        assert singular_result.llm_judge_prompt
        assert singular_result.llm_judge_response


@pytest.mark.parametrize(
    "example_completion", ["context_with_reference_pair", "context_no_reference_pair"], indirect=True
)
@pytest.mark.parametrize("llm_judge,should_error", [[FakeLLMJudge(), False], [ErrorFakeLLMJudge(), True]])
def test_llm_judge_mtbench_pair_evaluate_prompt(
    example_completion: Completion, llm_judge: BaseLLM, should_error: bool
) -> None:
    metric = MTBenchJudgePair(llm_judge)
    result: list[MetricResult] = metric.calculate(example_completion)
    assert len(result) == 1
    singular_result = result[0]
    assert (singular_result.error is not None) == should_error
    if not (should_error):
        assert singular_result.llm_judge_prompt
        assert singular_result.llm_judge_response


def test_prompt_scenarios_are_covered() -> None:
    required_scenarios = [
        ("single_turn", "without_reference"),
        ("multi_turn", "without_reference"),
        ("single_turn", "with_reference"),
        ("multi_turn", "with_reference"),
    ]

    def check_scenarios_coverage(prompt_sets: list[dict[str, dict[str, str]]], judge_type: str) -> None:
        covered_scenarios = set()

        for prompt_set in prompt_sets:
            assert len(prompt_set) > 0, f"Prompt set for {judge_type} is empty."

            for key, prompt_data in prompt_set.items():
                assert prompt_data.get("prompt_template") is not None, (
                    f"Prompt template missing for key: {key} in {judge_type}"
                )

                turn = "multi_turn" if "multi_turn" in key else "single_turn"
                reference = "with_reference" if "w_reference" in key else "without_reference"

                covered_scenarios.add((turn, reference))

        assert covered_scenarios == set(required_scenarios), (
            f"Required {judge_type} scenarios not fully covered. Missing: {set(required_scenarios) - covered_scenarios}"
        )

    check_scenarios_coverage(SINGLE_JUDGE_PROMPTS_LIST, "Single Judge")
    check_scenarios_coverage(PAIR_JUDGE_PROMPTS_LIST, "Pair Judge")


class TestOutputToRating:
    """Tests for MTBenchJudgePair._output_to_rating with position-aware scoring."""

    @pytest.mark.parametrize(
        "output,candidate_is_a,expected_score",
        [
            # When candidate is in position A
            ("The winner is [[A]]", True, 1.0),  # A wins -> candidate wins
            ("The winner is [[B]]", True, 0.0),  # B wins -> candidate loses
            ("It's a [[C]] tie", True, 0.5),  # Tie
            # When candidate is in position B (swapped)
            ("The winner is [[A]]", False, 0.0),  # A wins -> candidate loses (candidate is B)
            ("The winner is [[B]]", False, 1.0),  # B wins -> candidate wins (candidate is B)
            ("It's a [[C]] tie", False, 0.5),  # Tie stays tie
            # Unparseable outputs default to tie
            ("No clear winner", True, 0.5),
            ("No clear winner", False, 0.5),
            ("[[X]] invalid", True, 0.5),
        ],
    )
    def test_output_to_rating_with_position(self, output: str, candidate_is_a: bool, expected_score: float) -> None:
        score = MTBenchJudgePair._output_to_rating(output, candidate_is_a=candidate_is_a)
        assert score == expected_score


class TestPairJudgePromptsRandomization:
    """Tests for randomization in generate_pair_judge_prompts."""

    @pytest.fixture
    def single_turn_completion(self) -> Completion:
        """Create a single-turn completion for testing."""
        context = MTBenchJudgePairMetricContext(
            category="testing",
            answer=["Reference answer here"],
            reference=None,
        )
        return Completion(
            context=context,
            id=42,
            subject="en_test",
            ground_truth="42",
            prompt="What is the question?",
            prompt_sequence_positions=None,
            messages=[
                Message(role="user", content="What is the question?"),
            ],
            completion="Candidate answer here",
            raw_completion="Candidate answer here",
            raw_completion_sequence_positions=None,
        )

    def test_randomize_order_with_seed_deterministic(self, single_turn_completion: Completion) -> None:
        """Test that the same seed produces the same ordering."""
        prompts_1 = generate_pair_judge_prompts(single_turn_completion, randomize_order=True, seed=123)
        prompts_2 = generate_pair_judge_prompts(single_turn_completion, randomize_order=True, seed=123)

        assert len(prompts_1) == 1
        assert len(prompts_2) == 1
        assert prompts_1[0].candidate_is_a == prompts_2[0].candidate_is_a
        assert prompts_1[0].prompt_text == prompts_2[0].prompt_text

    def test_randomize_order_can_produce_both_outcomes(self, single_turn_completion: Completion) -> None:
        """Randomization can produce both True and False outcomes."""
        import random

        # Find seeds that produce each outcome (deterministic search)
        seed_for_true = next(i for i in range(100) if not random.Random(i).choice([True, False]))
        seed_for_false = next(i for i in range(100) if random.Random(i).choice([True, False]))

        prompt_true = generate_pair_judge_prompts(single_turn_completion, randomize_order=True, seed=seed_for_true)
        prompt_false = generate_pair_judge_prompts(single_turn_completion, randomize_order=True, seed=seed_for_false)

        assert prompt_true[0].candidate_is_a is True
        assert prompt_false[0].candidate_is_a is False

    def test_no_randomization_keeps_candidate_as_a(self, single_turn_completion: Completion) -> None:
        """Test that disabling randomization always puts candidate in position A."""
        prompts = generate_pair_judge_prompts(single_turn_completion, randomize_order=False)

        assert len(prompts) == 1
        assert prompts[0].candidate_is_a is True
        # Verify candidate answer appears in position A
        assert "Candidate answer here" in prompts[0].prompt_text.split("Assistant A")[1].split("Assistant B")[0]

    def test_prompt_contains_correct_answers_when_swapped(self, single_turn_completion: Completion) -> None:
        """Test that both answers appear in prompt regardless of ordering."""
        prompts = generate_pair_judge_prompts(single_turn_completion, randomize_order=True, seed=0)

        prompt_text = prompts[0].prompt_text
        # Both answers should be in the prompt
        assert "Candidate answer here" in prompt_text
        assert "Reference answer here" in prompt_text

    def test_default_seed_uses_response_id(self, single_turn_completion: Completion) -> None:
        """Test that default randomization uses response id as seed for reproducibility."""
        # With the same completion (same id), should get the same result
        prompts_1 = generate_pair_judge_prompts(single_turn_completion, randomize_order=True)
        prompts_2 = generate_pair_judge_prompts(single_turn_completion, randomize_order=True)

        assert prompts_1[0].candidate_is_a == prompts_2[0].candidate_is_a
