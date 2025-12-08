from collections.abc import Sequence
from typing import Any

import pytest

from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.llm_judge_mtbench_pair import (
    PAIR_JUDGE_PROMPTS_LIST,
    MTBenchJudgePair,
    MTBenchJudgePairMetricContext,
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
