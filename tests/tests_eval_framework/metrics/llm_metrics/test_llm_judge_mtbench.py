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
    singular_result = result[0]
    assert len(result) == 1
    assert (singular_result.error is not None) == should_error


@pytest.mark.parametrize(
    "example_completion", ["context_with_reference_pair", "context_no_reference_pair"], indirect=True
)
@pytest.mark.parametrize("llm_judge,should_error", [[FakeLLMJudge(), False], [ErrorFakeLLMJudge(), True]])
def test_llm_judge_mtbench_pair_evaluate_prompt(
    example_completion: Completion, llm_judge: BaseLLM, should_error: bool
) -> None:
    metric = MTBenchJudgePair(llm_judge)
    result: list[MetricResult] = metric.calculate(example_completion)
    singular_result = result[0]
    assert len(result) == 1
    assert (singular_result.error is not None) == should_error


def test_prompt_keys() -> None:
    single_judge_keys = ["single_assistant_single_turn", "single_assistant_single_turn_w_reference"]
    for prompt_set in SINGLE_JUDGE_PROMPTS_LIST:
        for key in prompt_set.keys():
            assert key in single_judge_keys, f"Unexpected prompt key: {key}"
            assert prompt_set["prompt_template"] is not None, "Prompt template should not be None"

    multi_judge_keys = ["pair_assistant_single_turn", "pair_assistant_single_turn_w_reference"]
    for prompt_set in PAIR_JUDGE_PROMPTS_LIST:
        for key in prompt_set.keys():
            assert key in multi_judge_keys, f"Unexpected prompt key: {key}"
            assert prompt_set["prompt_template"] is not None, "Prompt template should not be None"
