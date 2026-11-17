import pytest

from eval_framework.shared.types import BaseMetricContext, Completion, extract_context_metric


class _WantedContext(BaseMetricContext):
    value: str


class _OtherContext(BaseMetricContext):
    other: int


def _response(context: list[BaseMetricContext] | BaseMetricContext | None) -> Completion:
    return Completion(
        id=0,
        subject="s",
        ground_truth="a",
        prompt="p",
        prompt_num_tokens=None,
        messages=None,
        completion="a",
        raw_completion="a",
        raw_completion_num_tokens=None,
        context=context,
    )


def test_returns_the_first_matching_context_from_a_list() -> None:
    wanted = _WantedContext(value="x")

    found = extract_context_metric(_response([_OtherContext(other=1), wanted]), _WantedContext)

    assert found is wanted


def test_list_without_the_requested_context_fails_with_a_clear_assertion() -> None:
    with pytest.raises(AssertionError, match="Expected _WantedContext to be provided"):
        extract_context_metric(_response([_OtherContext(other=1)]), _WantedContext)


def test_single_context_is_returned_as_is() -> None:
    wanted = _WantedContext(value="x")

    assert extract_context_metric(_response(wanted), _WantedContext) is wanted
