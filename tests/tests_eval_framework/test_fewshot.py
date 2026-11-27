"""Specification of the ``FirstOf`` few-shot source: a fixed list of demonstrations, shown from the start."""

import random

import pytest

from eval_framework.fewshot import FirstOf

_DEMOS = [{"question": "first"}, {"question": "second"}, {"question": "third"}]
_ITEM = {"question": "the eval item"}


def _shown(source: FirstOf, num_fewshot: int, *, item: dict = _ITEM, seed: int = 0) -> list[dict]:
    """The demonstrations ``source`` shows before ``item`` when ``num_fewshot`` are requested."""
    pool = source.build_pool({}, sample_split="test", sample_rows=[item])
    count = source.resolve_count(num_fewshot)
    return source.draw(pool, item=item, count=count, rnd=random.Random(seed))


def test_first_of_shows_the_first_demonstrations_in_list_order() -> None:
    # Given a source over three demonstrations
    source = FirstOf(_DEMOS, label="Task")
    # When two are requested
    shown = _shown(source, 2)
    # Then the first two are shown, in list order:
    assert shown == [{"question": "first"}, {"question": "second"}]


def test_first_of_shows_the_same_demonstrations_for_every_item_and_seed() -> None:
    # Given a source over three demonstrations
    source = FirstOf(_DEMOS, label="Task")
    # When two are requested for different items under different seeds
    shown_a = _shown(source, 2, item={"question": "one item"}, seed=0)
    shown_b = _shown(source, 2, item={"question": "another item"}, seed=123)
    # Then both items see the same demonstrations:
    assert shown_a == shown_b


def test_first_of_shows_nothing_at_zero_shot() -> None:
    # Given a source over three demonstrations
    source = FirstOf(_DEMOS, label="Task")
    # When none are requested
    shown = _shown(source, 0)
    # Then none are shown:
    assert shown == []


def test_first_of_allows_requesting_the_whole_list() -> None:
    # Given a source over three demonstrations
    source = FirstOf(_DEMOS, label="Task")
    # When all three are requested
    shown = _shown(source, 3)
    # Then all three are shown:
    assert shown == _DEMOS


def test_first_of_rejects_more_demonstrations_than_it_holds() -> None:
    # Given a source over three demonstrations
    source = FirstOf(_DEMOS, label="Task")
    # When four are requested
    # Then it fails, naming the task and how many it holds:
    with pytest.raises(ValueError, match="Task has 3 predefined demonstrations, but 4 were requested"):
        source.resolve_count(4)
