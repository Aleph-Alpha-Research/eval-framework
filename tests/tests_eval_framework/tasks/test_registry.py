from collections.abc import Sequence
from typing import override

import pytest

from eval_framework.contract import Benchmark, Eval, ResponseType
from eval_framework.tasks.registry import Lazy, Registry
from template_formatting.formatter import BaseFormatter


class _BenchmarkStub(Benchmark):
    """A minimal benchmark used only to exercise registry registration and lookup — its topic is
    irrelevant."""

    def __init__(self, id: str = "BenchmarkStub") -> None:
        self._id = id

    @override
    def id(self) -> str:
        return self._id

    @override
    def response_type(self) -> ResponseType:
        return ResponseType.COMPLETION

    @override
    def metrics(self) -> list:
        return []

    @override
    def subjects(self, tokens: list[str] | None = None) -> list[str]:
        return ["stub"]

    @override
    def display_name(self) -> str:
        return "Benchmark Stub"

    @override
    def create(
        self,
        num_fewshot: int,
        custom_subjects: list[str] | None,
        custom_hf_revision: str | None,
        seed: int | None = None,
    ) -> Eval:
        raise NotImplementedError("the registry never creates evals itself")

    @override
    def markdown_doc(self, formatters: Sequence[BaseFormatter]) -> str:
        return ""


def test_case_insensitive_lookup() -> None:
    # Given benchmarks registered under their ids
    registry = Registry()
    registry.add(_BenchmarkStub())

    # Then lookups normalize case, whitespace and separators onto the registered key
    assert "Benchmark Stub" in registry
    assert set(registry.task_names()) == {"BenchmarkStub"}
    assert registry["Benchmark Stub"].id() == "BenchmarkStub"
    assert registry["benchmark stub"].id() == "BenchmarkStub"
    assert registry["BenchmarkStub"].id() == "BenchmarkStub"

    registry.add(_BenchmarkStub(id="MultiWordBenchmarkStub"))
    assert set(registry.task_names()) == {"BenchmarkStub", "MultiWordBenchmarkStub"}
    assert registry["multi word benchmark stub"].id() == "MultiWordBenchmarkStub"
    assert registry["MULTI WORD BENCHMARK STUB"].id() == "MultiWordBenchmarkStub"
    assert registry["Multi Word     Benchmark Stub"].id() == "MultiWordBenchmarkStub"
    assert registry["Multi-Word_Benchmark-Stub"].id() == "MultiWordBenchmarkStub"

    with pytest.raises(ValueError):
        registry["Multi.Word.Benchmark.Stub"]


def test_duplicate_registration_is_rejected() -> None:
    # Given a registered benchmark, When adding another whose id normalizes to the same key
    registry = Registry()
    registry.add(_BenchmarkStub())

    # Then the second registration is rejected
    with pytest.raises(ValueError, match="duplicate"):
        registry.add(_BenchmarkStub(id="Benchmark-Stub"))


def test_lazy_registration_defers_loading_until_lookup() -> None:
    # Given a lazily registered benchmark whose load is counted
    registry = Registry()
    loads: list[int] = []

    def load() -> Benchmark:
        loads.append(1)
        return _BenchmarkStub()

    registry.add(Lazy(id="BenchmarkStub", load=load))

    # Then registration alone loads nothing, and a lookup delegates to the loaded benchmark
    assert loads == []
    assert registry["Benchmark Stub"].display_name() == "Benchmark Stub"
    assert registry["Benchmark Stub"].subjects() == ["stub"]
    assert loads == [1]  # memoized: both delegating calls share one load
