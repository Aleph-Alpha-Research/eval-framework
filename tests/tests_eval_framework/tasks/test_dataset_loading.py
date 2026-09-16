"""Unit tests for the row-shaping dataset policies.

``Subset`` and ``SubjectColumn`` wrap another policy to carve a benchmark's rows out of a larger dataset,
so their filtering is specified here once rather than re-tested through every benchmark that uses them.
The fluent ``DatasetPolicy.subset`` / ``DatasetPolicy.subject_encoded_in_column`` constructors are exercised
along the way.
"""

from typing import Any, final, override

from datasets import Dataset, DatasetDict

from eval_framework.tasks.dataset_loading import DatasetLoader, DatasetPolicy
from tests.tests_eval_framework.benchmarks.utils import DatasetStub


def test_subset_keeps_only_the_rows_matching_the_predicate() -> None:
    # Given a dataset mixing rows that pass and fail a predicate, in more than one split
    rows = [{"q": "Q1", "keep": True}, {"q": "Q2", "keep": False}, {"q": "Q3", "keep": True}]
    policy = DatasetStub({"train": rows, "test": rows}).subset(lambda row: row["keep"])

    # When the policy loads its splits
    loaded = policy.loader(None).load(None)

    # Then every split drops the rows the predicate rejects
    assert [row["q"] for row in loaded["train"]] == ["Q1", "Q3"]
    assert [row["q"] for row in loaded["test"]] == ["Q1", "Q3"]


def test_subject_column_keeps_only_the_rows_whose_column_names_the_requested_subject() -> None:
    # Given one config that packs two subjects, tagged by a column (like German ARC's "deu" config)
    rows = [
        {"q": "Q1", "arc_config": "ARC-Easy"},
        {"q": "Q2", "arc_config": "ARC-Challenge"},
        {"q": "Q3", "arc_config": "ARC-Easy"},
    ]
    policy = DatasetStub({"test": rows}).subject_encoded_in_column(config="deu", column="arc_config")

    # When each subject is loaded, then it sees only the rows whose column matches
    assert [row["q"] for row in policy.loader(None).load("ARC-Easy")["test"]] == ["Q1", "Q3"]
    assert [row["q"] for row in policy.loader(None).load("ARC-Challenge")["test"]] == ["Q2"]


def test_subset_documentation_names_the_restriction_on_top_of_the_inner_docs() -> None:
    # Given a subset with a human-readable description, over an inner policy that has its own docs
    policy = DatasetStub({"train": []}).subset(lambda row: True, description="the diamond subset")

    # When the dataset section is rendered, then it keeps the inner docs and adds the restriction
    doc = policy.documentation()
    assert "fictional in-memory dataset" in doc
    assert "- Restricted to the diamond subset." in doc


def test_subject_column_documentation_names_the_config_and_column() -> None:
    # Given a column-encoded subject policy
    policy = DatasetStub({"test": []}).subject_encoded_in_column(config="deu", column="arc_config")

    # When the dataset section is rendered, then it explains the shared config and the splitting column
    doc = policy.documentation()
    assert "fictional in-memory dataset" in doc
    assert "`deu`" in doc and "`arc_config`" in doc


def test_subject_column_always_loads_its_fixed_config_regardless_of_the_requested_subject() -> None:
    # Given an inner policy that records which config it is asked to load
    inner = _RecordingPolicy([{"arc_config": "ARC-Easy"}, {"arc_config": "ARC-Challenge"}])
    loader = inner.subject_encoded_in_column(config="deu", column="arc_config").loader(None)

    # When different subjects are loaded
    loader.load("ARC-Easy")
    loader.load("ARC-Challenge")

    # Then the config that reaches the inner loader is the fixed one, never the subject name
    assert inner.requested_configs == ["deu", "deu"]


@final
class _RecordingLoader(DatasetLoader):
    """Records the config name it is asked to load, then serves fixed rows regardless."""

    def __init__(self, rows: list[dict[str, Any]], requested_configs: list[str | None]) -> None:
        self._rows = rows
        self._requested_configs = requested_configs

    @override
    def load(self, name: str | None) -> DatasetDict:
        self._requested_configs.append(name)
        return DatasetDict({"test": Dataset.from_list(self._rows)})

    @override
    def metadata(self) -> dict[str, str]:
        return {"dataset_path": "recording"}


@final
class _RecordingPolicy(DatasetPolicy):
    """A policy whose loader records every config name it is asked to load."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.requested_configs: list[str | None] = []

    @override
    def loader(self, custom_hf_revision: str | None) -> DatasetLoader:
        return _RecordingLoader(self._rows, self.requested_configs)

    @override
    def documentation(self) -> str:
        return "recording policy"
