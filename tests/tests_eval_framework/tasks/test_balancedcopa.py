import pytest
from datasets import Dataset

from eval_framework.tasks.benchmarks.balancedcopa import split_dataset_by_id_ranges


@pytest.fixture
def dummy_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "idx": list(range(1, 11)),
            "text": [f"row-{i}" for i in range(1, 11)],
        }
    )


class TestSplitDatasetByIdRanges:
    """Tests for split_dataset_by_id_ranges covering a variety of edge cases."""

    def test_basic_split_with_multiple_ranges(self, dummy_dataset: Dataset) -> None:
        # Ranges: [2,4] and [8,9] gives the ids: {2,3,4,8,9} in matched and the rest in rest.
        matched, rest = split_dataset_by_id_ranges(dummy_dataset, "idx", [(2, 4), (8, 9)])

        # nothing is lost.
        assert len(matched) + len(rest) == len(dummy_dataset)

        # correct ids in each split
        assert matched["idx"] == [2, 3, 4, 8, 9]
        assert rest["idx"] == [1, 5, 6, 7, 10]

        # text  values match
        for row in matched:
            assert row["text"] == f"row-{row['idx']}"
        for row in rest:
            assert row["text"] == f"row-{row['idx']}"
