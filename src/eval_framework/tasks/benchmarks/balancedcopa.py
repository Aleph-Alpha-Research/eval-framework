from datasets import Dataset, DatasetDict

from eval_framework.tasks.base import NO_SUBJECT, SubjectType
from eval_framework.tasks.benchmarks.copa import COPA


def split_dataset_by_id_ranges(
    dataset: Dataset, id_column: str, ranges: list[tuple[int, int]]
) -> tuple[Dataset, Dataset]:
    """Split a dataset into two based on whether the id column falls within given ranges.

    Args:
        dataset: The dataset to split.
        id_column: The name of the column containing the id values.
        ranges: A list of (low, high) tuples defining inclusive ranges.
            Rows whose id is within any of these ranges go into the first split.
    """

    def in_any_range(id_value: int) -> bool:
        return any(low <= id_value <= high for low, high in ranges)

    in_indices = [i for i, id_val in enumerate(dataset[id_column]) if in_any_range(id_val)]
    not_in_indices = [i for i, id_val in enumerate(dataset[id_column]) if not in_any_range(id_val)]

    return dataset.select(in_indices), dataset.select(not_in_indices)


class BalancedCOPA(COPA):
    """Balanced-COPA dataset: https://huggingface.co/datasets/pkavumba/balanced-copa"""

    NAME = "BalancedCOPA"
    DATASET_PATH = "pkavumba/balanced-copa"
    HF_REVISION = "813bd03cd6e07d9bd8d7333896ad5d40abb95ea9"
    SUBJECTS = ["no_subject"]

    def _split_dataset_into_train_and_val(self, dataset: DatasetDict) -> DatasetDict:
        # We split the train data into train and validation splits so that
        # the validation split matches the validation split of the original COPA dataset.
        # These magic numbers of the ids below were arrived at after manual inspection of the dataset.
        # The sanity of this version is maintained by the HF_REVISION above.
        dataset["validation"], dataset["train"] = split_dataset_by_id_ranges(
            dataset["train"], "id", [(401, 500), (1401, 1500)]
        )
        return dataset

    def _load_dataset(self, subject: SubjectType) -> None:
        # This method largely reimplements the _load_dataset method in the base class,
        # as the _shuffle_splits method drops any column not in FEWSHOT_SPLIT, SAMPLE_SPLIT.
        # Thus, we need to split the dataset into train and validation splits before shuffling.
        name = subject if subject != NO_SUBJECT else None
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=name)
        hf_dataset = self._split_dataset_into_train_and_val(hf_dataset)

        self.dataset = self._shuffle_splits(hf_dataset=hf_dataset)
