import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Generic, TypeVar
from unittest.mock import patch

from datasets import Dataset, DatasetDict

from eval_framework.constants import RED, RESET
from eval_framework.result_processors.base import Result
from eval_framework.tasks.base import BaseTask, SubjectType
from eval_framework.tasks.dataloader import HFDataloader

T = TypeVar("T", bound=BaseTask)


HASHES_FILE = Path(__file__).parent / "tasks" / "task-prompts-hashes.json"

logger = logging.getLogger(__name__)


def pretty_print(results: list[Result]) -> None:
    for result in results:
        logger.info(f"{RED}{result.llm_name} |{result.metric_name}: {result.value} | {result.subject}{RESET}")


def _almost_equal(x: float, y: float) -> bool:
    return 2 * abs(x - y) / abs(x + y + 1e-5) < 1e-4


def assert_hash_string(task_name: str, suffix_key: str, tested_string: str) -> str:
    """
    Compute the MD5 hash of a formatted sample or another string and verify or store it in a JSON file.
    """
    data_to_hash = tested_string.encode() if isinstance(tested_string, str) else tested_string
    tested_string_hash = hashlib.md5(data_to_hash).hexdigest()

    key = f"{task_name}.{suffix_key}"

    # Safely load the JSON dictionary of hashes
    if HASHES_FILE.exists():
        with HASHES_FILE.open("r", encoding="utf-8") as f:
            all_hashes = json.load(f)
    else:
        all_hashes = {}

    print("---")
    print(f"Tested string:\n---START---{tested_string}---END---")
    print(f"Hash: {tested_string_hash}")
    print("---")

    if key in all_hashes:
        assert all_hashes[key] == tested_string_hash, (
            f"Hash mismatch for key: {key}\nExpected: {all_hashes[key]}\nActual: {tested_string_hash}\n"
        )
    else:
        all_hashes[key] = tested_string_hash
        with HASHES_FILE.open("w", encoding="utf-8") as f:
            json.dump(dict(sorted(all_hashes.items())), f, indent=2, ensure_ascii=False)

        assert False, f"Hash for key '{key}' not found in {HASHES_FILE}. It was added for future local runs."

    return tested_string_hash


def create_mock_load_hf_dataset(
    subjects: list[SubjectType], captured_kwargs_list: list[dict[str, Any]]
) -> Callable[[Any], DatasetDict]:
    def mock_load_hf_dataset(self: BaseTask, **kwargs: Any) -> DatasetDict:
        # Find which subject this call corresponds to by matching kwargs
        for subject, captured_kwargs in zip(subjects, captured_kwargs_list):
            if kwargs == captured_kwargs:
                # return load_dataset('json', data_files=f'{subject}_data.json')

                with open(f"{subject}_data.json", "r") as f:
                    data = json.load(f)

                # Convert the JSON data back to the expected dataset format
                dataset_dict = {}
                for split_name, split_data in data.items():
                    dataset_dict[split_name] = Dataset.from_list(split_data)

                return DatasetDict(dataset_dict)
        raise ValueError(f"No matching subject found for kwargs: {kwargs}")

    return mock_load_hf_dataset


class DatasetPatcher(Generic[T]):
    def __init__(self, task_class: type[T], num_samples: int = 2, num_fewshot: int = 0):
        self.task_class = task_class
        self.num_fewshot = num_fewshot
        self.num_samples = num_samples
        self.cache_dir = Path(os.environ.get("HF_DATASET_CACHE_DIR", Path.home() / ".cache"))
        self.cache_dir = self.cache_dir / "test_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tmp_cache_dir = Path.home() / ".cache" / "tmp" / "huggingface" / "datasets"
        self.patch_obj = None

    def __enter__(self) -> T:
        dataloader = HFDataloader()
        task = self.task_class(num_fewshot=self.num_fewshot, dataloader=dataloader)

        # First, we record what arguments are passed to load_hf_dataset for each subject
        captured_kwargs = []

        def mock_get_arguments(**kwargs: Any) -> None:
            captured_kwargs.append(kwargs)

        with patch.object(dataloader, "load", side_effect=mock_get_arguments):
            for subject in task.SUBJECTS:
                # We expect it to error out because mock_get_arguments returns None
                try:
                    task._load_dataset(subject)
                    raise Exception("dataloader.load should have errored out")
                except Exception:
                    pass

        # Next, we use the captured (kw)args to load the dataset and turn them into streaming datasets
        # This will allow us to get data without having to download the entire dataset
        # (even though we pass cache_dir)
        for subject, kwargs in zip(task.SUBJECTS, captured_kwargs):
            json_path = self.cache_dir / f"{kwargs['path'].replace('/', '_')}_{subject}_{self.num_samples}.json"
            if json_path.exists():
                continue

            kwargs_copy = kwargs.copy()
            kwargs_copy["streaming"] = True

            # Patch HF_DATASET_CACHE_DIR environment variable when loading the dataset
            with patch.dict("os.environ", {"HF_DATASET_CACHE_DIR": str(self.tmp_cache_dir)}):
                hf_dataset = dataloader.load(**kwargs_copy)

            extracted_items = {}

            if hasattr(hf_dataset, "items"):
                dataset_items = hf_dataset.items()
            else:
                dataset_items = [("train", hf_dataset)]  # Default to 'train' split name

            for split, data in dataset_items:
                data_list = []
                for n, item in enumerate(data if hasattr(data, "__iter__") else []):
                    if n >= self.num_samples:
                        break
                    data_list.append(item)
                extracted_items[split] = data_list

            with open(json_path, "w") as json_file:
                json.dump(extracted_items, json_file, indent=4)

        if self.tmp_cache_dir.exists():
            shutil.rmtree(self.tmp_cache_dir)

        # Patch the dataloader's load_hf_dataset method to return the data subsets we extracted
        mock_load_hf_dataset = create_mock_load_hf_dataset(task.SUBJECTS, captured_kwargs)
        self.patch_obj = patch.object(dataloader, "load", mock_load_hf_dataset)  # type: ignore[assignment]
        assert self.patch_obj is not None
        self.patch_obj.__enter__()

        return task

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        if self.patch_obj:
            self.patch_obj.__exit__(exc_type, exc_val, exc_tb)
