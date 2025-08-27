import os
from pathlib import Path
from typing import Any

from pytest import MonkeyPatch

from eval_framework.llm.models import Llama31_8B_HF
from eval_framework.result_processors.hf_processor import HFProcessor
from eval_framework.task_names import TaskName
from eval_framework.tasks.eval_config import EvalConfig


# custom class to be the mock return value from the instzantiation of the HuggingFace API class.
# will override the requests.Response returned from requests.get
class MockHFAPI:
    # mock upload_file method that passes without throwing an exception if the upload was successful
    @staticmethod
    def upload_file(path_or_fileobj: str, path_in_repo: str, repo_id: str, repo_type: str = "dataset") -> None:
        # no actual operation (upload) in the mock. No exception thrown
        return None


def test_hf_processor_with_mocked_hf_api(monkeypatch: MonkeyPatch) -> None:
    task_name = TaskName.ARC
    config = EvalConfig(
        output_dir=Path(__file__).parent / "eval_framework_results",
        hf_upload_dir="",
        hf_upload_repo="",
        num_fewshot=5,
        num_samples=10,
        task_name=task_name,
        llm_class=Llama31_8B_HF,
        description="dummy",
    )

    # Any arguments may be passed and mock_get() will always return our
    # mocked object, which only has the .json() method.
    def mock_get(*args: Any, **kwargs: Any) -> MockHFAPI:  # type ignore
        return MockHFAPI()

    monkeypatch.setattr(HFProcessor, "_login_into_hf", mock_get)
    # define an environment variable for the HF repo name which needs to be
    # set in any case otherwise we will get a failing test.
    storage_dir = config.output_dir
    output_file = Path(storage_dir) / Path("dummy.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text('{"key": "value"}')

    hf_processor = HFProcessor(config, storage_dir)

    # mock object for HF API is injected here:
    _ = hf_processor._login_into_hf()

    # next we test the file upload with the mock object
    result, hf_url = hf_processor.upload_responses_to_HF()
    os.unlink(output_file)
    os.rmdir(storage_dir)
    assert result is True
    assert hf_url is not None
    assert "huggingface.co" in hf_url


def test_hf_processor_with_mocked_hf_api_2(monkeypatch: MonkeyPatch) -> None:
    task_name = TaskName.ARC
    config = EvalConfig(
        output_dir=Path(__file__).parent / "eval_framework_results",
        hf_upload_dir="dummy22",
        hf_upload_repo="dummy22",
        num_fewshot=5,
        num_samples=10,
        task_name=task_name,
        llm_class=Llama31_8B_HF,
        description="dummy",
    )

    # Any arguments may be passed and mock_get() will always return our
    # mocked object, which only has the .json() method.
    def mock_get(*args: Any, **kwargs: Any) -> MockHFAPI:  # type ignore
        return MockHFAPI()

    monkeypatch.setattr(HFProcessor, "_login_into_hf", mock_get)

    storage_dir = config.output_dir
    output_file = Path(storage_dir) / Path("dummy.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text('{"key": "value"}')

    hf_processor = HFProcessor(config, storage_dir)

    # mock object for HF API is injected here:
    _ = hf_processor._login_into_hf()

    # next we test the file upload with the mock object
    result, hf_url = hf_processor.upload_responses_to_HF()
    os.unlink(output_file)
    os.rmdir(storage_dir)
    assert result is True
    assert hf_url is not None
    assert "huggingface.co" in hf_url
