from pathlib import Path
from unittest.mock import Mock

import pytest
from pytest_mock import MockerFixture

from eval_framework.llm.huggingface import Qwen3_0_6B
from eval_framework.result_processors.hf_uploader import HFUploader
from eval_framework.tasks.benchmarks.arc import ARC
from eval_framework.tasks.eval_config import EvalConfig


@pytest.fixture
def sample_output_dir(tmp_path: Path) -> Path:
    output_dir = tmp_path / "my_model" / "my_task" / "fewshot_0"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "metadata.json").write_text('{"test": "config"}')
    (output_dir / "aggregated_results.json").write_text('{"accuracy": 0.85}')
    (output_dir / "results.jsonl").write_text('[{"large": "file"}]\n')
    (output_dir / "output.jsonl").write_text('[{"large": "output"}]\n')
    (output_dir / "summary.txt").write_text("Test summary")

    return output_dir


@pytest.fixture
def sample_config(tmp_path: Path) -> EvalConfig:
    return EvalConfig(
        task_name=ARC.NAME,
        num_fewshot=0,
        num_samples=10,
        output_dir=tmp_path,
        hf_upload_repo="test-org",
        hf_upload_dir="results",
        llm_class=Qwen3_0_6B,
    )


def test_upload_success(
    mocker: MockerFixture, sample_config: EvalConfig, tmp_path: Path, sample_output_dir: Path
) -> None:
    """Test successful upload of files."""
    mocker.patch("eval_framework.result_processors.hf_uploader.login")
    mock_hf_api_class = mocker.patch("eval_framework.result_processors.hf_uploader.HfApi")
    mock_hf_api_instance = Mock()
    mock_hf_api_class.return_value = mock_hf_api_instance

    mock_wandb_run = Mock()
    mock_wandb_run.notes = ""
    mock_wandb = mocker.patch("eval_framework.result_processors.hf_uploader.wandb")
    mock_wandb.run = mock_wandb_run

    # Update config to use the temporary directory
    # sample_config.output_dir = sample_output_dir.parent

    uploader = HFUploader(sample_config)
    uploader.upload("test-model", sample_config, sample_output_dir)

    # Verify upload_file was called for the correct files (excluding large files)
    expected_calls = [
        ("aggregated_results.json", "results/my_model/my_task/fewshot_0/aggregated_results.json"),
        ("metadata.json", "results/my_model/my_task/fewshot_0/metadata.json"),
    ]

    assert mock_hf_api_instance.upload_file.call_count == 2

    for filename, expected_dest in expected_calls:
        mock_hf_api_instance.upload_file.assert_any_call(
            path_or_fileobj=str(sample_output_dir / filename),
            path_in_repo=expected_dest,
            repo_id="test-org",
            repo_type="dataset",
        )

    # Verify wandb notes were updated
    expected_url = "https://huggingface.co/datasets/test-org/tree/main/results/my_model/my_task/fewshot_0"
    assert f"Results uploaded to HuggingFace: [{expected_url}]({expected_url})" in mock_wandb_run.notes


def test_init_login_failure(mocker: MockerFixture, sample_config: EvalConfig) -> None:
    """Test initialization when HF login fails."""
    mocker.patch("eval_framework.result_processors.hf_uploader.login", side_effect=Exception("Login failed"))
    mocker.patch("eval_framework.result_processors.hf_uploader.HfApi")

    uploader = HFUploader(sample_config)
    assert uploader.hf_api is None

    # Should return early without crashing
    uploader.upload("test-model", sample_config, Path("/non-existent"))


def test_init_turned_off() -> None:
    """Test initialization when HF details are not specified."""
    sample_config = EvalConfig(
        task_name=ARC.NAME,
        num_fewshot=0,
        num_samples=10,
        llm_class=Qwen3_0_6B,
    )
    uploader = HFUploader(sample_config)
    assert uploader.hf_api is None

    # Should return early without crashing
    uploader.upload("test-model", sample_config, Path("/non-existent"))
