import gzip
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import wandb

from eval_framework.llm.huggingface import Qwen3_0_6B
from eval_framework.result_processors.wandb_uploader import WandbUploader, register_artifact_upload_function
from eval_framework.tasks.benchmarks.arc import ARC
from eval_framework.tasks.eval_config import EvalConfig
from tests.mock_wandb import MockArtifact


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
        wandb_upload_results=True,
        hf_upload_dir="results",
        llm_class=Qwen3_0_6B,
    )


@pytest.fixture
def custom_upload_function() -> Generator[None, None, None]:
    def custom_artifact_upload_function(artifact_name: str, subpath: str, file_paths: list[Path]) -> str:
        return subpath

    register_artifact_upload_function(custom_artifact_upload_function)
    yield
    register_artifact_upload_function(None)


def get_logged_artifact(mock_wandb: Mock) -> list[str]:
    assert len(mock_wandb.run._logged_artifacts) == 1
    return sorted([f.path_uri for f in mock_wandb.run._logged_artifacts[0]["artifact"].files()])


def test_upload_only_json(mock_wandb: Mock, sample_config: EvalConfig, sample_output_dir: Path) -> None:
    """Test upload of only JSON files."""
    with wandb.init():
        uploader = WandbUploader(sample_config, include_all=False, compress_non_json=False)
        uploader.upload("test-model", sample_config, sample_output_dir)

        uploaded = get_logged_artifact(mock_wandb)
        assert uploaded == ["aggregated_results.json", "metadata.json"]


def test_upload_all_files(mock_wandb: Mock, sample_config: EvalConfig, sample_output_dir: Path) -> None:
    """Test upload of all files without compression."""
    with wandb.init():
        uploader = WandbUploader(sample_config, include_all=True, compress_non_json=False)
        uploader.upload("test-model", sample_config, sample_output_dir)

        uploaded = get_logged_artifact(mock_wandb)
        assert uploaded == ["aggregated_results.json", "metadata.json", "output.jsonl", "results.jsonl", "summary.txt"]


def test_upload_all_files_gz(
    mock_wandb: Mock, monkeypatch: pytest.MonkeyPatch, sample_config: EvalConfig, sample_output_dir: Path
) -> None:
    """Test upload of all files with compression of non-JSON files."""

    # Custom MockArtifact to verify gzipped files
    class CapturingMockArtifact(MockArtifact):
        def add_file(self, local_path: str, name: str | None = None, **kwargs: Any) -> None:  # type: ignore[override]
            super().add_file(local_path, name, **kwargs)
            if local_path.endswith(".gz"):
                with gzip.open(local_path, "rb") as f:
                    assert len(f.read()) > 1

    monkeypatch.setattr("wandb.Artifact", CapturingMockArtifact)

    with wandb.init():
        uploader = WandbUploader(sample_config, include_all=True, compress_non_json=True)
        uploader.upload("test-model", sample_config, sample_output_dir)

        uploaded = get_logged_artifact(mock_wandb)
        assert uploaded == [
            "aggregated_results.json",
            "metadata.json",
            "output.jsonl.gz",
            "results.jsonl.gz",
            "summary.txt.gz",
        ]
        assert len(list(sample_output_dir.glob("*.gz"))) == 0  # original files are not left in the directory


def test_init_missing(mock_wandb: Mock, sample_config: EvalConfig) -> None:
    """Test initialization when wandb is not initialized."""
    uploader = WandbUploader(sample_config, include_all=False, compress_non_json=False)
    assert not uploader.upload("test-model", sample_config, Path("/non-existent"))


def test_init_disabled(mock_wandb: Mock, sample_config: EvalConfig) -> None:
    """Test initialization when wandb is disabled."""
    with wandb.init(mode="disabled"):
        uploader = WandbUploader(sample_config, include_all=False, compress_non_json=False)
        assert not uploader.upload("test-model", sample_config, Path("/non-existent"))


def test_turned_off(mock_wandb: Mock) -> None:
    """Test initialization when the upload is disabled in config."""
    sample_config = EvalConfig(
        task_name=ARC.NAME,
        num_fewshot=0,
        num_samples=10,
        llm_class=Qwen3_0_6B,
        wandb_upload_results=False,
    )
    with wandb.init():
        uploader = WandbUploader(sample_config, include_all=False, compress_non_json=False)
        assert not uploader.upload("test-model", sample_config, Path("/non-existent"))


def test_custom_upload_function(
    mock_wandb: Mock, sample_config: EvalConfig, sample_output_dir: Path, custom_upload_function: None
) -> None:
    """Test that a custom artifact upload function is used."""
    with wandb.init():
        uploader = WandbUploader(sample_config, include_all=False, compress_non_json=False)
        uploader.upload("test-model", sample_config, sample_output_dir)

        uploaded = get_logged_artifact(mock_wandb)
        assert uploaded == ["my_model/my_task/fewshot_0"]


def test_name_and_alias(mock_wandb: Mock, sample_config: EvalConfig, sample_output_dir: Path) -> None:
    """Test that artifact name and alias are set correctly."""
    with wandb.init():
        uploader = WandbUploader(sample_config, include_all=False, compress_non_json=False)
        uploader.upload("test-model", sample_config, sample_output_dir)

        assert len(mock_wandb.run._logged_artifacts) == 1
        artifact = mock_wandb.run._logged_artifacts[0]["artifact"]
        aliases = mock_wandb.run._logged_artifacts[0]["aliases"]
        assert artifact.name == "test-model__ARC__fewshot_0__samples_10__1e5d9"
        assert "H-03b43ebe7c" in aliases
