import errno
import os
import tempfile
import threading
import time
from collections.abc import Generator
from pathlib import Path
from time import sleep as _original_sleep
from typing import Any
from unittest import mock
from unittest.mock import Mock, patch

import pytest
import wandb

from eval_framework.utils.file_ops import WandbFs


@pytest.fixture
def aws_env() -> dict[str, str]:
    return {
        "AWS_ENDPOINT_URL": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
    }


@pytest.fixture
def aws_env_no_protocol() -> dict[str, str]:
    return {
        "AWS_ENDPOINT_URL": "localhost:9000",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
    }


@pytest.fixture
def mock_s3_client() -> Generator[tuple[Mock, Mock], None, None]:
    with patch("boto3.client") as mock_boto_client:
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        yield mock_s3_client, mock_boto_client


@pytest.fixture
def wandb_fs_with_env(
    aws_env: dict[str, str], mock_s3_client: tuple[Mock, Mock], mock_wandb: Mock
) -> Generator[tuple[WandbFs, Mock, Mock], None, None]:
    mock_s3_client_instance, mock_boto_client = mock_s3_client
    with patch.dict(os.environ, aws_env):
        yield WandbFs(), mock_s3_client_instance, mock_boto_client


@pytest.fixture
def wandb_fs(wandb_fs_with_env: tuple[WandbFs, Mock, Mock]) -> WandbFs:
    wandb_fs_instance, _, _ = wandb_fs_with_env
    return wandb_fs_instance


class TestWandbFs:
    def test_get_and_download_artifact_s3(self, wandb_run: wandb.Run, wandb_fs: WandbFs) -> None:
        artifact = wandb.Artifact(name="test-model", type="model")
        artifact.add_reference("s3://bucket/model/config.json")
        wandb_run.log_artifact(artifact, aliases=["latest"])

        artifact = wandb_fs.get_artifact("test-entity/project/test-model")
        assert wandb_fs.download_artifact(artifact)

    def test_find_hf_checkpoint_from_s3_paths(self, wandb_fs: WandbFs) -> None:
        # Create temporary files to simulate the directory structure
        temp_dir = tempfile.TemporaryDirectory()
        wandb_fs.download_path = Path(temp_dir.name)
        tempdir = Path(temp_dir.name)
        model_dir = tempdir / "models" / "my-model"
        model_dir.mkdir(parents=True)

        (model_dir / "config.json").touch()
        (model_dir / "tokenizer.json").touch()
        (model_dir / "model.safetensors").touch()

        other_dir = tempdir / "other"
        other_dir.mkdir()
        (other_dir / "readme.txt").touch()

        result = wandb_fs.find_hf_checkpoint_root_from_path_list()
        assert result == tempdir / "models/my-model"

        # Clean up
        temp_dir.cleanup()

    def test_find_hf_checkpoint_from_empty_dir(self, wandb_fs: WandbFs) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        wandb_fs.download_path = Path(temp_dir.name)
        tempdir = Path(temp_dir.name)
        model_dir = tempdir / "models" / "my-model"
        model_dir.mkdir(parents=True)

        other_dir = tempdir / "other"
        other_dir.mkdir()

        result = wandb_fs.find_hf_checkpoint_root_from_path_list()
        assert result is None

        # Clean up
        temp_dir.cleanup()

    def test_download_artifact_out_of_disk_space(self, wandb_fs: WandbFs) -> None:
        """
        Test the download_artifact method to ensure it handles artifact downloads correctly.
        """

        def fail_open(*args: Any, **kwargs: Any) -> None:
            raise OSError(errno.ENOSPC, "No space left on device")

        # Call the download_artifact method
        mock_wandb_artifact = wandb.Artifact("__mock_artifact__", "model")
        with pytest.raises(OSError, match="No space left on device"):
            with mock.patch.object(mock_wandb_artifact, "download", side_effect=fail_open):
                # ignore the type since we're using a mock artifact
                _ = wandb_fs.download_artifact(mock_wandb_artifact)  # type: ignore
        assert wandb_fs._artifact_downloaded is False

    def test_download_artifact(self, wandb_fs: WandbFs) -> None:
        """
        Test the download_artifact method to ensure it handles artifact downloads correctly.
        """
        # ignore the type since we're using a mock artifact
        mock_wandb_artifact = wandb.Artifact("__mock_artifact__", "model")
        wandb_fs.download_artifact(mock_wandb_artifact)  # type: ignore
        assert wandb_fs._artifact_downloaded is True

    @pytest.mark.parametrize("two_files", [True, False])
    def test_local_artifact_not_downloaded(
        self, wandb_run: wandb.Run, wandb_fs: WandbFs, tmp_path: Path, two_files: bool
    ) -> None:
        """Test that local artifacts are not downloaded and not removed by WandbFs."""
        # GIVEN a local artifact with multiple files
        tmp_path_file = tmp_path / "test_file.txt"
        tmp_path_file.write_text("This is a test file.")
        if two_files:
            (tmp_path / "subdir").mkdir()
            (tmp_path / "subdir" / "subfile.txt").touch()

        artifact = wandb.Artifact(name="test-model", type="model")
        artifact.add_reference(tmp_path_file.as_uri())
        logged_artifact = wandb_run.log_artifact(artifact, aliases=["v0-local"])

        # WHEN downloading the artifact via WandbFs
        down_path = wandb_fs.download_artifact(logged_artifact)

        # THEN the artifact is not removed by WandbFs cleanup and the download path is correct
        wandb_fs._cleanup_temp_dir()
        wandb_fs._cleanup_user_dir()
        assert down_path == tmp_path
        assert tmp_path_file.exists()
        assert not hasattr(wandb_fs, "_artifact_downloaded")

    @pytest.mark.parametrize("query_version", ["v0", "v0-local"])
    def test_get_artifact_local_preferred(self, wandb_run: wandb.Run, wandb_fs: WandbFs, query_version: str) -> None:
        """Test that local version is used if available (irrespectively what's requested)."""
        artifact = wandb.Artifact(name="test-model", type="model")
        artifact.add_reference("file://" + __file__)
        wandb_run.log_artifact(artifact, aliases=["v0-local"])

        artifact = wandb_fs.get_artifact("test-entity/project/test-model", query_version)
        assert artifact.aliases == ["v0-local"]

    @pytest.mark.parametrize("query_version", ["v0", "v0-local"])
    def test_get_artifact_nonlocal_fallback(self, wandb_run: wandb.Run, wandb_fs: WandbFs, query_version: str) -> None:
        """Test that nonlocal version is used when local is invalid (irrespectively what's requested)."""
        artifact = wandb.Artifact(name="test-model", type="model")
        artifact.add_reference("file://" + __file__)
        artifact.add_reference("/non_existent_file_path")
        wandb_run.log_artifact(artifact, aliases=["v0-local"])

        artifact = wandb.Artifact(name="test-model", type="model")
        wandb_run.log_artifact(artifact, aliases=["v0"])

        artifact = wandb_fs.get_artifact("test-entity/project/test-model", query_version)
        assert artifact.aliases == ["v0"]

    @pytest.mark.parametrize("query_version", ["v0", "v0-local"])
    def test_get_artifact_local_nonexisting(self, wandb_run: wandb.Run, wandb_fs: WandbFs, query_version: str) -> None:
        """Test that nonlocal version is used when local does not exist (irrespectively what's requested)."""
        artifact = wandb.Artifact(name="test-model", type="model")
        wandb_run.log_artifact(artifact, aliases=["v0"])

        time_start = time.time()
        artifact = wandb_fs.get_artifact("test-entity/project/test-model", query_version)
        assert artifact.aliases == ["v0"]
        assert time.time() - time_start < 2  # no waiting when local does not exist

    @pytest.mark.parametrize("query_version", ["v0", "v0-local"])
    def test_get_artifact_nonlocal_waits(
        self, wandb_run: wandb.Run, wandb_fs: WandbFs, query_version: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that nonlocal version is used and waited for when local is invalid (irrespectively what's requested)"""
        artifact = wandb.Artifact(name="test-model", type="model")
        artifact.add_reference("/non_existent_file_path")
        wandb_run.log_artifact(artifact, aliases=["v0-local"])

        # Test that waiting can timeout
        monkeypatch.setenv("WANDB_ARTIFACT_WAIT_TIMEOUT_SEC", "1")
        monkeypatch.setattr("time.sleep", lambda x: _original_sleep(0.1))  # speed busy wait

        with pytest.raises(RuntimeError, match="Timed out"):
            artifact = wandb_fs.get_artifact("test-entity/project/test-model", query_version)

        # Test that waiting can succeed
        monkeypatch.setenv("WANDB_ARTIFACT_WAIT_TIMEOUT_SEC", "5")

        def v0_thread() -> None:
            time.sleep(2)
            artifact = wandb.Artifact(name="test-model", type="model")
            wandb_run.log_artifact(artifact, aliases=["v0"])  # not fully thread-safe but does the job

        thread = threading.Thread(target=v0_thread)
        thread.start()

        artifact = wandb_fs.get_artifact("test-entity/project/test-model", query_version)
        assert artifact.aliases == ["v0"]

        thread.join()
