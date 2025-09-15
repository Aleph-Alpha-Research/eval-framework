import errno
import os
import tempfile
from collections.abc import Generator
from pathlib import Path
from unittest import mock
from unittest.mock import Mock, patch

import pytest
import wandb

from eval_framework.utils.file_ops import (
    WandbFs,
)
from tests.mock_wandb import MockArtifact


@pytest.fixture
def wandb_run(mock_wandb_artifact: Mock) -> Generator[wandb.Run, None, None]:
    with wandb.init(project="test-project") as run:
        yield run


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
    aws_env: dict[str, str], mock_s3_client: tuple[Mock, Mock]
) -> Generator[tuple[WandbFs, Mock, Mock], None, None]:
    mock_s3_client_instance, mock_boto_client = mock_s3_client
    with patch.dict(os.environ, aws_env):
        yield WandbFs(), mock_s3_client_instance, mock_boto_client


@pytest.fixture
def wandb_fs(wandb_fs_with_env: tuple[WandbFs, Mock, Mock]) -> WandbFs:
    wandb_fs_instance, _, _ = wandb_fs_with_env
    return wandb_fs_instance


class TestWandbFs:
    """
    TestWandbFs tests filesystem-like operations from the class

    - test that the entity is returned correctly
    - file trees are created correctly from a flat list of artifacts
    - bucket and prefixes are correctly extracted
    - artifacts are downloaded to a temporary directory
    - hf checkpoints are found in a file tree
    """

    def test_entity_property(self, wandb_fs: WandbFs) -> None:
        assert wandb_fs.entity == "test-entity"

    def test_download_and_use_artifact_s3(
        self,
        aws_env: dict[str, str],
        mock_s3_client: tuple[Mock, Mock],
        wandb_run: wandb.Run,
        mock_wandb: Mock,
        wandb_fs_with_env: tuple[WandbFs, Mock, Mock],
        mock_wandb_api: Mock,
    ) -> None:
        with patch.dict(os.environ, aws_env):
            wandb_fs, _, _ = wandb_fs_with_env
            # Ensure the wandb_fs uses the same mock API instance
            wandb_fs.api = mock_wandb_api

            artifact = wandb.Artifact(name="test-model", type="model")
            artifact.add_reference("s3://bucket/model/config.json")
            logged_artifact = wandb_run.log_artifact(artifact, "model")
            assert logged_artifact
            # set artifact in api for testing purposes
            mock_wandb_api.set_artifact("test-model", [x.path_uri for x in logged_artifact.files()])

            artifact = wandb_fs.get_artifact(logged_artifact.name)
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

    def test_download_artifact_out_of_disk_space(
        self, wandb_fs: WandbFs, mock_wandb_api: Mock, mock_wandb_artifact: MockArtifact
    ) -> None:
        """
        Test the download_artifact method to ensure it handles artifact downloads correctly.
        """
        wandb_fs.api = mock_wandb_api

        def fail_open(*args, **kwargs):
            raise OSError(errno.ENOSPC, "No space left on device")

        # Call the download_artifact method
        with pytest.raises(OSError, match="No space left on device"):
            with mock.patch.object(mock_wandb_artifact, "download", side_effect=fail_open):
                _ = wandb_fs.download_artifact(mock_wandb_artifact)
        assert wandb_fs.artifact_downloaded is False

    def test_download_artifact(
        self, wandb_fs: WandbFs, mock_wandb_api: Mock, mock_wandb_artifact: MockArtifact
    ) -> None:
        """
        Test the download_artifact method to ensure it handles artifact downloads correctly.
        """
        # Mock the artifact to simulate download behavior
        # Assign the mock API to the WandbFs instance
        wandb_fs.api = mock_wandb_api

        wandb_fs.download_artifact(mock_wandb_artifact)
        assert wandb_fs.artifact_downloaded is True
