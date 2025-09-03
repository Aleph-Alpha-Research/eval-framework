import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import wandb

from eval_framework.utils.file_ops import (
    WandbFs,
)


@pytest.fixture
def wandb_run(mock_wandb_artifact):
    with wandb.init(project="test-project") as run:
        yield run


@pytest.fixture
def aws_env():
    return {
        "AWS_ENDPOINT_URL": "http://localhost:9000",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
    }


@pytest.fixture
def aws_env_no_protocol():
    return {
        "AWS_ENDPOINT_URL": "localhost:9000",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
    }


@pytest.fixture
def mock_s3_client():
    with patch("boto3.client") as mock_boto_client:
        mock_s3_client = Mock()
        mock_boto_client.return_value = mock_s3_client
        yield mock_s3_client, mock_boto_client


@pytest.fixture
def wandb_fs_with_env(aws_env, mock_s3_client):
    mock_s3_client_instance, mock_boto_client = mock_s3_client
    with patch.dict(os.environ, aws_env):
        yield WandbFs(), mock_s3_client_instance, mock_boto_client


@pytest.fixture
def wandb_fs(wandb_fs_with_env):
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

    def test_entity_property(self, wandb_fs):
        assert wandb_fs.entity == "test-entity"

    def test_get_bucket_prefix(self, wandb_fs):
        bucket, prefix = wandb_fs.get_bucket_prefix("s3://my-bucket/path/to/file.json")

        assert bucket == "my-bucket"
        assert prefix == "/path/to/file.json"

    def test_ls(self, wandb_fs):
        # Set up artifact with specific files
        wandb_fs.api.set_artifact("test-model", ["s3://bucket/model/config.json", "s3://bucket/model/tokenizer.json"])
        artifact = wandb_fs.get_artifact("test-model")

        file_list = wandb_fs.ls(artifact)

        assert file_list == ["s3://bucket/model/config.json", "s3://bucket/model/tokenizer.json"]

    def test_download_artifacts(self, wandb_fs_with_env):
        wandb_fs, mock_s3_client_instance, _ = wandb_fs_with_env

        wandb_fs.api.set_artifact(
            "test-model",
            [
                "s3://bucket/model/huggingface/config.json",
                "s3://bucket/model/huggingface/tokenizer.json",
                "s3://bucket/model/huggingface/model.safetensors",
                "s3://bucket/model/other.txt",
            ],
        )
        artifact = wandb_fs.get_artifact("test-model")

        result = wandb_fs.download_artifacts(artifact)

        assert wandb_fs.download_path is not None
        assert result == wandb_fs.download_path.name
        # Should only download first 4 files
        assert mock_s3_client_instance.download_fileobj.call_count == 4

    def test_download_and_use_artifact_s3(self, aws_env, mock_s3_client, wandb_run, mock_wandb, wandb_fs_with_env):
        with patch.dict(os.environ, aws_env):
            wandb_fs, _, _ = wandb_fs_with_env
            artifact = wandb.Artifact(name="test-model", type="model")
            artifact.add_reference("s3://bucket/model/config.json")
            logged_artifact = wandb_run.log_artifact(artifact, "model")
            assert logged_artifact
            # set artifact in api for testing purposes
            wandb_fs.api.set_artifact("test-model", [x.path_uri for x in logged_artifact.files()])

            artifact = wandb_fs.get_artifact(logged_artifact.name)
            assert wandb_fs.download_and_use_artifact(artifact)

    def test_find_hf_checkpoint_from_s3_paths(self, wandb_fs):
        # Create temporary files to simulate the directory structure
        wandb_fs.download_path = tempfile.TemporaryDirectory()
        tempdir = Path(wandb_fs.download_path.name)
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

    def test_find_hf_checkpoint_from_empty_dir(self, wandb_fs):
        wandb_fs.download_path = tempfile.TemporaryDirectory()
        tempdir = Path(wandb_fs.download_path.name)
        model_dir = tempdir / "models" / "my-model"
        model_dir.mkdir(parents=True)

        other_dir = tempdir / "other"
        other_dir.mkdir()

        result = wandb_fs.find_hf_checkpoint_root_from_path_list()
        assert result is None
