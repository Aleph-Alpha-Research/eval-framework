import atexit
import os
import signal
import tempfile
import urllib
import warnings
from pathlib import Path
from types import FrameType
from typing import Any
from unittest.mock import patch

import boto3
import boto3.session
import requests
import wandb


class WandbFs:
    """
    WandbFs provides an interface to interact with Weights & Biases artifacts.

    WandB provides a unified API to access artifacts with artifact.download().

    Several issues with the standard WandB artifact handling motivated the creation of this class:
    1. Custom S3 endpoints: Users may have custom S3-compatible storage solutions.
    The standard WandB artifact handling does not natively support self-signed certificates.

    2. Artifacts may not always be in a HuggingFace-compatible format and may have extra directories.
    This class includes methods to find HuggingFace checkpoints in downloaded artifacts.

    3. Custom download paths with clean up upon failure: Rather than downloading to a
    WandB-managed cache directory, this class allows users to specify a custom download path
    (which can be a persistent directory). If no path is provided, a temporary directory is
    created and cleaned up automatically.

    Args:
        user_supplied_download_path: Optional path to download artifacts to. This acts
        as a cache, so if the artifact is already present, it will not be re-downloaded.

    example usage:
    >>> with WandbFs("./my-download-path/) as wandb_fs:
    ...    artifact = wandb_fs.get_artifact("my-artifact", version="v1")
    ...    file_list = wandb_fs.ls(artifact)
    ...    download_path = wandb_fs.download_and_use_artifact(artifact)
    ...    checkpoint_root = wandb_fs.find_hf_checkpoint_root_from_path_list(file_list)

    """

    def __init__(self, user_supplied_download_path: str | None = None):
        self.api = wandb.Api()
        self.user_supplied_download_path: Path | tempfile.TemporaryDirectory | None = (
            Path(user_supplied_download_path) if user_supplied_download_path else None
        )
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self.download_path: Path | None = None
        self._setup_s3_client()
        self._setup_cleanup_handlers()
        self.original_resource = boto3.session.Session().resource

    def _unverified_resource(self, service_name: str, *args: Any, **kwargs: Any) -> Any:
        kwargs["verify"] = False
        return self.original_resource(service_name, *args, **kwargs)

    def _setup_cleanup_handlers(self) -> None:
        """
        because wandbfs deals with downloading files, we will need to
        make sure that at exit and at failure, the directory does not persist
        """
        atexit.register(self._cleanup_temp_dir)
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum: int, frame: FrameType | None) -> None:
        self._cleanup_temp_dir()
        # we need to re-raise the signal to terminate gracefully
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _setup_s3_client(self) -> None:
        required_env_vars = ["AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"Missing required environment variable: {var}")
        endpoint = os.environ["AWS_ENDPOINT_URL"]
        if not endpoint.startswith(("http://", "https://")):
            os.environ["AWS_ENDPOINT_URL"] = f"https://{endpoint}"

    @property
    def entity(self) -> str | None:
        return self.api.default_entity

    def get_artifact(self, artifact_id: str, version: str = "latest") -> wandb.Artifact:
        return self.api.artifact(f"wandb-registry-model/{artifact_id}:{version}")

    def get_bucket_prefix(self, artifact: str) -> tuple[str, str]:
        _, bucket, prefix, *_ = urllib.parse.urlparse(artifact)
        return bucket, prefix

    def ls(self, artifact: wandb.Artifact) -> list[str]:
        """
        list all files in the artifact

        Args:
            artifact: The wandb.Artifact to list files from.
        Returns:
            list | The list of file paths in the artifact.
        """
        file_list = list(map(lambda x: x.path_uri, [x for x in artifact.files()]))
        return file_list

    def download_and_use_artifact(
        self,
        artifact: wandb.Artifact,
    ) -> Path:
        """
        use_artifact determines the filesystem that the artifact is on, and does one of two things:
        1. if the file system is remote (s3 type storage) then it will download the artifact to a temp diectory
           we then call wandb.use_artifact on the artifact that we downloaded.
        2. if this download fails, we patch boto3 to retrieve the artifact.
        """
        # create tempdir
        artifact_subdir = "/".join(artifact.name.split(":"))
        if self.user_supplied_download_path is None:
            temp_dir = tempfile.TemporaryDirectory()
            self.download_path = Path(temp_dir.name) / artifact_subdir
            self._temp_dir = temp_dir  # Keep reference to prevent pre-mature cleanup
        else:
            assert isinstance(self.user_supplied_download_path, Path)
            self.download_path = self.user_supplied_download_path / artifact_subdir
            if self.download_path.exists():
                return self.download_path

        with patch("boto3.session.Session.resource", new=self._unverified_resource):
            with warnings.catch_warnings():
                # this is to suppress the insecure request warning from urllib3
                # the attribute exists, but mypy cannot resolve it
                warnings.simplefilter(
                    "ignore",
                    category=requests.packages.urllib3.exceptions.InsecureRequestWarning,  # type: ignore
                )
                artifact_path = artifact.download(root=str(self.download_path))
        return Path(artifact_path)

    def find_hf_checkpoint_root_from_path_list(self) -> str | None:
        """Find HuggingFace checkpoint root from a list of file paths.

        Args:
            file_paths: List of file paths (can be S3 URIs or local paths)

        Returns:
            str | None: Path to the HuggingFace checkpoint root folder, or None if not found
        """
        download_path = (
            self.download_path.name
            if isinstance(self.download_path, tempfile.TemporaryDirectory)
            else self.download_path
        )

        if not download_path:
            return None

        checkpoint_roots = [x for x in Path(download_path).glob("**/config.json")]
        if checkpoint_roots:
            assert len(checkpoint_roots) == 1, (
                "Multiple checkpoints found"
            )  # if there are more than one, we have a problem
            return str(checkpoint_roots[0].parent)

        return None

    def cleanup(self) -> None:
        self._cleanup_temp_dir()

    def __enter__(self) -> "WandbFs":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self._cleanup_temp_dir()

    def _cleanup_temp_dir(self) -> None:
        if hasattr(self, "_temp_dir") and self._temp_dir:
            try:
                self._temp_dir.cleanup()
            except (OSError, FileNotFoundError):
                # Directory might already be cleaned up or removed
                pass
            finally:
                self._temp_dir = None
                self.download_path = None

    def __del__(self) -> None:
        self._cleanup_temp_dir()
