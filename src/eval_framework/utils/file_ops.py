import atexit
import os
import shutil
import signal
import tempfile
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
    REGISTRY_MODEL_ROOT = "wandb-registry-model"
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

    4. Cleanup is handled in two ways, and exit/signal handles are registered accordingly:
        - when not used as a context manager atexit handlers ensure cleanup on normal script termination
        - when used as a context manager, cleanup is handled in __exit__ directly

    Args:
        user_supplied_download_path: Optional path to download artifacts to. This acts
        as a cache, so if the artifact is already present, it will not be re-downloaded.

    example usage:
    >> with WandbFs("./my-download-path/") as wandb_fs:
    ..    artifact = wandb_fs.get_artifact("my-artifact", version="v1")
    ..    download_path = wandb_fs.download_artifact(artifact)
    ..    file_root = wandb_fs.find_hf_checkpoint_root_from_path_list()
    """

    def __init__(self, user_supplied_download_path: str | Path | None = None):
        self.api = wandb.Api()
        self.user_supplied_download_path = (
            Path(user_supplied_download_path) if user_supplied_download_path is not None else None
        )
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self.download_path: Path | None = None
        self._setup_s3_client()
        self._setup_cleanup_handlers()
        self.original_resource = boto3.session.Session().resource
        self._used_as_context_manager = False

    def _unverified_resource(self, service_name: str, *args: Any, **kwargs: Any) -> Any:
        kwargs["verify"] = False
        return self.original_resource(service_name, *args, **kwargs)

    def _setup_cleanup_handlers(self) -> None:
        """
        because wandbfs deals with downloading files, we will need to
        make sure that at exit and at failure, the directory does not persist
        """
        if self._used_as_context_manager:
            return  # if used as a context manager, __exit__ handles cleanup

        # we only want to register the appropriate cleanup function
        if self.user_supplied_download_path:
            atexit.register(self._cleanup_user_dir)
        else:
            atexit.register(self._cleanup_temp_dir)

    def _clean_on_signal(self, signum: int, frame: FrameType | None) -> None:
        if self.user_supplied_download_path:
            self._cleanup_user_dir()
        else:
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
        return self.api.artifact(f"{self.REGISTRY_MODEL_ROOT}/{artifact_id}:{version}")

    def download_artifact(
        self,
        artifact: wandb.Artifact,
    ) -> Path:
        """
        download_artifact downloads the specified artifact to either a user-specified
        directory or a temporary directory. If the user-specified directory already
        contains the artifact, it will not be re-downloaded.

        Args:
            artifact: The WandB artifact object to download.
        Returns:
            Path: The path to the downloaded artifact.
        """
        # create the base path for either a temp or user dir
        if self.user_supplied_download_path is None:
            self._temp_dir = tempfile.TemporaryDirectory()
            base_path = Path(self._temp_dir.name)
        else:
            base_path = self.user_supplied_download_path

        artifact_subdir = "/".join(artifact.name.split(":"))
        self.download_path = base_path / artifact_subdir
        if self.user_supplied_download_path and self.download_path.exists():
            return self.download_path

        with patch("boto3.session.Session.resource", new=self._unverified_resource):
            with warnings.catch_warnings():
                # this is to suppress the insecure request warning from urllib3
                # the attribute exists, but mypy cannot resolve it
                warnings.simplefilter(
                    "ignore",
                    category=requests.packages.urllib3.exceptions.InsecureRequestWarning,  # type: ignore
                )
                print(f"Downloading artifact to {self.download_path}")
                # Since the cache lives inside the docker container, it is unused in future
                # runs. Skipping the cache also avoids file duplication and extra copying.
                self._artifact_downloaded = False
                artifact_path = artifact.download(root=str(self.download_path), skip_cache=True)
                self._artifact_downloaded = True

        return Path(artifact_path)

    def find_hf_checkpoint_root_from_path_list(self) -> Path | None:
        """Find HuggingFace checkpoint root from a list of file paths.

        Args:
            file_paths: List of file paths (can be S3 URIs or local paths)

        Returns:
            str | None: Path to the HuggingFace checkpoint root folder, or None if not found
        """

        # self.download_path can be None if download_artifact was never called
        if self.download_path and self.download_path.exists():
            checkpoint_roots = [x for x in Path(self.download_path).glob("**/config.json")]
            if checkpoint_roots:
                assert len(checkpoint_roots) == 1, (
                    "Multiple checkpoints found"
                )  # if there are more than one, we have a problem
                return checkpoint_roots[0].parent

        return None

    def __enter__(self) -> "WandbFs":
        self._used_as_context_manager = True
        # Store original handlers for restoration
        self._original_sigterm = signal.signal(signal.SIGTERM, self._clean_on_signal)
        self._original_sigint = signal.signal(signal.SIGINT, self._clean_on_signal)

        # on interrupt or termination
        signal.signal(signal.SIGTERM, self._clean_on_signal)
        signal.signal(signal.SIGINT, self._clean_on_signal)

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        exit the context manager, cleaning up the temporary directory if it was used
        or the user directory if it was specified and failed to download.
        """
        if self.user_supplied_download_path:
            self._cleanup_user_dir()
        else:
            self._cleanup_temp_dir()

        signal.signal(signal.SIGTERM, self._original_sigterm)
        signal.signal(signal.SIGINT, self._original_sigint)

    def _cleanup_user_dir(self) -> None:
        """
        _cleanup_user_dir will remove the contents of the user specified
        download path if there was an attempt to download the artifact and it failed.
        """
        if (
            # check to make sure this flag was set; if not, there was no attempt to download.
            hasattr(self, "_artifact_downloaded")
            and not self._artifact_downloaded  # check that the artifact was not downloaded
            and self.user_supplied_download_path
            and self.download_path
            and self.download_path.exists()
        ):
            # remove the contents of the download path.
            print(f"Cleaning up user-specified download path...{self.download_path}")
            shutil.rmtree(self.download_path)

    def _cleanup_temp_dir(self) -> None:
        if hasattr(self, "_temp_dir"):
            try:
                self._temp_dir.cleanup()
            except (OSError, FileNotFoundError):
                # Directory might already be cleaned up or removed
                pass
            finally:
                self._temp_dir = None
                self.download_path = None
