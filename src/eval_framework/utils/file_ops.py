import atexit
import os
import signal
import tempfile
import urllib
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

import boto3
import wandb
from tqdm import tqdm
from wandb.sdk.lib.paths import StrPath


class FileSystem(Enum):
    LOCAL = "local"
    S3 = "s3"


class WandbFs:
    def __init__(self):
        self.api = wandb.Api()
        self.temp_dir = None
        self._setup_s3_client()
        self._setup_cleanup_handlers()

    def _setup_cleanup_handlers(self):
        """
        because wandbfs deals with downloading files, we will need to
        make sure that at exit and at failure, the directory does not persist
        """
        atexit.register(self._cleanup_temp_dir)
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame):
        self._cleanup_temp_dir()
        # we need to re-raise the signal to terminate gracefully
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _setup_s3_client(self):
        required_env_vars = ["AWS_ENDPOINT_URL", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"Missing required environment variable: {var}")
        endpoint = os.environ["AWS_ENDPOINT_URL"]
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"http://{endpoint}"
        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=endpoint,
                aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
                aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                verify=False,  # <-- disables SSL certificate validation
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create S3 client: {e}")

    @property
    def entity(self):
        return self.api.entity

    def get_artifact(self, artifact_id: str, version: str = "latest"):
        return self.api.artifact(f"wandb-registry-model/{artifact_id}:{version}")

    def get_bucket_prefix(self, artifact: str) -> tuple[str, str]:
        _, bucket, prefix, *_ = urllib.parse.urlparse(artifact)
        return bucket, prefix

    def ls(self, artifact: wandb.Artifact) -> list:
        """
        list all files in the artifact

        Args:
            artifact: The wandb.Artifact to list files from.
        Returns:
            list | The list of file paths in the artifact.
        """
        file_list = list(map(lambda x: x.path_uri, [x for x in artifact.files()]))
        return file_list

    def file_system_detector(self, file_list: list[str]):
        if all(file.startswith("s3://") for file in file_list):
            return FileSystem.S3
        return FileSystem.LOCAL

    def download_artifacts(self, artifact: wandb.Artifact) -> Path:
        """download_artifacts downloads all artifacts associated with a registered artifact

        If the filesystem is local, then the artifact references the path of the checkpoint directory.

        Args:
            artifact: The wandb.Artifact to download.
        Returns:
            str | The path to the downloaded artifact.
        """
        # create tempdir
        if self.temp_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory()

        file_list = self.ls(artifact)

        # Use tqdm for progress bar
        with ThreadPoolExecutor() as executor:
            with tqdm(total=len(file_list), desc="Downloading artifacts", unit="file") as pbar:

                def download_with_progress(file):
                    result = self._download_artifact(file)
                    pbar.update(1)
                    return result

                list(executor.map(download_with_progress, file_list))
        return self.temp_dir.name

    def _download_artifact(self, file: str) -> None:
        """
        downloads the provided file from a wandb artifact

        Args:
            str | file: The file to download.
        Returns:
            None
        """
        local_path = Path(urllib.parse.urlparse(file).path)
        bucket, prefix = self.get_bucket_prefix(file)
        local_temp_path = Path(os.path.join(self.temp_dir.name, str(local_path).strip("/")))
        local_temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_temp_path, "wb") as f:
            self.s3_client.download_fileobj(Bucket=bucket, Key=prefix.lstrip("/"), Fileobj=f)

    def download_and_use_artifact(
        self,
        artifact: wandb.Artifact,
        root: StrPath | None = None,
        allow_missing_references: bool = False,
        skip_cache: bool | None = None,
        path_prefix: StrPath | None = None,
        multipart: bool | None = None,
    ):
        """
        use_artifact determines the filesystem that the artifact is on, and does one of two things:
        1. if the file system is remote (s3 type storage) then it will download the artifact to a temp diectory
           we then call wandb.use_artifact on the artifact that we downloaded.

        2. if the file system is local, then we just call wandb.use_artifact
        """
        # TODO get the artifact without using first, download, and then upon success, use artifact
        artifact = wandb.use_artifact(artifact)

        file_system = self.file_system_detector(self.ls(artifact))
        if file_system == FileSystem.S3:
            try:
                artifact_path = artifact.download(
                    root=root,
                    allow_missing_references=allow_missing_references,
                    skip_cache=skip_cache,
                    path_prefix=path_prefix,
                    multipart=multipart,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download artifact: {e}")
            artifact_path = self.download_artifacts(artifact)
        else:
            artifact_path = artifact.download(
                root=root,
                allow_missing_references=allow_missing_references,
                skip_cache=skip_cache,
                path_prefix=path_prefix,
                multipart=multipart,
            )

        return artifact_path

    def find_hf_checkpoint_root_from_path_list(self, file_paths: list[str]) -> str | None:
        """Find HuggingFace checkpoint root from a list of file paths.

        Args:
            file_paths: List of file paths (can be S3 URIs or local paths)

        Returns:
            str | None: Path to the HuggingFace checkpoint root folder, or None if not found
        """
        if not file_paths:
            return None

        checkpoint_roots = [x for x in Path(self.temp_dir.name).glob("**/config.json")]
        assert len(checkpoint_roots) == 1, "Multiple checkpoints found"  # if there are more than one, we have a problem
        return checkpoint_roots[0].parent

    def cleanup(self):
        self._cleanup_temp_dir()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._cleanup_temp_dir()

    def _cleanup_temp_dir(self):
        if self.temp_dir:
            try:
                self.temp_dir.cleanup()
            except (OSError, FileNotFoundError):
                # Directory might already be cleaned up or removed
                pass
            finally:
                self.temp_dir = None

    def __del__(self):
        self._cleanup_temp_dir()
