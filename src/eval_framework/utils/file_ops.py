from enum import Enum
from pathlib import Path
from typing import List, Tuple
import boto3
import wandb
import urllib
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor

class FileSystem(Enum):
    LOCAL = "local"
    S3 = "s3"

class WandbFs:
    def __init__(self):
        self.api = wandb.Api()
        self.temp_dir=None
        self._setup_s3_client()

    def _setup_s3_client(self):
        required_env_vars = ['AWS_ENDPOINT_URL', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
        for var in required_env_vars:
            if var not in os.environ:
                raise ValueError(f"Missing required environment variable: {var}")
        endpoint = os.environ['AWS_ENDPOINT_URL']
        if not endpoint.startswith(('http://', 'https://')):
            endpoint = f"http://{endpoint}"
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=endpoint,
                aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
                aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
                verify=False  # <-- disables SSL certificate validation
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create S3 client: {e}")

    @property
    def entity(self):
        return self.api.entity

    def create_file_tree(self, artifact_reference_list: List[str]) -> dict:
        tree = {"files": []}
        for file in artifact_reference_list:
            dirs = file.strip(r"s3://").split(r"/")
            c = tree
            for _, d in enumerate(dirs):
                if "." in d:
                    c["files"].append(d)
                else:
                    if d not in c:
                        c[d] = {"files": []}
                    c = c[d]
        return tree

    def get_artifact(self, artifact_id: str, version: str="latest"):
        return self.api.artifact(f"wandb-registry-model/{artifact_id}:{version}")

    def get_bucket_prefix(self, artifact: str)-> Tuple[str, str]:
        _, bucket, prefix, *_ = urllib.parse.urlparse(artifact)
        return bucket, prefix

    def ls(self, artifact: wandb.Artifact) -> list:
        file_list = list(map(lambda x: x.path_uri, [x for x in artifact.files()]))
        return file_list
        
    def file_system_detector(self, file_list: List[str]):
        if all(file.startswith("s3://") for file in file_list):
            return FileSystem.S3
        return FileSystem.LOCAL

    def download_artifacts(self, artifact: wandb.Artifact) -> Path:
        """download_artifacts downloads all artifacts associated with a registered artifact

        If the filesystem is local, then the artifact references the path of the checkpoint directory. 
        """
        file_list = self.ls(artifact)
        with ThreadPoolExecutor() as executor:
            executor.map(self._download_artifact, file_list[:4])
        return self.temp_dir.name

    def _download_artifact(self, file):
        local_path = Path(urllib.parse.urlparse(file).path)
        bucket, prefix = self.get_bucket_prefix(file)
        # create and write to tempdir
        if self.temp_dir is None:
            self.temp_dir = tempfile.TemporaryDirectory()

        local_temp_path = Path(os.path.join(self.temp_dir.name, str(local_path).strip("/")))
        local_temp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_temp_path, "wb") as f:
            self.s3_client.download_fileobj(Bucket=bucket, Key=prefix.lstrip('/'), Fileobj=f)


    def use_artifact(self, artifact):
        """
        use_artifact determines the filesystem that the artifact is on, and does one of two things:
        1. if the file system is remote (s3 type storage) then it will download the artifact to a temp diectory
           we then call wandb.use_artifact on the artifact that we downloaded.

        2. if the file system is local, then we just call wandb.use_artifact
        """
        file_system = self.file_system_detector(self.ls(artifact))
        if file_system == FileSystem.LOCAL:
            pass
        elif file_system == FileSystem.S3:
            artifact_dir = self.download_artifacts(artifact)
        return artifact_dir


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure to cleanup once done
        if self.temp_dir:
            self.temp_dir.cleanup()


    
def rec(directory, current_path):
    if len(directory):
        for direc in directory:
            rec(directory[direc], os.path.join(current_path, direc))
    else:
        os.makedirs(current_path)

if __name__ == "__main__":
    with WandbFs() as wandb_fs:
        name = "qwen-3-32b-dccp"
        version = "latest"
        artifact = wandb_fs.get_artifact(name, version)
        wandb_fs.download_artifacts(artifact)
