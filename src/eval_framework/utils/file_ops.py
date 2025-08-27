from pathlib import Path
from typing import Tuple
import boto3
import wandb
import urllib
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor


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

    def get_artifact(self, artifact_id: str, version: str="latest"):
        return self.api.artifact(f"wandb-registry-model/{artifact_id}:{version}")

    def get_bucket_prefix(self, artifact: str)-> Tuple[str, str]:
        _, bucket, prefix, *_ = urllib.parse.urlparse(artifact)
        return bucket, prefix

    def ls(self, artifact: wandb.Artifact) -> list:
        file_list = list(map(lambda x: x.path_uri, [x for x in artifact.files()]))
        return file_list
        

    def download_artifacts(self, artifact: wandb.Artifact):
        file_list = self.ls(artifact)
        with ThreadPoolExecutor() as executor:
            executor.map(self._download_artifact, file_list[:4])

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure to cleanup once done
        if self.temp_dir:
            self.temp_dir.cleanup()

def create_file_tree(artifact_reference_list):
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
    
def rec(directory, current_path):
    if len(directory):
        for direc in directory:
            rec(directory[direc], os.path.join(current_path, direc))
    else:
        os.makedirs(current_path)

if __name__ == "__main__":
    with WandbFs() as wandb_fs:
        artifact = wandb_fs.get_artifact("qwen-3-32b-dccp", "latest")
        wandb_fs.download_artifacts(artifact)
