"""
Module for writing result folder and its contents to HuggingFace
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from tqdm import tqdm

from eval_framework.constants import RED, RESET
from eval_framework.tasks.eval_config import EvalConfig

load_dotenv()

logger = logging.getLogger(__name__)


class HFProcessor:
    def __init__(self, config: EvalConfig, current_dir: Path) -> None:
        self.output_dir = config.output_dir
        self.current_dir = current_dir
        self.hf_upload_dir = config.hf_upload_dir
        self.hf_upload_repo = config.hf_upload_repo
        assert self.output_dir is not None
        assert self.current_dir is not None
        assert self.hf_upload_dir is not None
        self.hf_upload_dir = self.hf_upload_dir.replace("/", "")
        self.hf_api = HFProcessor._login_into_hf()

    @classmethod
    def _login_into_hf(cls) -> HfApi | None:
        try:
            login(token=os.environ.get("HF_TOKEN", ""))
            logger.info("logged into HF")
            return HfApi()

        except Exception:
            logger.info("Could not login into HuggingFace. Check credentials")
            return None

    def upload_responses_to_HF(self) -> bool:
        hf_repo_name = self.hf_upload_repo
        assert hf_repo_name is not None, "No HF upload repository configured (hf_upload_repo)!"

        if self.hf_api is None:
            logger.info("Not logged into HuggingFace")
            return False

        try:
            self.upload_dir = Path(self.current_dir).relative_to(Path(self.output_dir))
            self.upload_dir = Path(str(self.hf_upload_dir)) / self.upload_dir  # type ignore
            logger.info(f"{RED}[ HF upload to {self.upload_dir} ------- ]{RESET}")

        except Exception as e:
            logger.info(f"Upload path not properly defined: {e}")
            return False

        upload_counter = 0
        for filename in tqdm(os.listdir(self.current_dir)):
            if filename not in ["results.jsonl", "output.jsonl"]:
                upload_counter += 1
                source_filename = str(Path(self.current_dir) / filename)
                dest_filename = str(Path(self.upload_dir) / filename)
            else:
                logger.info(f"Skipping {filename}; file too large")

            try:
                self.hf_api.upload_file(
                    path_or_fileobj=source_filename,
                    path_in_repo=dest_filename,
                    repo_id=hf_repo_name,
                    repo_type="dataset",
                )
            except Exception as e:
                self.status = "Problem during HF file upload: " + str(e)
                logger.info(self.status)
                break

        logger.info(f"uploaded {upload_counter} files")
        return True
