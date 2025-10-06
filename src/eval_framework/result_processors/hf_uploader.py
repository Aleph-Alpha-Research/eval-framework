"""
Module for writing result folder and its contents to HuggingFace
"""

import logging
import os
from pathlib import Path

import wandb
from huggingface_hub import HfApi, login

from eval_framework.result_processors.base import ResultsUploader
from eval_framework.tasks.eval_config import EvalConfig

logger = logging.getLogger(__name__)


class HFUploader(ResultsUploader):
    def __init__(self, config: EvalConfig):
        if not config.hf_upload_dir:
            logger.warning("Results will not be persisted in HuggingFace (`hf_upload_dir` not configured).")
            return
        if config.output_dir is None:
            raise ValueError("Output directory is not set in the configuration.")
        if not config.hf_upload_repo:
            raise ValueError("HuggingFace upload repository is not set in the configuration.")

        self.hf_api = HFUploader._login_into_hf()
        if self.hf_api is None:
            logger.error("Could not login into HuggingFace (check HF_TOKEN). Results not persisted in HuggingFace.")

    def upload(self, llm_name: str, config: EvalConfig, output_dir: Path) -> bool:
        if getattr(self, "hf_api", None) is None:
            return False
        assert config.hf_upload_repo is not None and config.hf_upload_dir is not None

        rel_upload_dir = output_dir.relative_to(config.output_dir)
        upload_dir = Path(config.hf_upload_dir.replace("/", "")) / rel_upload_dir
        logger.info(f"HuggingFace upload starting to: {upload_dir}")

        upload_counter = 0
        for fp in output_dir.iterdir():
            if fp.name not in ["aggregated_results.json", "metadata.json"]:
                logger.info(f"Skipping {fp}; file too large.")
            else:
                try:
                    self.hf_api.upload_file(
                        path_or_fileobj=str(fp),
                        path_in_repo=str(upload_dir / fp.name),
                        repo_id=config.hf_upload_repo,
                        repo_type="dataset",
                    )
                    upload_counter += 1
                except Exception as e:
                    logger.error("Problem during HF file upload: " + str(e))
                    return False

        hf_url = f"https://huggingface.co/datasets/{config.hf_upload_repo}/tree/main/{upload_dir}"
        logger.info(f"Uploaded {upload_counter} result files to {hf_url}.")

        if wandb.run is not None:
            try:
                wandb.run.notes = f"Results uploaded to HuggingFace: [{hf_url}]({hf_url})"
            except Exception as e:
                logger.warning(f"Failed to update wandb notes with HF URL: {e}")
        return True

    @classmethod
    def _login_into_hf(cls) -> HfApi | None:
        try:
            login(token=os.environ.get("HF_TOKEN", ""))
            logger.debug("logged into HF")
            return HfApi()
        except Exception:
            return None
