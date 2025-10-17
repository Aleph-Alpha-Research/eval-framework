"""
Module for writing result folder to a W&B artifact
"""

import hashlib
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

import wandb

from eval_framework.result_processors.base import ResultsUploader
from eval_framework.tasks.eval_config import EvalConfig

logger = logging.getLogger(__name__)

ArtifactUploadFunction = Callable[[str, Path, list[Path]], str | None]  # returns reference path for W&B or None
_ARTIFACT_UPLOAD_FUNCTION: ArtifactUploadFunction | None = None


def register_artifact_upload_function(func: ArtifactUploadFunction | None) -> None:
    global _ARTIFACT_UPLOAD_FUNCTION
    _ARTIFACT_UPLOAD_FUNCTION = func


def artifact_upload_function(artifact_name: str, root: Path, file_paths: list[Path]) -> str | None:
    if _ARTIFACT_UPLOAD_FUNCTION is None:
        return None
    logger.info(f"Uploading '{artifact_name}' from {root}.")
    reference_path = _ARTIFACT_UPLOAD_FUNCTION(artifact_name, root, file_paths)
    if reference_path is None:
        logger.warning("Failed uploading, the custom upload function returned empty destination path!")
    else:
        logger.info(f"Successfully uploaded to {reference_path}.")
    return reference_path


class WandbUploader(ResultsUploader):
    def __init__(
        self,
        config: EvalConfig,
        include_all: bool = True,
        compress_non_json: bool = True,
        wandb_registry: str | None = None,
    ) -> None:
        if not config.wandb_upload_results:
            logger.warning("Results will not be persisted in WandB (`wandb_upload_results` not set).")
            return
        if config.output_dir is None:
            raise ValueError("Output directory is not set in the configuration.")
        if wandb.run is None or wandb.run.settings._noop:
            logger.warning("Results will not be persisted in WandB (no WandB run active / `wandb_project` not set).")
            return

        self._include_all = include_all
        self._compress_non_json = compress_non_json
        self._wandb_registry = wandb_registry

    def upload(self, llm_name: str, config: EvalConfig, output_dir: Path) -> bool:
        if hasattr(self, "_wandb_registry") is False:
            return False  # not initialized

        try:
            if self._include_all and self._compress_non_json:
                # note: individual gz files can be easily read by `less` or `grepz`, unlike a tar of multiple files
                subprocess.run(
                    ["find", output_dir, "-type", "f", "!", "-name", "*.json", "-exec", "gzip", "-k", "{}", ";"],
                    check=True,
                )
                file_paths = list(output_dir.glob("*.json")) + list(output_dir.glob("*.gz"))
            elif self._include_all:
                file_paths = list(output_dir.glob("*"))
            else:
                file_paths = list(output_dir.glob("*.json"))

            artifact_name = self._get_artifact_name(llm_name, config)
            alias_name = self._get_alias(output_dir)

            try:
                reference_path = artifact_upload_function(artifact_name, config.output_dir, file_paths)
            except Exception as e:
                logger.error(f"Problem during artifact upload function, aborting registration: {e}.")
                return False

            artifact = wandb.Artifact(name=artifact_name, type="eval")  # note: metadata is added from run automatically
            if reference_path:
                artifact.add_reference(reference_path, checksum=True)
            else:
                logger.info("Uploading results directly to WandB.")
                for fp in file_paths:
                    artifact.add_file(str(fp))

            # Because metadata and e.g. logs are added to the artifact, we get a new version everytime!
            # To mitigate this, we also add an alias based on the content hash of the "pure" result files.
            wandb.log_artifact(artifact, aliases=[alias_name])
            if self._wandb_registry:
                artifact.link(f"wandb-registry-{self._wandb_registry}/{artifact_name}")
            logger.info(f"Successfully registered '{artifact_name}' in WandB")

        finally:
            for fp in file_paths:
                if fp.suffix == ".gz" and self._compress_non_json:
                    fp.unlink(missing_ok=True)

        return True

    def _get_artifact_name(self, llm_name: str, config: EvalConfig) -> str:
        llm_name = llm_name.replace("/", "_")  # assuming this has class name and checkpoint name in it

        # As in generate_output_dir() for consistency
        # But we don't include the eval_framework version and timestamp here (-> handled via W&B versioning)
        fewshot_str = f"fewshot_{config.num_fewshot}" if config.num_fewshot is not None else "fewshot_None"
        samples_str = f"__samples_{config.num_samples}" if config.num_samples is not None else "__samples_None"
        tokens_str = f"__tokens_{config.max_tokens}" if config.max_tokens is not None else ""
        params_str = f"{fewshot_str}{samples_str}{tokens_str}"

        config_json = config.model_json_robust_subset_dump()
        config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:5]

        return f"{llm_name}__{config.task_name}__{params_str}__{config_hash}"

    def _get_alias(self, output_dir: Path) -> str:
        digests = []
        # These files don't contain result-irrelevant things such as timestamps or paths and are fields are ordered.
        # This makes them good for generating a hash that identifies the actual results and not "random" metadata.
        for filename in ["aggregated_results.json", "output.jsonl", "results.jsonl"]:
            if (output_dir / filename).exists():
                with open(output_dir / filename, "rb") as f:
                    digests.append(hashlib.file_digest(f, "sha256").hexdigest())
        hash = hashlib.sha256("".join(digests).encode("utf-8")).hexdigest()
        return f"H-{hash[:10]}"
