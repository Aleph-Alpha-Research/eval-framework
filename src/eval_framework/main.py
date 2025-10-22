import gc
import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import wandb

from eval_framework.evaluation_generator import EvaluationGenerator, Result
from eval_framework.llm.base import BaseLLM
from eval_framework.response_generator import ResponseGenerator
from eval_framework.result_processors.hf_uploader import HFUploader
from eval_framework.result_processors.result_processor import ResultsFileProcessor, generate_output_dir
from eval_framework.result_processors.wandb_uploader import WandbUploader
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.utils.constants import RED, RESET
from eval_framework.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main(
    llm: BaseLLM,
    config: EvalConfig,
    should_preempt_callable: Callable[[], bool] | None = None,
    trial_id: int | None = None,
    *args: Any,
    resource_cleanup: bool = False,
) -> list[Result]:
    """Runs the entire evaluation process: responses generation and evaluation."""
    # Set up centralized logging early
    output_dir = generate_output_dir(llm.name, config)
    print(f"Output directory for evaluation: {output_dir}")
    setup_logging(output_dir=output_dir, log_level=logging.INFO, log_filename="evaluation.log")

    logger.info(f"{RED}[ Running full evaluation process ------- ]{RESET}")
    logger.info(f"Evaluating {llm.name} on {config.task_name}")
    logger.info(f"Configuration: num_fewshot={config.num_fewshot}, num_samples={config.num_samples}")
    logger.info(f"Output directory: {output_dir}")

    if not should_preempt_callable:
        should_preempt_callable = lambda: False  # noqa: E731
    preemption_data = None

    if trial_id:
        preemption_data = _read_preemption_data(config, trial_id)

    if preemption_data is None:
        output_dir = generate_output_dir(llm.name, config)
        wandb_run_id = config.wandb_run_id  # defaults to none, if no run_id is provided then it starts a new one
    else:
        logger.info("Found preempted run restarting ...")
        output_dir = preemption_data["output_dir"]
        wandb_run_id = preemption_data.get("wandb_run_id", None)

    logger.info(f"Output directory: {output_dir}")
    assert output_dir is not None

    file_processor = ResultsFileProcessor(output_dir)
    response_generator = ResponseGenerator(llm, config, file_processor)

    with wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        group=llm.name[:127],
        job_type=config.task_name[:63],
        id=wandb_run_id,  # (potentially resuming run after preemption)
        config=response_generator._get_metadata(),
        resume="allow",
        mode=_wandb_mode(config.wandb_project),
    ) as run:
        if hasattr(llm, "artifact"):
            wandb.use_artifact(llm.artifact)
        for additional_artifact in os.getenv("WANDB_ADDITIONAL_ARTIFACT_REFERENCES", "").split(","):
            if additional_artifact.strip():
                wandb.use_artifact(additional_artifact.strip())

        _, preempted = response_generator.generate(should_preempt_callable)

        if preempted:
            logger.info("Response generation was preempted")
            assert trial_id is not None
            run.mark_preempting()
            _save_preemption_data(config, trial_id, output_dir, wandb_run_id=run.id)
            wandb.finish(exit_code=1)
            return []
        # update config from response generator with get metadata
        if trial_id is not None:
            _delete_preemption_file(config, trial_id)

        if resource_cleanup:
            del response_generator
            gc.collect()

        evaluator = EvaluationGenerator(config, file_processor)
        results = evaluator.run_eval()

        upload_success = False
        for uploader in [HFUploader(config), WandbUploader(config)]:
            upload_success |= uploader.upload(llm.name, config, output_dir)

        if config.delete_output_dir_after_upload and upload_success:
            logger.warning(f"Deleting output directory '{output_dir}' after successful upload(s)!")
            shutil.rmtree(output_dir, ignore_errors=True)
            if output_dir.exists() and any(output_dir.iterdir()):
                logger.warning("Could not delete output directory, some files remain.")

    return results


def _read_preemption_data(config: EvalConfig, trial_id: int) -> dict[str, Any] | None:
    preemption_file = config.output_dir / f"preemption_trial_{trial_id}.json"
    if not preemption_file.is_file():
        return None
    with open(preemption_file, "rb") as f:
        preemption_data = json.load(f)
        preemption_data["output_dir"] = Path(preemption_data["output_dir"])
        preemption_data["wandb_run_id"] = preemption_data.get("wandb_run_id", "")
        logger.info(f"Loaded preemption data from {preemption_file}")
        return preemption_data


def _save_preemption_data(config: EvalConfig, trial_id: int, output_dir: Path, wandb_run_id: str = "") -> None:
    preemption_file = config.output_dir / f"preemption_trial_{trial_id}.json"
    with open(preemption_file, "w") as f:
        json.dump({"output_dir": str(output_dir), "wandb_run_id": wandb_run_id}, f)


def _delete_preemption_file(config: EvalConfig, trial_id: int) -> None:
    preemption_file = config.output_dir / f"preemption_trial_{trial_id}.json"
    if preemption_file.is_file():
        preemption_file.unlink()
        logger.info(f"Deleted preemption file: {preemption_file}")
    else:
        logger.info(f"No preemption file found to delete: {preemption_file}")
    logger.info(f"Saved preemption data to {preemption_file}")


def _wandb_mode(project: str | None) -> Literal["online", "disabled"] | None:
    """
    Checks to see if a WandB API key is found. If not, wandb starts in offline mode.
    """
    if project is None:
        logger.warning("No WandB project specified, disabling logging.")
        return "disabled"
    else:
        try:
            api_key = wandb.api.api_key
            if api_key is None:
                logger.warning(
                    """No wandb API key found. Disabling Wandb logging.
                    If you have a WandB account set the environment variable 'WANDB_API_KEY'"""
                )
                return "disabled"
            else:
                logger.info("Wandb login detected. Using online mode.")
        except Exception as e:
            logger.warning(f"Wandb login check failed: {e}. Disabling Wandb logging.")
            return "disabled"
    return "online"
