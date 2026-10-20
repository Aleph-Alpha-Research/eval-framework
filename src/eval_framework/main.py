import logging
import os
import shutil
from typing import Literal

import wandb

from eval_framework.evaluation_generator import EvaluationGenerator
from eval_framework.llm.base import BaseLLM
from eval_framework.response_generator import ResponseGenerator
from eval_framework.result_processors.base import Result
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
    verbosity: int = 1,
) -> list[Result]:
    """Runs the entire evaluation process: responses generation and evaluation."""
    # Set up centralized logging early
    output_dir = generate_output_dir(llm.name, config)
    setup_logging(output_dir=output_dir, log_level=verbosity, log_filename="evaluation.log")
    logger.info(f"Output directory for evaluation: {output_dir}")

    logger.info(f"{RED}[ Running full evaluation process ------- ]{RESET}")
    logger.info(f"Evaluating {llm.name} on {config.task_name}")
    logger.info(f"Configuration: num_fewshot={config.num_fewshot}, num_samples={config.num_samples}")
    logger.info(f"Output directory: {output_dir}")

    file_processor = ResultsFileProcessor(output_dir)
    response_generator = ResponseGenerator(llm, config, file_processor)

    with wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        group=(config.wandb_group or llm.name)[:127],
        job_type=config.task_name[:63],
        id=config.wandb_run_id,  # none starts a new run
        config=response_generator._get_metadata(),
        resume="allow",
        mode=_wandb_mode(config.wandb_project),
        settings=wandb.Settings(disable_code=True),  # ("wandb-history" artifacts not needed)
    ):
        for additional_artifact in os.getenv("WANDB_ADDITIONAL_ARTIFACT_REFERENCES", "").split(","):
            if additional_artifact.strip():
                wandb.use_artifact(additional_artifact.strip())

        response_generator.generate()

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
