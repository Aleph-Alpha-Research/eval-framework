import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import wandb

from eval_framework.evaluation_generator import EvaluationGenerator, Result
from eval_framework.llm.base import BaseLLM
from eval_framework.response_generator import ResponseGenerator
from eval_framework.result_processors.hf_processor import HFProcessor
from eval_framework.result_processors.result_processor import ResultsFileProcessor, generate_output_dir
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.utils.constants import RED, RESET
from eval_framework.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def main(
    llm: BaseLLM,
    config: EvalConfig,
    should_preempt_callable: Callable[[], bool] | None = None,
    trial_id: int | None = None,
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
        # config.wandb_run_id  defaults to none, if no run_id is provided then it starts a new one
        wandb_run_id = config.wandb_run_id
    else:
        logger.info("Found preempted run restarting ...")
        output_dir = preemption_data["output_dir"]
        wandb_run_id = preemption_data.get("wandb_run_id", None)

    logger.info(f"Output directory: {output_dir}")
    assert output_dir is not None

    file_processor = ResultsFileProcessor(output_dir)
    response_generator = ResponseGenerator(llm, config, file_processor)
    # take care of init after preemption handling. If we have a run
    # id from preemption, then we resume the original wandb run
    with wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        group=llm.name,
        job_type=config.task_name,
        id=wandb_run_id,
        config=response_generator._get_metadata(),
        resume="allow",
        mode=_wandb_mode(config.wandb_project),
    ) as run:
        # check to see if this is a registered model. If so, use the artifact.
        # this is placed here in the case that an artifact is used to generate completions,
        # crashes during the evaluation step, and subsequent reruns use the same generations,
        # the runs are still linked to the artifact
        if hasattr(llm, "_model") and hasattr(llm._model, "artifact"):  # BaseLLM doesn't have _model attribute
            wandb.use_artifact(llm._model.artifact)

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

        evaluator = EvaluationGenerator(config, file_processor)
        results = evaluator.run_eval()

        if config.hf_upload_dir:
            hf_processor = HFProcessor(config, output_dir)
            status, hf_url = hf_processor.upload_responses_to_HF()
            if not status:
                status_message = "*** Warning: Result upload to HF failed ***"
            else:
                status_message = "Successfully uploaded results to HuggingFace"
                if hf_url and run:
                    try:
                        run.notes = f"Results uploaded to HuggingFace: [{hf_url}]({hf_url})"
                    except Exception as e:
                        logger.warning(f"Failed to update wandb notes with HF URL: {e}")
        else:
            status_message = f"{RED}[ Results not persisted in a HuggingFace repo ------- ]{RESET}"

        logger.info(status_message)

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


def _configure_logging(output_dir: Path) -> None:
    """Configure logging to save logs to a file in the output directory."""

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up log file path
    log_file = output_dir / "evaluation.log"

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)

    # Get the root logger and add the file handler
    root_logger = logging.getLogger()

    # Remove existing file handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)

    root_logger.addHandler(file_handler)

    # Set logging level if not already set
    if root_logger.level == logging.NOTSET:
        root_logger.setLevel(logging.INFO)


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
                logger.info("wandb login detected. Using online mode.")
        except Exception as e:
            logger.warning(f"wandb login check failed: {e}. Disabling Wandb logging.")
            return "disabled"
    return "online"
