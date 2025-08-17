import datetime
from pathlib import Path

from eval_framework.run import run_with_kwargs
from eval_framework.utils.logging_config import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info(Path("models.py"))
    now = datetime.datetime.now()
    logger.info("starting time:", now)
    # insert API token here

    # main block for local debugging
    kwargs = {
        "context": "local",
        "models": Path("src/eval_framework/llm/models.py"),
        "judge_models": None,
        "judge_model_name": None,
        "judge_model_args": {},
        # ---
        "llm_name": "Llama31_8B_API",
        "llm_args": {},
        "num_samples": 1,
        "max_tokens": None,
        "num_fewshot": 4,
        "task_name": "Math",  # complete task
        "task_subjects": None,
        "hf_revision": None,
        "output_dir": Path("outputs"),
        "hf_upload_dir": "",
        "description": "",
        "batch_size": 1,
        "perturbation_type": None,
        "perturbation_probability": None,
        "perturbation_seed": None,
        "save_logs": True,
    }
    run_with_kwargs(kwargs)

    logger.info("time since start:", datetime.datetime.now() - now)
