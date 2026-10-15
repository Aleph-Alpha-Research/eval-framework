import datetime
import logging
from pathlib import Path

from eval_framework.run import run_with_kwargs

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    now = datetime.datetime.now()
    logger.info("starting time:", now)
    # insert API token here

    # main block for local debugging
    kwargs = {
        "context": "local",
        "judge_model_args": {},
        # ---
        "llm_args": {"model_name": "gpt-4o-mini"},
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
        "save_logs": True,
    }
    run_with_kwargs(kwargs)

    logger.info("time since start:", datetime.datetime.now() - now)
