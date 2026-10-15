from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from eval_framework.tasks.eval_config import EvalConfig


def require_model_names(llm_args: dict[str, Any], judge_model_args: dict[str, Any]) -> None:
    """Fail early when `OpenAIModel`, used for both the model and the judge, would be missing its `model_name`."""
    if "model_name" not in llm_args:
        raise ValueError("llm_args must set model_name, e.g. `--llm-args model_name=gpt-4o-mini`.")
    if judge_model_args and "model_name" not in judge_model_args:
        raise ValueError("judge_model_args must set model_name, e.g. `--judge-model-args model_name=gpt-4o`.")


class EvalContext(AbstractContextManager):
    def __init__(
        self,
        num_samples: int | None = None,
        max_tokens: int | None = None,
        num_fewshot: int | None = None,
        task_name: str | None = None,
        task_subjects: list[str] | None = None,
        hf_revision: str | None = None,
        output_dir: Path | None = None,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_run_id: str | None = None,
        wandb_group: str | None = None,
        wandb_upload_results: bool | None = None,
        hf_upload_dir: str | None = None,
        hf_upload_repo: str | None = None,
        llm_args: dict[str, Any] | None = None,
        judge_model_args: dict[str, Any] | None = None,
        batch_size: int | None = None,
        description: str | None = None,
        randomize_judge_order: bool = False,
        delete_output_dir_after_upload: bool | None = None,
        repeats: int | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.max_tokens = max_tokens
        self.num_fewshot = num_fewshot
        self.task_name = task_name
        self.task_subjects = task_subjects
        self.hf_revision = hf_revision
        self.output_dir = output_dir
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_run_id = wandb_run_id
        self.wandb_group = wandb_group
        self.wandb_upload_results = wandb_upload_results
        self.hf_upload_dir = hf_upload_dir
        self.hf_upload_repo = hf_upload_repo
        self.llm_args = llm_args if llm_args is not None else {}
        self.judge_model_args = judge_model_args if judge_model_args is not None else {}
        self.batch_size = batch_size
        self.description = description
        self.randomize_judge_order = randomize_judge_order
        self.delete_output_dir_after_upload = delete_output_dir_after_upload
        self.repeats = repeats

        self.config: EvalConfig | None = None

    def should_preempt(self) -> bool:
        return False

    def get_trial_id(self) -> int | None:
        return None
