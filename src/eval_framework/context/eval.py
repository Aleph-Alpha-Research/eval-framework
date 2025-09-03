import importlib.util
import inspect
import sys
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from eval_framework.llm.base import BaseLLM
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.perturbation import PerturbationConfig


def import_models(models_file: Path) -> dict[str, type[BaseLLM]]:
    models_file = models_file.resolve()

    module_name = models_file.stem
    subclasses = {}

    spec = importlib.util.spec_from_file_location(module_name, str(models_file))

    if spec is None:
        raise ImportError(f"Could not load module '{models_file}'.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    if spec.loader is None:
        raise ImportError(f"Could not load module '{models_file}'.")

    spec.loader.exec_module(module)

    for name, clazz in inspect.getmembers(module, inspect.isclass):
        if issubclass(clazz, BaseLLM) and clazz is not BaseLLM:
            subclasses[name] = clazz

    return subclasses


class EvalContext(AbstractContextManager):
    def __init__(
        self,
        llm_name: str,
        models_path: Path,
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
        hf_upload_dir: str | None = None,
        hf_upload_repo: str | None = None,
        llm_args: dict[str, Any] | None = None,
        judge_models_path: Path | None = None,
        judge_model_name: str | None = None,
        judge_model_args: dict[str, Any] | None = None,
        batch_size: int | None = None,
        description: str | None = None,
        perturbation_type: str | None = None,
        perturbation_probability: float | None = None,
        perturbation_seed: int | None = None,
    ) -> None:
        self.llm_name = llm_name
        self.models_path = models_path
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
        self.hf_upload_dir = hf_upload_dir
        self.hf_upload_repo = hf_upload_repo
        self.llm_args = llm_args
        self.judge_models_path = judge_models_path
        self.judge_model_name = judge_model_name
        self.judge_model_args = judge_model_args
        self.batch_size = batch_size
        self.description = description

        if perturbation_type or perturbation_probability is not None:
            perturbation = {
                "type": perturbation_type,
                "probability": perturbation_probability,
                "seed": perturbation_seed,
            }
            self.perturbation_config: PerturbationConfig | None = PerturbationConfig(
                **{k: v for k, v in perturbation.items() if v is not None}
            )
        else:
            self.perturbation_config = None

        self.config: EvalConfig | None = None

    def should_preempt(self) -> bool:
        return False

    def get_trial_id(self) -> int | None:
        return None
