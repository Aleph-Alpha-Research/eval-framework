import importlib
from os import PathLike
from typing import Any

from eval_framework.context.eval import EvalContext, import_models
from eval_framework.llm.base import BaseLLM
from eval_framework.tasks.eval_config import EvalConfig


def _load_model(llm_name: str, models_path: str | PathLike | None, *, info: str = "") -> type[BaseLLM]:
    """Load a model class either from a models file or as a fully qualified module path.

    Args:
        llm_name: The name of the model class to load, or a fully qualified module path.
        models_path: The path to a Python file containing model class definitions
        info: Additional info to include in error messages.
    Returns:
        The model class.
    """
    if models_path is None or "." in llm_name:
        # The llm_name must a a fully qualified module path
        if "." not in llm_name:
            raise ValueError(f"LLM {info} '{llm_name}' is not a fully qualified module path.")
        module_path, llm_class_name = llm_name.rsplit(".", 1)
        module = importlib.import_module(module_path)
        if not hasattr(module, llm_class_name):
            raise ValueError(f"LLM '{llm_class_name}' not found in module '{module_path}'.")
        return getattr(module, llm_class_name)
    else:
        models_dict = import_models(models_path)
        if llm_name not in models_dict:
            if info:
                info = f"{info.strip()} "
            raise ValueError(f"LLM {info} '{llm_name}' not found in {models_path}.")
        return models_dict[llm_name]


class LocalContext(EvalContext):
    def __enter__(self) -> "LocalContext":
        llm_class = _load_model(self.llm_name, models_path=self.models_path)
        self.llm_judge_class: type[BaseLLM] | None = None
        if self.judge_model_name is not None:
            self.llm_judge_class = _load_model(self.judge_model_name, models_path=self.judge_models_path, info="judge")

        self.config = EvalConfig(
            llm_class=llm_class,
            llm_args=self.llm_args,
            num_samples=self.num_samples,
            max_tokens=self.max_tokens,
            num_fewshot=self.num_fewshot,
            perturbation_config=self.perturbation_config,
            task_name=self.task_name,
            task_subjects=self.task_subjects,
            hf_revision=self.hf_revision,
            output_dir=self.output_dir,
            hf_upload_dir=self.hf_upload_dir,
            hf_upload_repo=self.hf_upload_repo,
            wandb_entity=self.wandb_entity,
            wandb_project=self.wandb_project,
            wandb_run_id=self.wandb_run_id,
            llm_judge_class=self.llm_judge_class,
            judge_model_args=self.judge_model_args,
            batch_size=self.batch_size,
            description=self.description,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> None:
        pass
