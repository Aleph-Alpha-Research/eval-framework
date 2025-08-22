from typing import Any, Type

from eval_framework.context.eval import EvalContext, import_models
from eval_framework.llm.base import BaseLLM
from eval_framework.tasks.eval_config import EvalConfig


class LocalContext(EvalContext):
    def __enter__(self) -> "LocalContext":
        models = import_models(self.models_path)
        if self.llm_name not in models:
            raise ValueError(f"LLM '{self.llm_name}' not found.")
        llm_class = models[self.llm_name]

        self.llm_judge_class: Type[BaseLLM] | None = None
        if self.judge_models_path is not None and self.judge_model_name is not None:
            judge_models = import_models(self.judge_models_path)
            if self.judge_model_name not in judge_models:
                raise ValueError(f"LLM judge '{self.judge_model_name}' not found.")
            self.llm_judge_class = judge_models[self.judge_model_name]

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
