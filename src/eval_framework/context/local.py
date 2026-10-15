from typing import Any

from eval_framework.context.eval import EvalContext, require_model_names
from eval_framework.llm.base import BaseLLM
from eval_framework.llm.openai import OpenAIModel
from eval_framework.tasks.eval_config import EvalConfig


class LocalContext(EvalContext):
    def __enter__(self) -> "LocalContext":
        require_model_names(self.llm_args, self.judge_model_args)
        llm_class = OpenAIModel
        self.llm_judge_class: type[BaseLLM] | None = None
        if self.judge_model_args:
            self.llm_judge_class = OpenAIModel

        self.config = EvalConfig(
            llm_class=llm_class,
            llm_args=self.llm_args,
            num_samples=self.num_samples,
            max_tokens=self.max_tokens,
            num_fewshot=self.num_fewshot,
            task_name=self.task_name,
            task_subjects=self.task_subjects,
            hf_revision=self.hf_revision,
            output_dir=self.output_dir,
            hf_upload_dir=self.hf_upload_dir,
            hf_upload_repo=self.hf_upload_repo,
            wandb_entity=self.wandb_entity,
            wandb_project=self.wandb_project,
            wandb_run_id=self.wandb_run_id,
            wandb_group=self.wandb_group,
            wandb_upload_results=self.wandb_upload_results,
            llm_judge_class=self.llm_judge_class,
            judge_model_args=self.judge_model_args,
            batch_size=self.batch_size,
            description=self.description,
            randomize_judge_order=self.randomize_judge_order,
            delete_output_dir_after_upload=self.delete_output_dir_after_upload,
            repeats=self.repeats,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> None:
        pass
