import logging
from pathlib import Path
from typing import Annotated, Any

from determined._info import get_cluster_info
from determined.core._context import Context
from determined.core._context import init as determined_core_init
from determined.core._distributed import DummyDistributedContext
from pydantic import AfterValidator, BaseModel, ConfigDict

from eval_framework.context.eval import EvalContext
from eval_framework.context.local import _load_model
from eval_framework.llm.base import BaseLLM
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.perturbation import PerturbationConfig
from eval_framework.tasks.registry import validate_task_name
from eval_framework.tasks.task_loader import load_extra_tasks

logger = logging.getLogger(__name__)


class TaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_name: Annotated[str, AfterValidator(validate_task_name)]
    num_fewshot: int
    num_samples: int | None = None
    max_tokens: int | None = None
    batch_size: int | None = None
    judge_model_name: str | None = None
    judge_model_args: dict[str, Any] = {}
    task_subjects: list[str] | None = None
    hf_revision: str | None = None
    perturbation_config: PerturbationConfig | None = None


class Hyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_name: str
    output_dir: Path
    hf_upload_dir: str | None = None
    hf_upload_repo: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    wandb_upload_results: bool | None = None
    description: str | None = None
    task_args: TaskArgs
    llm_args: dict[str, Any] | None = {}
    extra_task_modules: list[str] | None = None
    delete_output_dir_after_upload: bool | None = None


class DeterminedContext(EvalContext):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._core_context: Context | None = None

    def __enter__(self) -> "DeterminedContext":
        distributed_context = DummyDistributedContext()
        self._core_context = determined_core_init(distributed=distributed_context)
        self._core_context.start()
        info = get_cluster_info()

        if info is None:
            raise RuntimeError("Failed to retrieve cluster info.")

        # Load extra tasks if specified first
        extra_task_modules = info.trial.hparams.get("extra_task_modules", None)
        if extra_task_modules:
            name = "extra_task_modules"
            val_cli = getattr(self, name, None)
            val_hparams = extra_task_modules
            if val_hparams:
                if val_cli and val_hparams and val_cli != val_hparams:
                    logger.info(
                        f"CLI argument {name} ({val_cli}) is being overridden by hyperparameters:"
                        f"({val_hparams}). If it fails due to duplicate task names, remove the CLI argument and"
                        "consolidate as a determined hyperparameter instead."
                    )
                load_extra_tasks(val_hparams)

        self.hparams = Hyperparameters(**info.trial.hparams)

        for name in [
            "llm_name",
            "llm_args",
            "output_dir",
            "hf_upload_dir",
            "hf_upload_repo",
            "wandb_project",
            "wandb_entity",
            "wandb_run_id",
            "wandb_upload_results",
            "description",
            "delete_output_dir_after_upload",
        ]:
            val_cli = getattr(self, name, None)
            val_hparams = getattr(self.hparams, name, None)
            if val_cli and val_hparams and val_cli != val_hparams:
                logger.info(f"CLI argument {name} ({val_cli}) is being overridden by hyperparameters: ({val_hparams}).")

        for name in [
            "num_samples",
            "max_tokens",
            "num_fewshot",
            "task_name",
            "task_subjects",
            "batch_size",
            "hf_revision",
            "judge_model_name",
            "judge_model_args",
            "perturbation_config",
        ]:
            val_cli = getattr(self, name, None)
            val_hparams = getattr(self.hparams.task_args, name, None)
            if val_cli and val_hparams and val_cli != val_hparams:
                logger.info(f"CLI argument {name} ({val_cli}) is being overridden by hyperparameters: ({val_hparams}).")

        # Hyperparameters take precedence over core context
        llm_name = self.hparams.llm_name or self.llm_name
        judge_model_name = self.hparams.task_args.judge_model_name or self.judge_model_name

        llm_class = _load_model(llm_name, models_path=self.models_path)
        llm_judge_class: type[BaseLLM] | None = (
            _load_model(judge_model_name, models_path=self.judge_models_path, info="judge")
            if judge_model_name
            else None
        )

        # for all optional hyperparameters, resort to the respective CLI argument if the hyperparameter is not set
        self.config = EvalConfig(
            llm_class=llm_class,
            llm_args=self.hparams.llm_args or self.llm_args,
            num_samples=self.hparams.task_args.num_samples or self.num_samples,
            max_tokens=self.hparams.task_args.max_tokens or self.max_tokens,
            num_fewshot=self.hparams.task_args.num_fewshot,
            task_name=self.hparams.task_args.task_name,
            task_subjects=self.hparams.task_args.task_subjects,
            hf_revision=self.hparams.task_args.hf_revision or self.hf_revision,
            perturbation_config=self.hparams.task_args.perturbation_config or self.perturbation_config,
            output_dir=self.hparams.output_dir,
            llm_judge_class=llm_judge_class,
            judge_model_args=self.hparams.task_args.judge_model_args or self.judge_model_args,
            hf_upload_dir=self.hparams.hf_upload_dir or self.hf_upload_dir,
            hf_upload_repo=self.hparams.hf_upload_repo or self.hf_upload_repo,
            wandb_project=self.hparams.wandb_project or self.wandb_project,
            wandb_entity=self.hparams.wandb_entity or self.wandb_entity,
            wandb_run_id=self.hparams.wandb_run_id or self.wandb_run_id,
            wandb_upload_results=self.hparams.wandb_upload_results or self.wandb_upload_results,
            batch_size=self.hparams.task_args.batch_size or self.batch_size,
            description=self.hparams.description or self.description,
            delete_output_dir_after_upload=self.hparams.delete_output_dir_after_upload
            or self.delete_output_dir_after_upload,
        )

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any | None,
    ) -> None:
        if self._core_context is not None:
            self._core_context.close()
            self._core_context = None

    def should_preempt(self) -> bool:
        if self._core_context is None:
            return False
        return self._core_context.preempt.should_preempt()

    def get_trial_id(self) -> int | None:
        if self._core_context is None:
            return None
        return self._core_context.train._trial_id
