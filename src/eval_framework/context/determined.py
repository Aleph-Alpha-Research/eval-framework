import logging
from pathlib import Path
from typing import Any, Type

from determined import get_cluster_info
from determined.core import Context, DummyDistributedContext
from determined.core import init as determined_core_init
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

from eval_framework.context.eval import EvalContext, import_models
from eval_framework.llm.base import BaseLLM
from eval_framework.task_loader import load_extra_tasks
from eval_framework.task_names import TaskName
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.perturbation import PerturbationConfig

logger = logging.getLogger(__name__)


class TaskArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_fewshot: int
    num_samples: int | None = None
    max_tokens: int | None = None
    batch_size: int | None = None
    task_name: TaskName
    judge_model_name: str | None = None
    judge_model_args: dict[str, Any] = {}
    task_subjects: list[str] | None = None
    hf_revision: str | None = None
    perturbation_config: PerturbationConfig | None = None

    @field_serializer("task_name")
    def serialize_task_name(self, value: TaskName) -> str:
        return value.value.NAME

    @field_validator("task_name", mode="before")
    @classmethod
    def validate_task_name(cls, value: str | TaskName) -> TaskName:
        if isinstance(value, str):
            return TaskName.from_name(value)
        return value


class Hyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    llm_name: str
    output_dir: Path
    hf_upload_dir: str | None = None
    hf_upload_repo: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    description: str | None = None
    task_args: TaskArgs
    llm_args: dict[str, Any] | None = {}
    extra_task_modules: list[str] | None = None


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
            "description",
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

        models = import_models(self.models_path)
        if self.hparams.llm_name not in models:
            raise ValueError(f"LLM '{self.hparams.llm_name}' not found.")
        llm_class = models[self.hparams.llm_name]

        llm_judge_class: Type[BaseLLM] | None = None
        judge_model_name = self.hparams.task_args.judge_model_name or self.judge_model_name
        if self.judge_models_path is not None and judge_model_name is not None:
            judge_models = import_models(self.judge_models_path)
            if judge_model_name not in judge_models:
                raise ValueError(f"LLM judge '{judge_model_name}' not found.")
            llm_judge_class = judge_models[judge_model_name]

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
            batch_size=self.hparams.task_args.batch_size or self.batch_size,
            description=self.hparams.description or self.description,
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
