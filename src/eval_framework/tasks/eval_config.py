import json
from pathlib import Path
from typing import Any

from pydantic import Field, field_serializer, field_validator, model_validator

from eval_framework.base_config import BaseConfig
from eval_framework.constants import ROOT_DIR
from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.llm_metrics.base import BaseLLMJudgeMetric
from eval_framework.task_names import TaskName
from eval_framework.tasks.base import BaseTask
from eval_framework.tasks.perturbation import PerturbationConfig


class EvalConfig(BaseConfig):
    output_dir: Path = Field(ROOT_DIR)
    wandb_project: str | None = Field(None)
    wandb_entity: str | None = Field(None)
    hf_upload_dir: str | None = Field(None)
    hf_upload_repo: str | None = Field(None)
    num_fewshot: int = Field(0, ge=0)
    num_samples: int | None = Field(10, ge=1)  # Allows None or int
    max_tokens: int | None = Field(None)
    perturbation_config: PerturbationConfig | None = Field(None)
    task_name: TaskName = Field()
    task_subjects: list[str] | None = Field(None)
    hf_revision: str | None = Field(None)
    llm_class: type[BaseLLM] = Field()
    llm_args: dict[str, Any] = Field(default_factory=dict)
    llm_judge_class: type[BaseLLM] | None = Field(None)
    judge_model_args: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(1, ge=1)
    description: str | None = Field(None)
    save_intermediate_results: bool = Field(True)
    save_logs: bool = Field(True)

    @field_serializer("output_dir")
    def serialize_output_dir(self, value: Path) -> str:
        return str(value)

    @field_serializer("task_name")
    def serialize_task_name(self, value: TaskName) -> type[BaseTask]:
        return value.value

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, value: str | Path) -> Path:
        if isinstance(value, str):
            return Path(value)
        return value

    @field_validator("task_name", mode="before")
    @classmethod
    def validate_task_name(cls, value: str | TaskName) -> TaskName:
        if isinstance(value, str):
            return TaskName.from_name(value)
        return value

    @field_validator("llm_args", mode="before")
    @classmethod
    def validate_llm_args(cls, value: dict[str, Any]) -> dict[str, Any]:
        typed_value = {}
        for k, v in value.items():
            try:  # maybe this llm argument is actually a number?
                if "." in str(v):
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass
            typed_value[k] = v
        return typed_value

    @field_validator("judge_model_args", mode="before")
    @classmethod
    def validate_judge_model_args(cls, value: dict[str, Any]) -> dict[str, Any]:
        typed_value = {}
        for k, v in value.items():
            try:  # maybe this llm argument is actually a number?
                if "." in str(v):
                    v = float(v)
                else:
                    v = int(v)
            except ValueError:
                pass
            typed_value[k] = v
        return typed_value

    @model_validator(mode="after")
    def validate_llm_judge_defined(self) -> "EvalConfig":
        for metric_class in self.task_name.value.METRICS:
            if issubclass(metric_class, BaseLLMJudgeMetric):
                assert self.llm_judge_class is not None, "The LLM Judge must be defined for this evaluation task."
        return self

    @field_serializer("llm_class")
    def serialize_llm_class(self, value: type[BaseLLM] | None) -> str | None:
        """Serialize the class into its fully qualified name."""
        if value:
            return value.__name__
        return None

    @field_serializer("llm_judge_class")
    def serialize_llm_judge_class(self, value: type[BaseLLM] | None) -> str | None:
        """Serialize the class into its fully qualified name."""
        if value:
            return value.__name__
        return None

    def model_json_dump(self) -> str:
        model_dump = super().model_dump()
        model_dump["task_name"] = model_dump["task_name"].NAME
        return json.dumps(model_dump, sort_keys=True)
