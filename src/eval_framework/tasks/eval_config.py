import ast
import json
from pathlib import Path
from typing import Annotated, Any

from pydantic import AfterValidator, Field, field_serializer, field_validator, model_validator

from eval_framework.base_config import BaseConfig
from eval_framework.constants import ROOT_DIR
from eval_framework.llm.base import BaseLLM
from eval_framework.metrics.llm_metrics.base import BaseLLMJudgeMetric
from eval_framework.tasks.perturbation import PerturbationConfig
from eval_framework.tasks.registry import get_task, validate_task_name


class EvalConfig(BaseConfig):
    output_dir: Path = Field(default=ROOT_DIR)
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_run_id: str | None = None
    hf_upload_dir: str | None = None
    hf_upload_repo: str | None = None
    num_fewshot: int = Field(0, ge=0)
    num_samples: int | None = Field(10, ge=1)  # Allows None or int
    max_tokens: int | None = None
    perturbation_config: PerturbationConfig | None = None
    task_name: Annotated[str, AfterValidator(validate_task_name)]
    task_subjects: list[str] | None = None
    hf_revision: str | None = None
    llm_class: type[BaseLLM]
    llm_args: dict[str, Any] = Field(default_factory=dict)
    llm_judge_class: type[BaseLLM] | None = None
    judge_model_args: dict[str, Any] = Field(default_factory=dict)
    batch_size: int = Field(default=1, ge=1)
    description: str | None = None
    save_intermediate_results: bool = True
    save_logs: bool = True

    @field_serializer("output_dir")
    def serialize_output_dir(self, value: Path) -> str:
        return str(value)

    @field_validator("output_dir", mode="before")
    @classmethod
    def validate_output_dir(cls, value: str | Path) -> Path:
        if isinstance(value, str):
            return Path(value)
        return value

    @field_validator("llm_args", mode="before")
    @classmethod
    def validate_llm_args(cls, value: dict[str, Any]) -> dict[str, Any]:
        def convert_value(v: Any) -> Any:
            if isinstance(v, dict):
                # Recursively process nested dictionaries (like sampling_params)
                return {k: convert_value(nested_v) for k, nested_v in v.items()}
            elif isinstance(v, str):
                try:
                    # Try to evaluate as a Python literal (int, float, bool, None, list, dict, etc.)
                    return ast.literal_eval(v)
                except (ValueError, SyntaxError):
                    return v  # keep as string if not a valid literal
            else:
                return v  # already proper type

        return convert_value(value)

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
        task = get_task(self.task_name)
        for metric_class in task.METRICS:
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
        model_dump = self.model_dump()
        return json.dumps(model_dump, sort_keys=True)
