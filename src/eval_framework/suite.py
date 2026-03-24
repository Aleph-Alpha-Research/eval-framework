import importlib.util
import json
import logging
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Self, cast

import numpy as np
import wandb
import yaml
from pydantic import BaseModel, BeforeValidator, ConfigDict, field_validator, model_validator

from eval_framework.context.local import _load_model
from eval_framework.result_processors.result_processor import generate_output_dir
from eval_framework.run import _run_single_task
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.registry import is_registered

logger = logging.getLogger(__name__)

# Fields on TaskSuite that are routed to llm_args when building run kwargs
_LLM_ARG_FIELDS = {"temperature", "top_p", "top_k"}

# Fields on TaskSuite that map directly to EvalConfig / run_with_kwargs keys
_EVAL_CONFIG_FIELDS = {
    "num_samples",
    "num_fewshot",
    "max_tokens",
    "repeats",
    "batch_size",
    "task_subjects",
    "hf_revision",
}

_HYPERPARAM_FIELDS = _LLM_ARG_FIELDS | _EVAL_CONFIG_FIELDS


def parse_strings_to_task_or_suite(v: str | list) -> str | list:
    """Expand bare strings in a list to leaf-suite dicts. Pydantic validates them into TaskSuite."""
    if isinstance(v, str):
        return v
    return [{"tasks": item, "name": item} if isinstance(item, str) else item for item in v]


_VALID_METHODS = {"mean", "median", "passthrough"}


class SuiteAggregate(BaseModel):
    """Model to aggregate results from a suite of tasks."""

    model_config = ConfigDict(extra="forbid")

    name: str
    metric: Annotated[list[str], BeforeValidator(lambda v: [v] if isinstance(v, str) else v)]
    method: str | Callable[[list[float]], float] = "mean"

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str | Callable) -> str | Callable:
        if isinstance(v, str) and v not in _VALID_METHODS:
            raise ValueError(f"Unknown method '{v}'. Valid string methods: {sorted(_VALID_METHODS)}.")
        return v


class TaskSuite(BaseModel):
    # TODO: Figure out versioning for suites. This differs from the versioning of the eval_framework package.
    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    # Tasks can be a string or a list of strings (which becomes a suite) or a suite.
    tasks: Annotated[str | list[str | Self], BeforeValidator(parse_strings_to_task_or_suite)] = []
    aggregates: list[SuiteAggregate] = []

    # things passed to LLM class:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    # a dumping dict for all the non-standard like api keys.
    extra_llm_args: dict[str, Any] = {}

    # things passed to EvalConfig:
    num_samples: int | None = None
    num_fewshot: int | None = None
    max_tokens: int | None = None
    repeats: int | None = None
    batch_size: int | None = None
    task_subjects: list[str] | None = None
    hf_revision: str | None = None

    @model_validator(mode="after")
    def validate_suite(self) -> Self:
        if isinstance(self.tasks, str):
            if self.name is None:
                self.name = self.tasks
            if not is_registered(self.tasks):
                raise ValueError(f"Task '{self.tasks}' is not registered.")
        elif not self.tasks:
            raise ValueError(f"TaskSuite '{self.name}': 'tasks' must not be empty.")
        elif self.name is None:
            raise ValueError("Composite TaskSuite must have a 'name'.")
        return self

    @property
    def is_leaf(self) -> bool:
        return isinstance(self.tasks, str)

    @property
    def task_name(self) -> str:
        """The registered task name. Only valid for leaf tasks."""
        assert self.is_leaf, "task_name is only valid for leaf tasks."
        return self.tasks  # type: ignore[return-value]

    def get_hyperparam_overrides(self) -> dict[str, Any]:
        """Return hyperparam fields that were explicitly set in the suite definition."""
        explicitly_set = self.model_fields_set
        overrides: dict[str, Any] = {}
        for field_name in _HYPERPARAM_FIELDS:
            if field_name in explicitly_set:
                overrides[field_name] = getattr(self, field_name)
        if "extra_llm_args" in explicitly_set:
            overrides["extra_llm_args"] = self.extra_llm_args
        if "task_subjects" in explicitly_set:
            overrides["task_subjects"] = self.task_subjects
        return overrides

    @classmethod
    def load_from_yaml(cls, path: Path) -> Self:
        data = yaml.safe_load(path.read_text())
        return cls.model_validate(data)

    @classmethod
    def load_from_py(cls, path: Path | str) -> Self:
        if isinstance(path, str):
            path = Path(path)
        path = path.resolve()
        module_name = f"_suite_{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load suite module from '{path}'.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "suite"):
            raise ValueError(f"Suite file '{path}' must define a 'suite' variable.")
        suite = module.suite
        if not isinstance(suite, TaskSuite):
            raise TypeError(f"'suite' in '{path}' must be a TaskSuite instance, got {type(suite).__name__}.")
        return cast(Self, suite)

    @classmethod
    def load(cls, path: Path | str) -> Self:
        path = Path(path)
        if path.suffix in (".yaml", ".yml"):
            return cls.load_from_yaml(path)
        elif path.suffix == ".py":
            return cls.load_from_py(path)
        else:
            raise ValueError(f"Unsupported suite file format: '{path.suffix}'. Use .yaml, .yml, or .py.")


class SuiteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    task_results: dict[str, Self] = {}  # this stores the full hierarchy of results. Not used at all.
    aggregates: dict[str, float | None] = {}


def resolve_to_evalconfig_kwargs(
    leaf: TaskSuite, resolved_defaults: dict[str, Any], cli_kwargs: dict[str, Any]
) -> dict:
    """Build the kwargs dict expected by run_with_kwargs() for a single leaf task.

    Merges CLI kwargs as the base, overlays resolved suite defaults, and routes
    temperature/top_p/extra_llm_args into the llm_args dict.
    """
    kwargs = cli_kwargs.copy()

    kwargs["task_name"] = leaf.task_name

    for key, value in resolved_defaults.items():
        if key in _EVAL_CONFIG_FIELDS:
            kwargs[key] = value
        if key in _LLM_ARG_FIELDS:
            kwargs["llm_args"][key] = value
        elif key == "extra_llm_args":
            kwargs["llm_args"].update(value)
    return kwargs


def compute_aggregates(
    aggregates: list[SuiteAggregate],
    child_results: dict[str, SuiteResult],
) -> dict[str, float | None]:
    """Compute suite-level stats from children's results.

    Each SuiteAggregate either reduces a metric across all children
    (method="mean", "median", or a callable) or surfaces a metric
    from exactly one child as-is (method="passthrough").
    """
    result: dict[str, float | None] = {}

    for agg in aggregates:
        if agg.method == "passthrough":
            child = child_results.get(agg.name)
            if child is None:
                raise ValueError(
                    f"SuiteAggregate '{agg.name}' uses method='passthrough' but no child "
                    f"named '{agg.name}' exists. Available children: {list(child_results.keys())}."
                )
            # don't count NaN and None values.
            result[agg.name] = next(
                (v for m in agg.metric if (v := child.aggregates.get(m)) is not None and not math.isnan(v)),
                None,
            )
        else:
            values: dict[str, float] = {}
            for child_name, child in child_results.items():
                for m in agg.metric:
                    val = child.aggregates.get(m)
                    if val is not None and not math.isnan(val):
                        values[child_name] = val
                        break
            result[agg.name] = _apply_method(agg.method, values) if values else None

    return result


def _apply_method(
    method: str | Callable[[list[float]], float],
    values: dict[str, float],
) -> float:
    vals = list(values.values())

    if callable(method):
        return method(vals)
    elif method == "mean":
        return float(np.mean(vals))
    elif method == "median":
        return float(np.median(vals))
    else:
        raise ValueError(f"Unknown aggregation method: '{method}'. Use mean, weighted_mean, median.")


def _merge_defaults(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Merge child overrides on top of parent defaults."""
    return {**parent, **child}


def run_suite(
    suite: TaskSuite,
    cli_kwargs: dict[str, Any],
    parent_defaults: dict[str, Any] | None = None,
    root_suite_name: str | None = None,
) -> SuiteResult:
    """Recursively run all tasks in a suite and compute aggregates bottom-up using
    post-order traversal.

    For a leaf suite: runs the single task via _run_single_task and returns
    the aggregated results directly.
    For a composite suite: recurses into each child, collects results, then
    computes this suite's aggregates.
    """

    parent_defaults = parent_defaults or {}
    current_defaults = _merge_defaults(parent_defaults, suite.get_hyperparam_overrides())
    suite_name = suite.name  # guaranteed non-None by validate_suite
    assert suite_name is not None

    # Track the top-level suite name so all leaf tasks share the same W&B group.
    if root_suite_name is None:
        root_suite_name = suite_name

    # Lets do post-order traversal here. If leaf, go to the code in run.py
    if suite.is_leaf:
        resolved = resolve_to_evalconfig_kwargs(suite, current_defaults, cli_kwargs)
        # Each task in a suite gets its own W&B run (nulling a shared run_id prevents
        # all tasks from piling into the same W&B run), and shares the suite group.
        resolved["wandb_run_id"] = None
        resolved["wandb_group"] = root_suite_name
        logger.info(f"Running task: {suite.task_name}")
        _run_single_task(resolved)
        return SuiteResult(
            name=suite_name,
            task_results={},
            aggregates=_load_aggregated_results(resolved),
        )

    # else keep going down the tree depth first.
    children = cast(list[TaskSuite], suite.tasks)
    child_results: dict[str, SuiteResult] = {}

    for child in children:
        assert child.name is not None
        child_results[child.name] = run_suite(
            child, cli_kwargs, parent_defaults=current_defaults, root_suite_name=root_suite_name
        )

    # we can only compute the aggregates after all the children are run.
    suite_aggregates = compute_aggregates(suite.aggregates, child_results)

    output_dir = Path(cli_kwargs.get("output_dir", "outputs"))
    # check that individual task results are saved in the output directory.
    save_suite_results(output_dir / suite_name, suite_aggregates)
    _log_suite_aggregates_to_wandb(suite_name, root_suite_name, suite_aggregates, cli_kwargs)

    return SuiteResult(name=suite_name, task_results=child_results, aggregates=suite_aggregates)


# I don't like this way of loading the results. But this is how it was done in the original
# eval_framework. I reconstruct the EvalConfig just to create a hash and load that file.
def _load_aggregated_results(resolved_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Load the aggregated_results.json for a completed task run."""

    llm_class = _load_model(resolved_kwargs["llm_name"], models_path=resolved_kwargs["models"])
    llm_instance = llm_class(**resolved_kwargs.get("llm_args", {}))

    config = EvalConfig(
        llm_class=llm_class,
        llm_args=resolved_kwargs.get("llm_args", {}),
        num_samples=resolved_kwargs.get("num_samples"),
        max_tokens=resolved_kwargs.get("max_tokens"),
        num_fewshot=resolved_kwargs.get("num_fewshot", 0),
        task_name=resolved_kwargs["task_name"],
        task_subjects=resolved_kwargs.get("task_subjects"),
        hf_revision=resolved_kwargs.get("hf_revision"),
        output_dir=resolved_kwargs.get("output_dir", "outputs"),
        wandb_project=resolved_kwargs.get("wandb_project"),
        wandb_entity=resolved_kwargs.get("wandb_entity"),
        wandb_run_id=resolved_kwargs.get("wandb_run_id"),
        wandb_upload_results=resolved_kwargs.get("wandb_upload_results"),
        hf_upload_dir=resolved_kwargs.get("hf_upload_dir"),
        hf_upload_repo=resolved_kwargs.get("hf_upload_repo"),
        batch_size=resolved_kwargs.get("batch_size", 1),
        repeats=resolved_kwargs.get("repeats", 1),
        description=resolved_kwargs.get("description"),
        randomize_judge_order=resolved_kwargs.get("randomize_judge_order", False),
        delete_output_dir_after_upload=resolved_kwargs.get("delete_output_dir_after_upload", False),
    )

    output_dir = generate_output_dir(llm_instance.name, config)
    agg_file = output_dir / "aggregated_results.json"
    if agg_file.exists():
        with open(agg_file) as f:
            return json.load(f)

    raise ValueError(f"No aggregated_results.json found at {agg_file}")


def _log_suite_aggregates_to_wandb(
    suite_name: str,
    root_suite_name: str,
    aggregates: dict[str, float | None],
    cli_kwargs: dict[str, Any],
) -> None:
    """Create a W&B run for a composite suite and log its aggregate metrics."""
    from eval_framework.main import _wandb_mode

    wandb_project = cli_kwargs.get("wandb_project")
    if not wandb_project:
        return

    with wandb.init(
        entity=cli_kwargs.get("wandb_entity"),
        project=wandb_project,
        group=root_suite_name,
        job_type="suite",
        name=suite_name,
        mode=_wandb_mode(wandb_project),
        settings=wandb.Settings(disable_code=True),
    ) as run:
        run.log({k: v for k, v in aggregates.items() if v is not None})
    logger.info(f"Logged suite aggregates for '{suite_name}' to W&B project '{wandb_project}'.")


def save_suite_results(output_dir: Path, results: dict[str, float | None]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "suite_aggregated_results.json", "w") as f:
        json.dump(results, f, indent=4, sort_keys=True)
    logger.info(f"Saved suite aggregated results to {output_dir / 'suite_aggregated_results.json'}")
