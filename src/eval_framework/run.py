import argparse
import datetime
from pathlib import Path
from typing import Any

try:
    from eval_framework.context.determined import DeterminedContext
except ImportError:
    DeterminedContext = None  # type: ignore

from eval_framework.context.local import LocalContext
from eval_framework.main import main
from eval_framework.task_loader import load_extra_tasks
from eval_framework.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

CONTEXT = {
    "local": LocalContext,
    "determined": DeterminedContext,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context",
        type=str,
        required=False,
        default="local",
        choices=["local", "determined"],
        help="The context in which the evaluation is run.",
    )
    parser.add_argument(
        "--models",
        type=Path,
        required=False,
        default=Path(__file__).parent / "llm" / "models.py",
        help="The path to the Python module file containing model classes.",
    )
    parser.add_argument(
        "--extra-task-modules",
        nargs="*",
        default=[],
        required=False,
        help="List of files and folders containing additional task definitions.",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        required=False,
        help=(
            "The class derived from `eval_framework.llm.base.BaseLLM` found in the "
            "models module to instantiate for evaluation."
        ),
    )
    parser.add_argument(
        "--llm-args",
        type=str,
        nargs="*",
        required=False,
        help="The arguments to pass to the LLM as key=value pairs.",
    )
    parser.add_argument(
        "--num-samples", type=int, required=False, help="The number of samples per subject to evaluate."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=False,
        help="The maximum number of tokens to generate for each sample. Overwrites any task default value.",
    )
    parser.add_argument(
        "--num-fewshot", type=int, required=False, default=0, help="The number of fewshot examples to use."
    )
    parser.add_argument("--task-name", type=str, required=False, help="The name of the task to evaluate.")

    # Perturbation arguments
    parser.add_argument(
        "--perturbation-type",
        type=str,
        required=False,
        choices=[
            "editor",
            "permute",
            "replace",
            "delete",
            "uppercase",
        ],
        help=(
            "The type of perturbation to apply. Note that this may not make sense especially for prompts"
            " containing math and code."
        ),
    )
    parser.add_argument(
        "--perturbation-probability",
        type=float,
        required=False,
        default=None,
        help="The probability of applying perturbation to each word or character (between 0.0 and 1.0).",
    )
    parser.add_argument(
        "--perturbation-seed",
        type=int,
        required=False,
        default=42,
        help="Random seed controlling perturbations.",
    )
    parser.add_argument(
        "--task-subjects",
        type=str,
        nargs="+",
        required=False,
        help="The subjects of the task to evaluate. If empty, all subjects are evaluated. Subjects in the form of "
        "tuples can be specified in a comma-delimited way, possibly using wildcard * in some dimensions of a tuple, "
        'e.g. "DE_DE, *" or "FR_FR, astronomy".',
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        required=False,
        default=None,
        help="A tag name, a branch name, or commit hash for the task HF dataset",
    )
    parser.add_argument(
        "--judge-models",
        type=Path,
        required=False,
        help="The path to the Python module file containing LLM judge model classes.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs",
        required=False,
        help="The path for the evaluation outputs.",
    )
    parser.add_argument(
        "--hf-upload-repo",
        type=str,
        default="Aleph-Alpha/evaluation-results",
        required=False,
        help="Customizable path for the HuggingFace git repository where runs will be stored",
    )
    parser.add_argument(
        "--hf-upload-dir",
        type=str,
        default="",
        required=False,
        help="Folder name for the HuggingFace git repository where runs will be stored",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        required=False,
        help="The name of the Weights & Biases project to log runs to. "
        "The environment variable WANDB_API_KEY must be set",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        required=False,
        help="The name of the Weights & Biases entity to log runs to. Defaults to the user's default entity",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        required=False,
        help="The ID of an existing Weights & Biases run to resume. "
        "If not given, creates a new fun. If given and exists, "
        "will continue the run but will overwrite the pthon command logged in wandb.",
    )
    parser.add_argument(
        "--description",
        type=str,
        required=False,
        help="Description of the run. This will be added to the metadata of the run to help with bookkeeping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        required=False,
        help=(
            "Size of batch of samples to send to the LLM for evaluation in parallel."
            "Use 1 for sequential running (default) and None to have a single batch."
        ),
    )
    parser.add_argument(
        "--save-logs",
        action="store_true",
        default=True,
        required=False,
        help="Whether to save logs to a file in the output directory (default: True).",
    )
    parser.add_argument(
        "--judge-model-name",
        type=str,
        required=False,
        help=(
            "The class derived from `eval_framework.llm.base.BaseLLM` found in the "
            "judge-models module to instantiate for LLM judge evaluation metrics."
        ),
    )
    parser.add_argument(
        "--judge-model-args",
        type=str,
        required=False,
        nargs="*",
        help=("The args of the judge model used within OpenAIModel wrapper."),
    )

    llm_args: dict[str, Any] = {}
    args = parser.parse_args()

    for arg in args.llm_args or []:
        if "=" in arg:
            key, value = arg.split("=", 1)

            # Handle nested keys like "sampling_params.temperature=0.7"
            if "." in key:
                nested_key, sub_key = key.split(".", 1)
                if nested_key not in llm_args:
                    llm_args[nested_key] = {}
                llm_args[nested_key][sub_key] = value
            else:
                llm_args[key] = value

    args.llm_args = llm_args

    judge_model_args = {}
    for arg in args.judge_model_args or []:
        if "=" in arg:
            key, value = arg.split("=", 1)
            judge_model_args[key] = value

    args.judge_model_args = judge_model_args

    # if args.extra_task_modules:
    #     # Convert the comma-separated string into a list
    #     args.extra_task_modules = [file_or_dir.strip() for file_or_dir in args.extra_task_modules.split(",")]
    # else:
    #     args.extra_task_modules = None

    return args


def run_with_kwargs(kwargs: dict) -> None:
    # Setup logging for the output directory
    output_dir = kwargs.get("output_dir", "results")
    setup_logging(output_dir)

    now = datetime.datetime.now()
    logger.info(f"starting time: {now}")

    if kwargs["extra_task_modules"]:
        load_extra_tasks(kwargs["extra_task_modules"])

    context_name = kwargs.pop("context")

    context = CONTEXT[context_name](
        llm_name=kwargs["llm_name"],
        models_path=kwargs["models"],
        num_samples=kwargs["num_samples"],
        max_tokens=kwargs["max_tokens"],
        num_fewshot=kwargs["num_fewshot"],
        task_name=kwargs["task_name"],
        task_subjects=kwargs["task_subjects"],
        hf_revision=kwargs["hf_revision"],
        output_dir=kwargs["output_dir"],
        wandb_project=kwargs["wandb_project"],
        wandb_entity=kwargs["wandb_entity"],
        wandb_run_id=kwargs["wandb_run_id"],
        hf_upload_dir=kwargs["hf_upload_dir"],
        hf_upload_repo=kwargs["hf_upload_repo"],
        llm_args=kwargs["llm_args"],
        judge_models_path=kwargs["judge_models"],
        judge_model_name=kwargs["judge_model_name"],
        judge_model_args=kwargs["judge_model_args"],
        batch_size=kwargs["batch_size"],
        description=kwargs["description"],
        perturbation_type=kwargs["perturbation_type"],
        perturbation_probability=kwargs["perturbation_probability"],
        perturbation_seed=kwargs["perturbation_seed"],
        # save_logs=kwargs["save_logs"],
    )

    with context as ctx:
        if ctx.config is None:
            raise ValueError(f"Context configuration is not set for '{type(ctx)}'.")

        main(
            llm=ctx.config.llm_class(**ctx.config.llm_args),
            config=ctx.config,
            should_preempt_callable=ctx.should_preempt,
            trial_id=ctx.get_trial_id(),
        )

    logger.info(f"time since start: {datetime.datetime.now() - now}")


def run() -> None:
    run_with_kwargs(vars(parse_args()))


# Enable execution via `python -m eval_framework.run`. Useful for
# debugging via `debugpy -m eval_framework.run`
if __name__ == "__main__":
    run()
