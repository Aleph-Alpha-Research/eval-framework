import argparse
import inspect
import os
import re

import tqdm

from eval_framework.tasks.registry import get_task, registered_task_names
from eval_framework.tasks.task_loader import load_extra_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter

OUTPUT_DOCS_DIRECTORY = "docs/tasks"

EXCLUDED_TASKS = [
    "SQUAD",  # failing loading the dataset: Feature type 'List' not found
    "SQUAD2",  # failing loading the dataset: Feature type 'List' not found
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add-prompt-examples",
        action="store_true",
        default=False,
        required=False,
        help="If set, examples prompts for each of the formatters will be added in the generated docs.",
    )
    parser.add_argument(
        "--exclude-tasks",
        nargs="*",
        type=str,
        default=[],
        required=False,
        help="List of task names to exclude from documentation generation.",
    )
    parser.add_argument(
        "--extra-task-modules",
        nargs="*",
        type=str,
        default=[],
        required=False,
        help="List of files and folders containing additional task definitions.",
    )
    parser.add_argument(
        "--formatter",
        nargs="*",
        type=str,
        required=False,
        default=["ConcatFormatter", "Llama3Formatter"],
        help="Specify which formatter to use for formatting the task samples. "
        "If not explicitly specified, default formatters will be used.",
    )
    parser.add_argument(
        "--only-tasks",
        nargs="*",
        type=str,
        default=[],
        required=False,
        help="List of task names to generate documentation for. If empty, all tasks will be processed.",
    )
    return parser.parse_args()


def generate_docs_for_task(task_name: str, formatters: list[BaseFormatter], add_prompt_examples: bool) -> None:
    """Generate documentation for a specific task."""
    task_class = get_task(task_name)

    try:
        num_fewshot = 1
        task = task_class(num_fewshot=num_fewshot, custom_subjects=None, custom_hf_revision=None)
    except Exception:
        try:
            num_fewshot = 0
            task = task_class(num_fewshot=num_fewshot, custom_subjects=None, custom_hf_revision=None)
        except Exception as e:
            print(f"Failed to instantiate task {task_name}: {e}")
            return

    with open(f"{OUTPUT_DOCS_DIRECTORY}/{task_name}.md", "w") as f:
        f.write(f"# {task_name}\n\n")
        http_path = f"https://huggingface.co/datasets/{task.DATASET_PATH}" if task.DATASET_PATH else None

        f.write("````\n")  # fence with 4 thicks because some prompts have code blocks with 3 thicks
        f.write(f"NAME = {task_name}\n".strip())
        if hasattr(task, "DATASET_PATH"):
            f.write(f"DATASET_PATH = {task.DATASET_PATH}\n".strip())
        if hasattr(task, "SAMPLE_SPLIT"):
            f.write(f"SAMPLE_SPLIT = {task.SAMPLE_SPLIT}\n".strip())
        if hasattr(task, "FEWSHOT_SPLIT"):
            f.write(f"FEWSHOT_SPLIT = {task.FEWSHOT_SPLIT}\n".strip())
        if hasattr(task, "RESPONSE_TYPE"):
            f.write(f"RESPONSE_TYPE = {task.RESPONSE_TYPE.name}\n".strip())
        if hasattr(task, "METRICS"):
            metrics_list = [f"{m.__name__}" for m in task.METRICS]
            f.write(f"METRICS = [{', '.join(metrics_list)}]\n".strip())
        if hasattr(task, "SUBJECTS"):
            f.write(f"SUBJECTS = {repr(task.SUBJECTS)}\n".strip())
        if hasattr(task, "LANGUAGE"):
            f.write(f"LANGUAGE = {repr(task.LANGUAGE)}\n".strip())
        f.write("````\n\n")

        f.write(f"- Module: [{task_class.__module__}]({task_class.__module__})\n\n")
        task_file = re.sub(".*eval_framework/", "src/eval_framework/", inspect.getfile(task_class))
        f.write(f"- File: [{task_file}](../../{task_file})\n\n")

        if http_path:
            f.write(f"- Link to dataset: [{http_path}]({http_path})\n\n")

        if not add_prompt_examples:
            f.write(
                f"More detailed documentation, with prompt examples and ground truth completions, can be generated "
                f"with `uv run python utils/generate-task-docs.py --add-prompt-examples --only-tasks "
                f'"{task_name}"`.\n'
            )

        else:
            s = next(iter(task.iterate_samples(1)))
            for split in task.dataset:
                f.write(f"- `{split}` has {len(task.dataset[split])} samples\n\n")

            for formatter in formatters:
                f.write(f"## Example prompt with {formatter.__class__.__name__} ({num_fewshot}-shot)\n\n")
                formatted_sample = formatter.format(s.messages, output_mode="string")
                f.write("````\n")
                f.write(f'"{formatted_sample}"')
                f.write("\n````\n\n")

            f.write("## Possible completions:\n\n")
            f.write("````\n")
            if s.possible_completions:
                for item in (
                    s.possible_completions if isinstance(s.possible_completions, list) else [s.possible_completions]
                ):
                    f.write(f'- "{item}"\n')
            else:
                f.write("None\n")
            f.write("````\n\n")

            f.write("## Ground truth:\n\n")
            f.write("````\n")
            if s.ground_truth:
                for item in s.ground_truth if isinstance(s.ground_truth, list) else [s.ground_truth]:
                    f.write(f'- "{item}"\n')
            else:
                f.write("None\n")
            f.write("````\n")


def generate_readme_list() -> None:
    """Generate a README file listing all tasks based on the list of files present in the target directory."""

    with open(f"{OUTPUT_DOCS_DIRECTORY}/README.md", "w") as f:
        f.write(
            "# Task documentation\n\n"
            "This directory contains the generated documentation for all tasks available in `eval-framework`.\n\n"
            "The documentation can be generated or updated with `uv run python utils/generate-task-docs.py`.\n\n"
            "NOTE: This is an automatically generated file. Any manual modifications will not be preserved when"
            "the file is next updated.\n\n"
        )

        f.write("## List of tasks\n\n")
        for file in os.listdir(OUTPUT_DOCS_DIRECTORY):
            if file.endswith(".md") and file != "README.md":
                task_name = file[:-3]
                f.write(f"- [{task_name}]({task_name}.md)\n")


if __name__ == "__main__":
    print("Generating task documentation...")
    args = parse_args()

    # Load extra tasks if specified
    if args.extra_task_modules:
        print(f"Loading extra tasks from: {args.extra_task_modules}")
        load_extra_tasks(args.extra_task_modules)

    # List the tasks to process
    filtered_tasks = []
    for task_name in registered_task_names():
        if args.only_tasks and task_name not in args.only_tasks:
            continue
        if task_name in args.exclude_tasks or task_name in EXCLUDED_TASKS:
            continue
        filtered_tasks.append(task_name)
    filtered_tasks.sort()

    print(f"Found {len(filtered_tasks)} tasks to process: {', '.join([task_name for task_name in filtered_tasks])}")

    # List the formatters to use
    supported_formatters = {f.__class__.__name__: f for f in [ConcatFormatter(), Llama3Formatter()]}
    formatters = []
    for f in args.formatter:
        if f in supported_formatters:
            formatters.append(supported_formatters[f])
        else:
            raise ValueError(f"Unsupported formatter: {f}")

    # Create the output directory if it does not exist
    os.makedirs(OUTPUT_DOCS_DIRECTORY, exist_ok=True)

    for task_name in tqdm.tqdm(filtered_tasks, desc="Generating documentation for tasks"):
        try:
            generate_docs_for_task(task_name, formatters=formatters, add_prompt_examples=args.add_prompt_examples)

        except Exception as e:
            print("---")
            print(f"failed generating documentation for task {task_name}: {e}")
            file_path = f"{OUTPUT_DOCS_DIRECTORY}/{task_name}.md"
            if os.path.exists(file_path):
                os.remove(file_path)
            print("---")

    generate_readme_list()
