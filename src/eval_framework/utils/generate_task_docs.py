import argparse
import os
from pathlib import Path

import tqdm

from eval_framework.tasks.registry import registered_task_names, registry
from eval_framework.tasks.task_loader import load_extra_tasks
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter

DEFAULT_OUTPUT_DOCS_DIRECTORY = Path("docs/tasks")

EXCLUDED_TASKS: list[str] = []

# Base URL for the main repository to ensure links work even in external/companion repos
REPO_URL = "https://github.com/Aleph-Alpha-Research/eval-framework/blob/main"


def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments for the script."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add-prompt-examples",
        action="store_true",
        default=False,
        required=False,
        help="Unused. Only there for backwards compatibility",
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
    return parser.parse_args(args=cli_args)


def generate_docs_for_task(output_docs_directory: Path, task_name: str, formatters: list[BaseFormatter]) -> None:
    """Generate documentation for a specific task."""
    task_class = registry()[task_name].task_class()

    try:
        task = task_class(num_fewshot=1)
    except (TypeError, ValueError, AssertionError):
        task = task_class(num_fewshot=0)

    (output_docs_directory / f"{task_name}.md").write_text(task.markdown_doc(formatters), encoding="utf-8")


def generate_readme_list(output_docs_directory: Path, total_tasks: int) -> None:
    """Generate a README file listing all tasks with total count."""

    with open(f"{output_docs_directory}/README.md", "w") as f:
        f.write(
            "# Task documentation\n\n"
            "This directory contains the generated documentation for all benchmark tasks available in the package.\n\n"
            f"**Total number of tasks: {total_tasks}**\n\n"
            "The documentation can be generated or updated with "
            "`uv run -m eval_framework.utils.generate_task_docs`.\n\n"
            "NOTE: This is an automatically generated file. Any manual modifications will not be preserved when "
            "the file is updated.\n\n"
        )

        f.write("## List of tasks\n\n")
        # sort files alphabetically and ignore README.md
        for file in sorted(os.listdir(output_docs_directory)):
            if file.endswith(".md") and file != "README.md":
                task_name = file[:-3]
                f.write(f"- [{task_name}]({task_name}.md)\n")


def generate_all_docs(args: argparse.Namespace, output_docs_directory: Path) -> None:
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
    os.makedirs(output_docs_directory, exist_ok=True)

    for task_name in tqdm.tqdm(filtered_tasks, desc="Generating documentation for tasks"):
        try:
            generate_docs_for_task(
                output_docs_directory=output_docs_directory,
                task_name=task_name,
                formatters=formatters,
            )

        except Exception as e:
            print("---")
            print(f"failed generating documentation for task {task_name}: {e}")
            file_path = f"{output_docs_directory}/{task_name}.md"
            if os.path.exists(file_path):
                os.remove(file_path)
            print("---")

    # Pass the total number of processed tasks to the README generator
    generate_readme_list(output_docs_directory=output_docs_directory, total_tasks=len(filtered_tasks))


if __name__ == "__main__":
    print("Generating task documentation...")
    args = parse_args()
    generate_all_docs(args, output_docs_directory=DEFAULT_OUTPUT_DOCS_DIRECTORY)
