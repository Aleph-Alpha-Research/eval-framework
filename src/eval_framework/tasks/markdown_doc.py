from collections.abc import Sequence
from io import StringIO
from typing import Any

from template_formatting.formatter import BaseFormatter, Message


def markdown_doc(
    *,
    name: str,
    dataset_path: str | None,
    dataset_doc: str,
    sample_split: str | None,
    fewshot_split: str | None,
    response_type: str,
    metrics: Sequence[str],
    subjects: Any,
    language: Any,
    num_fewshot: int,
    formatters: Sequence[BaseFormatter],
    example_messages: list[Message] | None,
    split_sizes: dict[str, int] | None,
    possible_completions: str | list[str] | None,
    ground_truth: str | list[str] | None,
) -> str:
    """Render a task's documentation as markdown"""
    buf = StringIO()
    buf.write(f"# {name}\n\n")

    buf.write("````\n")  # fence with 4 thicks because some prompts have code blocks with 3 thicks
    buf.write(f"NAME = {name}".strip() + "\n")
    if dataset_path is not None:
        buf.write(f"DATASET_PATH = {dataset_path}".strip() + "\n")
    if sample_split is not None:
        buf.write(f"SAMPLE_SPLIT = {sample_split}".strip() + "\n")
    if fewshot_split is not None:
        buf.write(f"FEWSHOT_SPLIT = {fewshot_split}".strip() + "\n")
    buf.write(f"RESPONSE_TYPE = {response_type}".strip() + "\n")
    buf.write(f"METRICS = [{', '.join(metrics)}]".strip() + "\n")
    if subjects is not None:
        buf.write(f"SUBJECTS = {subjects!r}".strip() + "\n")
    if language is not None:
        buf.write(f"LANGUAGE = {language!r}".strip() + "\n")
    buf.write("````\n\n")

    buf.write(f"## Dataset\n\n{dataset_doc}\n\n")

    if example_messages is not None:
        for split, size in (split_sizes or {}).items():
            buf.write(f"- `{split}` has {size} samples\n\n")

        for formatter in formatters:
            buf.write(f"## Example prompt with {formatter.__class__.__name__} ({num_fewshot}-shot)\n\n")
            formatted_sample = formatter.format(example_messages, output_mode="string")
            buf.write("````\n")
            buf.write(f'"{formatted_sample}"')
            buf.write("\n````\n\n")

        buf.write("## Possible completions:\n\n")
        buf.write("````\n")
        if possible_completions:
            for item in possible_completions if isinstance(possible_completions, list) else [possible_completions]:
                buf.write(f'- "{item}"\n')
        else:
            buf.write("None\n")
        buf.write("````\n\n")

        buf.write("## Ground truth:\n\n")
        buf.write("````\n")
        if ground_truth:
            for item in ground_truth if isinstance(ground_truth, list) else [ground_truth]:
                buf.write(f'- "{item}"\n')
        else:
            buf.write("None\n")
        buf.write("````\n")

    return buf.getvalue()
