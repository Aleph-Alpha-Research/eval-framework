from eval_framework.tasks.markdown_doc import markdown_doc
from template_formatting.formatter import Message, Role


class ExampleFormatter:
    """Test double that renders each message as ``ROLE: content`` so the doc reflects the actual messages."""

    def format(self, messages: list[Message], output_mode: str = "string") -> str:
        return "\n".join(f"{message.role.name}: {message.content}" for message in messages)


def test_markdown_doc_with_examples() -> None:
    doc = markdown_doc(
        name="MyTask",
        dataset_doc="No information about dataset",
        sample_split="test",
        fewshot_split="train",
        response_type="COMPLETION",
        metrics=["Accuracy", "F1"],
        subjects=["no_subject"],
        language=None,
        num_fewshot=1,
        formatters=[ExampleFormatter()],
        example_messages=[Message(role=Role.USER, content="Q")],
        split_sizes={"test": 3, "train": 5},
        possible_completions=["A", "B"],
        ground_truth="A",
    )

    assert (
        doc
        == """\
# MyTask

## Dataset

No information about dataset

````
NAME = MyTask
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [Accuracy, F1]
SUBJECTS = ['no_subject']
````

- `test` has 3 samples

- `train` has 5 samples

## Example prompt with ExampleFormatter (1-shot)

````
"USER: Q"
````

## Possible completions:

````
- "A"
- "B"
````

## Ground truth:

````
- "A"
````
"""
    )


def test_markdown_doc_inserts_dataset_section_verbatim() -> None:
    doc = markdown_doc(
        name="MyTask",
        dataset_doc="- line one\n- line two",
        sample_split="test",
        fewshot_split="train",
        response_type="COMPLETION",
        metrics=["Accuracy", "F1"],
        subjects=["no_subject"],
        language=None,
        num_fewshot=1,
        formatters=[ExampleFormatter()],
        example_messages=None,
        split_sizes=None,
        possible_completions=None,
        ground_truth=None,
    )

    assert (
        doc
        == """\
# MyTask

## Dataset

- line one
- line two

````
NAME = MyTask
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [Accuracy, F1]
SUBJECTS = ['no_subject']
````
"""
    )
