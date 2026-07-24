from eval_framework.tasks.markdown_doc import markdown_doc
from template_formatting.formatter import Message, Role


class ExampleFormatter:
    """Test double that renders each message as ``ROLE: content`` so the doc reflects the actual messages."""

    def format(self, messages: list[Message], output_mode: str = "string") -> str:
        return "\n".join(f"{message.role.name}: {message.content}" for message in messages)


def test_markdown_doc_with_examples() -> None:
    doc = markdown_doc(
        name="MyTask",
        module="eval_framework.tasks.benchmarks.mytask",
        dataset_path=None,
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

````
NAME = MyTask
SAMPLE_SPLIT = test
FEWSHOT_SPLIT = train
RESPONSE_TYPE = COMPLETION
METRICS = [Accuracy, F1]
SUBJECTS = ['no_subject']
````

- Module: `eval_framework.tasks.benchmarks.mytask`

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
