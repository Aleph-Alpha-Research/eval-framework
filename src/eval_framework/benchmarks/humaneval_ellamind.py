"""German HumanEval (EllaMind): https://huggingface.co/datasets/ellamind/humaneval-multilingual

The German counterpart of HumanEval — the dataset mirrors the original, so these reuse the builders and prompt
shapes from the ``humaneval`` module, retargeted to the German ``deu`` subject. ``HumanEvalDEInstruct`` differs:
it asks (in German) for the function in a markdown block and extracts the code from the free-form response.
"""

from typing import Any

from eval_framework.answer import ReconstructProgram
from eval_framework.benchmarks.humaneval import (
    CODE_TO_EXECUTE,
    HumanEvalMetricContext,
    bpb,
    execution,
    olmes_target,
    v2_bpb,
    v2_execution,
    v2_prompt,
)
from eval_framework.contract import Benchmark
from eval_framework.metrics.completion.code_assertion import CodeCompletionAssertion
from eval_framework.shared.types import BaseMetricContext
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle
from eval_framework.tasks.utils import extract_python_code_from_response
from template_formatting.formatter import Message

HUMANEVAL_ELLAMIND_DATASET_PATH = "ellamind/humaneval-multilingual"
_SUBJECTS = ListOfSubjects(["deu"])


def _de_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_by_framework(HUMANEVAL_ELLAMIND_DATASET_PATH)


def humaneval_de_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    # The German OLMES prompt is the V2 (rstrip + fenced) instruction, but keeps the original OLMES few-shot
    # target (``…canonical``` ``); on this dataset the two targets coincide (the solutions end in a newline).
    return execution(
        "HumanEvalDE_OLMES",
        dataset_path=HUMANEVAL_ELLAMIND_DATASET_PATH,
        metrics=[CodeCompletionAssertion],
        build_prompt=v2_prompt,
        fewshot_target=olmes_target,
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


def humaneval_de_olmes_v2(dataset: DatasetPolicy | None = None) -> Benchmark:
    return v2_execution(
        "HumanEvalDE_OLMES_V2",
        dataset_path=HUMANEVAL_ELLAMIND_DATASET_PATH,
        metrics=[CodeCompletionAssertion],
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


def humaneval_de_bpb_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    # The raw prompt as the question (a trailing newline is added by the styler); the gold solution, already
    # 4-space indented, is scored without a leading space.
    return bpb(
        "HumanEvalDE_BPB_OLMES",
        dataset_path=HUMANEVAL_ELLAMIND_DATASET_PATH,
        styler=BPBStyle(question_prefix="", cue_text="", leading_space_continuations=False),
        question=lambda item: item["prompt"].strip("\n"),
        gold=lambda item: item["canonical_solution"],
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


def humaneval_de_bpb_olmes_v2(dataset: DatasetPolicy | None = None) -> Benchmark:
    return v2_bpb(
        "HumanEvalDE_BPB_OLMES_V2",
        dataset_path=HUMANEVAL_ELLAMIND_DATASET_PATH,
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


def _instruct_prompt(item: dict[str, Any]) -> str:
    return (
        "Vervollständige die folgende Python-Funktion. Gib ausschließlich die vollständige Funktion "
        f"in einem Markdown-Codeblock zurück:\n```python\n{item['prompt'].strip()}\n```\n"
    )


def _instruct_target(item: dict[str, Any]) -> str:
    code = item["prompt"].strip() + "\n" + item["canonical_solution"].rstrip()
    return f"```python\n{code}\n```"


def _instruct_reconstruct(
    completion_text: str,
    *,
    context: BaseMetricContext | list[BaseMetricContext] | None,
    ground_truth: str | list[str] | None,
    messages: list[Message],
) -> str:
    # The model answers in free-form German prose; pull the function out of its markdown code block and splice
    # it into the test harness (the prompt is already inside the extracted block, so start_of_code is empty).
    assert isinstance(context, HumanEvalMetricContext)
    return CODE_TO_EXECUTE.format(
        start_of_code="",
        completion_text=extract_python_code_from_response(completion_text),
        test_code=context.test,
        entry_point=context.entry_point,
    )


def humaneval_de_instruct(dataset: DatasetPolicy | None = None) -> Benchmark:
    return execution(
        "HumanEvalDEInstruct",
        dataset_path=HUMANEVAL_ELLAMIND_DATASET_PATH,
        metrics=[CodeCompletionAssertion],
        build_prompt=_instruct_prompt,
        fewshot_target=_instruct_target,
        answer=ReconstructProgram(_instruct_reconstruct, stop_sequences=[], max_tokens=1024),
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


HUMANEVAL_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    humaneval_de_olmes(),
    humaneval_de_olmes_v2(),
    humaneval_de_bpb_olmes(),
    humaneval_de_bpb_olmes_v2(),
    humaneval_de_instruct(),
]
