"""HumanEval code generation: https://huggingface.co/datasets/openai/openai_humaneval

The model completes a Python function stub; a sandboxed metric runs the result against the problem's tests.
The ``_OLMES`` variants generate the body and are scored by execution; the ``BPB`` variants instead score the
loglikelihood of the gold solution as a single candidate. The builders and prompt shapes here are exposed for
the sibling ``humaneval_plus`` and ``humaneval_ellamind`` modules to reuse (retargeted to their datasets /
language), mirroring how their BaseTask ancestors subclassed this task.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from eval_framework.answer import AnswerPolicy, ReconstructProgram
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark, LanguageSpec
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Generative, ItemText
from eval_framework.fewshot import FewShot, FewshotExample, FunctionRenderer, SampleSplit
from eval_framework.metrics.completion.code_assertion import CodeCompletionAssertion
from eval_framework.shared.types import BaseMetricContext
from eval_framework.subjects import SubjectsSelector
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle
from template_formatting.formatter import Message

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric

HUMANEVAL_DATASET_PATH = "openai/openai_humaneval"

# The generation runs until the model starts a new top-level construct or closes the code fence; the same
# list re-truncates the generation before it is spliced into the program to run.
_OLMES_STOP_SEQUENCES = ["\nclass", "\nif", "\nprint", "\n#", "\n```", "\n```\n\n", "<|eot_id|>"]

# Reconstructs a runnable HumanEval program: the original code prompt, the generated body, and the test
# harness, which calls check() on the entry point and prints True/False for CodeCompletionAssertion to read.
CODE_TO_EXECUTE = """
{start_of_code}
{completion_text}
{test_code}
try:
  check({entry_point})
  print(True)
except Exception as e:
  print(e)
  print(False)
"""


class HumanEvalMetricContext(BaseMetricContext):
    test: str
    entry_point: str
    prompt: str


@dataclass(frozen=True)
class SingleGoldReader(ChoiceReader):
    """Choice reader for the BPB (loglikelihood) variants: the sole candidate is the gold solution, always at
    index 0. ``question`` renders the code prompt and ``gold`` the reference solution from the raw item, so
    the same reader serves both the scored sample and its few-shot demonstrations."""

    question: ItemText
    gold: ItemText

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(raw_question=self.question(item), choices=[self.gold(item)], correct_index=0)


class _BareTargetBPBStyle(BPBStyle):
    """HumanEvalBPB shows the raw canonical solution as the few-shot target while scoring a space-prefixed
    completion — an asymmetry of the original task preserved here by dropping the leading space (and cue)
    from the demonstration target only."""

    @override
    def get_fewshot_target_text(self, choices: list[str], correct_index: int) -> str:
        return choices[correct_index]


def humaneval_context(item: dict[str, Any]) -> HumanEvalMetricContext:
    return HumanEvalMetricContext(test=item["test"], entry_point=item["entry_point"], prompt=item["prompt"])


def _reconstruct_program(
    completion_text: str,
    *,
    context: BaseMetricContext | list[BaseMetricContext] | None,
    ground_truth: str | list[str] | None,
    messages: list[Message],
) -> str:
    assert isinstance(context, HumanEvalMetricContext)
    for stop in _OLMES_STOP_SEQUENCES:
        if stop in completion_text:
            completion_text = completion_text.split(stop)[0]
    return CODE_TO_EXECUTE.format(
        start_of_code=context.prompt,
        completion_text=completion_text,
        test_code=context.test,
        entry_point=context.entry_point,
    )


def execution(
    id: str,
    *,
    dataset_path: str,
    metrics: list[type["BaseMetric"]],
    build_prompt: ItemText,  # item -> the code prompt
    fewshot_target: ItemText,  # item -> the demonstration solution
    answer: AnswerPolicy | None = None,
    subjects: SubjectsSelector | None = None,
    language: LanguageSpec = Language.ENG,
    dataset: DatasetPolicy | None,
) -> Benchmark:
    """A code-generation-scored-by-execution benchmark. ``answer`` defaults to the standard HumanEval
    reconstruction (truncate at a stop sequence, splice into the test harness); an instruct variant can inject
    its own (e.g. extracting a markdown code block)."""
    return ComposedBenchmark.compose(
        id=id,
        kind=Generative(
            build_prompt=build_prompt,
            cue="",  # the model continues the open code fence directly
            ground_truth=lambda item: "Success",  # execution decides pass/fail; the gold string is a placeholder
            metrics=metrics,
            context=humaneval_context,
        ),
        answer=answer
        if answer is not None
        else ReconstructProgram(_reconstruct_program, stop_sequences=_OLMES_STOP_SEQUENCES, max_tokens=1024),
        sample_split="test",
        fewshot=FewShot(
            SampleSplit(),  # HumanEval has no dedicated few-shot split; draw (leak-safe) from the eval split
            FunctionRenderer(lambda row: FewshotExample(prompt=build_prompt(row), answer=fewshot_target(row))),
        ),
        subjects=subjects,
        dataset_policy=dataset if dataset is not None else pinned_by_framework(dataset_path),
        language=language,
    )


def bpb(
    id: str,
    *,
    dataset_path: str,
    styler: BPBStyle,
    question: ItemText,
    gold: ItemText,
    subjects: SubjectsSelector | None = None,
    language: LanguageSpec = Language.ENG,
    dataset: DatasetPolicy | None,
) -> Benchmark:
    """A BPB (loglikelihood-of-the-gold-solution) benchmark: one candidate, scored by ``styler``."""
    return ComposedBenchmark.choice(
        id=id,
        reader=SingleGoldReader(question=question, gold=gold),
        styler=styler,
        sample_split="test",
        fewshot_split="test",
        subjects=subjects,
        dataset_policy=dataset if dataset is not None else pinned_by_framework(dataset_path),
        language=language,
    )


def _olmes_prompt(item: dict[str, Any]) -> str:
    return "```python\n" + item["prompt"]


def olmes_target(item: dict[str, Any]) -> str:
    return item["canonical_solution"] + "```"


def v2_prompt(item: dict[str, Any]) -> str:
    return "```python\n" + item["prompt"].rstrip() + "\n"


def _v2_target(item: dict[str, Any]) -> str:
    return item["canonical_solution"].rstrip() + "\n```"


def v2_execution(
    id: str,
    *,
    dataset_path: str,
    metrics: list[type["BaseMetric"]],
    subjects: SubjectsSelector | None = None,
    language: LanguageSpec = Language.ENG,
    dataset: DatasetPolicy | None,
) -> Benchmark:
    """The ``_OLMES_V2`` execution shape (rstrip + fenced), retargetable to another dataset / metric /
    language — the building block ``humaneval_plus`` and ``humaneval_ellamind`` reuse."""
    return execution(
        id,
        dataset_path=dataset_path,
        metrics=metrics,
        build_prompt=v2_prompt,
        fewshot_target=_v2_target,
        subjects=subjects,
        language=language,
        dataset=dataset,
    )


def v2_bpb(
    id: str,
    *,
    dataset_path: str,
    subjects: SubjectsSelector | None = None,
    language: LanguageSpec = Language.ENG,
    dataset: DatasetPolicy | None,
) -> Benchmark:
    """The ``BPB_V2`` loglikelihood shape (fenced gold, no leading space), retargetable to another dataset /
    language — the building block ``humaneval_plus`` and ``humaneval_ellamind`` reuse."""
    return bpb(
        id,
        dataset_path=dataset_path,
        styler=BPBStyle(question_prefix="", cue_text="", trailing_newline=False, leading_space_continuations=False),
        question=v2_prompt,
        gold=_v2_target,
        subjects=subjects,
        language=language,
        dataset=dataset,
    )


def humaneval_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return execution(
        "HumanEval_OLMES",
        dataset_path=HUMANEVAL_DATASET_PATH,
        metrics=[CodeCompletionAssertion],
        build_prompt=_olmes_prompt,
        fewshot_target=olmes_target,
        dataset=dataset,
    )


def humaneval_olmes_v2(dataset: DatasetPolicy | None = None) -> Benchmark:
    return v2_execution(
        "HumanEval_OLMES_V2", dataset_path=HUMANEVAL_DATASET_PATH, metrics=[CodeCompletionAssertion], dataset=dataset
    )


def humaneval_bpb(dataset: DatasetPolicy | None = None) -> Benchmark:
    return bpb(
        "HumanEvalBPB",
        dataset_path=HUMANEVAL_DATASET_PATH,
        styler=_BareTargetBPBStyle(question_prefix="", cue_text="", trailing_newline=False),
        question=lambda item: "```python\n" + item["prompt"].lstrip(),
        gold=lambda item: item["canonical_solution"],
        dataset=dataset,
    )


def humaneval_bpb_v2(dataset: DatasetPolicy | None = None) -> Benchmark:
    return v2_bpb("HumanEvalBPB_V2", dataset_path=HUMANEVAL_DATASET_PATH, dataset=dataset)


HUMANEVAL_BENCHMARKS: list[Benchmark] = [
    humaneval_olmes(),
    humaneval_olmes_v2(),
    humaneval_bpb(),
    humaneval_bpb_v2(),
]
