"""MBPP: https://huggingface.co/datasets/google-research-datasets/mbpp

The model writes a Python function; a sandboxed metric appends the gold ``assert`` tests and runs it. The
``_OLMES`` / ``_EvalPlus`` variants generate the body (scored by execution) from a fixed 3-shot block; the
``BPB`` variants score the loglikelihood of the gold solution as a single candidate. The builders, reconstruct
functions, and shared pieces here are exposed for the sibling ``mbpp_ellamind`` module to reuse (retargeted to
the German dataset), mirroring how its BaseTask ancestors subclassed this task.
"""

import ast
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from eval_framework.answer import AnswerPolicy, PickFromCandidates, ReconstructProgram
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark, LanguageSpec
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice, Generative, ItemText
from eval_framework.fewshot import (
    ChoiceRenderer,
    FewShot,
    FewshotExample,
    FewShotPolicy,
    FewShotSource,
    FewShotSplit,
    FunctionRenderer,
    Predefined,
)
from eval_framework.metrics.completion.code_assertion import CodeCompletionAssertion
from eval_framework.shared.types import BaseMetricContext
from eval_framework.subjects import ListOfSubjects, SubjectsSelector
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle
from eval_framework.tasks.utils import extract_python_code_from_response
from template_formatting.formatter import Message

if TYPE_CHECKING:
    from eval_framework.metrics.base import BaseMetric

logger = logging.getLogger(__name__)

MBPP_DATASET_PATH = "google-research-datasets/mbpp"
BEGIN = "```python"
END = "```"

# The OLMES generation runs until the code fence closes or the next docstring/assert/comment; the same list
# re-truncates the generation before the gold asserts are appended.
OLMES_STOPS = ["```", '\n"""', "\nassert", "\n#"]


class MBPPMetricContext(BaseMetricContext):
    tests_code: str


@dataclass(frozen=True)
class SingleGoldReader(ChoiceReader):
    """Choice reader for the BPB (loglikelihood) variants: the sole candidate is the gold solution, always at
    index 0. ``question`` renders the code prompt and ``gold`` the reference solution from the raw item."""

    question: ItemText
    gold: ItemText

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(raw_question=self.question(item), choices=[self.gold(item)], correct_index=0)


def mbpp_context(item: dict[str, Any]) -> MBPPMetricContext:
    return MBPPMetricContext(tests_code="\n".join(item["test_list"]))


def mbpp_ground_truth(item: dict[str, Any]) -> str:
    # The gold answers are the stringified list of assert statements, which the metric appends and runs.
    return f"{item['test_list']}"


def _code_expander(code: str, gt_asserts: str) -> str:
    """Append the gold ``assert`` statements to the generated code, wrapped so the sandbox prints
    ``True``/``False``. ``gt_asserts`` is a stringified list; ``ast.literal_eval`` safely turns it back into
    the individual assert strings."""
    if not gt_asserts:  # no ground-truth asserts given: return the code unchanged
        return code
    asserts = ast.literal_eval(gt_asserts)  # never use eval!
    if not isinstance(asserts, list):
        logger.info("*** WARNING, we expect a list of ground truth asserts here! Sample can not be finalized")
        return code
    stacked_asserts = "".join("    " + gt_assert + "\n" for gt_assert in asserts)
    postfix = "try:\n" + stacked_asserts + "    score = True\nexcept:\n    score = False\nprint(score)"
    return code + postfix


def olmes_reconstruct(
    completion_text: str,
    *,
    context: BaseMetricContext | list[BaseMetricContext] | None,
    ground_truth: str | list[str] | None,
    messages: list[Message],
) -> str:
    for stop in OLMES_STOPS:
        if stop in completion_text:
            completion_text = completion_text.split(stop)[0]
    return _code_expander(completion_text + "\n", str(ground_truth))


def evalplus_reconstruct(
    completion_text: str,
    *,
    context: BaseMetricContext | list[BaseMetricContext] | None,
    ground_truth: str | list[str] | None,
    messages: list[Message],
) -> str:
    # EvalPlus relies on generation stop sequences alone (no extra truncation) before appending the asserts.
    return _code_expander(completion_text + "\n", str(ground_truth))


def instruct_reconstruct(
    completion_text: str,
    *,
    context: BaseMetricContext | list[BaseMetricContext] | None,
    ground_truth: str | list[str] | None,
    messages: list[Message],
) -> str:
    # Instruct models answer in free-form prose; pull the function out of its markdown code block first.
    return _code_expander(extract_python_code_from_response(completion_text) + "\n", str(ground_truth))


def execution(
    id: str,
    *,
    dataset_path: str,
    instruction: ItemText,
    cue: str,
    fewshot_target: ItemText,
    answer: AnswerPolicy,
    fewshot_source: FewShotSource,
    subjects: SubjectsSelector | None = None,
    language: LanguageSpec = Language.ENG,
    dataset: DatasetPolicy | None,
    metrics: list[type["BaseMetric"]] | None = None,
) -> Benchmark:
    """A code-generation-scored-by-execution benchmark: the gold answers are the ``assert`` tests (appended by
    ``answer``'s reconstruction), and demonstrations are drawn from ``fewshot_source``."""
    return ComposedBenchmark.compose(
        id=id,
        kind=Generative(
            build_prompt=instruction,
            cue=cue,
            ground_truth=mbpp_ground_truth,
            metrics=metrics if metrics is not None else [CodeCompletionAssertion],
            context=mbpp_context,
        ),
        answer=answer,
        sample_split="test",
        fewshot=FewShot(
            fewshot_source,
            FunctionRenderer(lambda row: FewshotExample(prompt=instruction(row), answer=fewshot_target(row))),
        ),
        subjects=subjects,
        dataset_policy=dataset if dataset is not None else pinned_by_framework(dataset_path),
        language=language,
    )


def bpb(
    id: str,
    *,
    dataset_path: str,
    reader: ChoiceReader,
    styler: BPBStyle,
    fewshot: FewShotPolicy,
    subjects: SubjectsSelector | None = None,
    language: LanguageSpec = Language.ENG,
    dataset: DatasetPolicy | None,
) -> Benchmark:
    """A BPB (loglikelihood-of-the-gold-solution) benchmark: one candidate, scored by ``styler``. The few-shot
    policy is passed whole because its source (predefined block vs sampled split) and renderer vary."""
    return ComposedBenchmark.compose(
        id=id,
        kind=Choice(reader, styler),
        answer=PickFromCandidates(),
        sample_split="test",
        fewshot=fewshot,
        subjects=subjects,
        dataset_policy=dataset if dataset is not None else pinned_by_framework(dataset_path),
        language=language,
    )


# --- English variants ---


def _mbpp_instruction(item: dict[str, Any]) -> str:
    text = item["text"] if "text" in item else item["prompt"]
    tests = "\n".join(item["test_list"])
    return (
        f"You are an expert Python programmer, and here is your task: {text} "
        f"Your code should pass these tests:\n\n{tests}\n"
    )


def _evalplus_instruction(item: dict[str, Any]) -> str:
    text = item["text"] if "text" in item else item["prompt"]
    test = item["test_list"][0]
    return (
        "Please provide a self-contained Python script that solves the following problem"
        f" in a markdown code block:\n```\n{text.strip()}\n{test}\n```\n"
    )


_MBPP_OLMES_CUE = "Here is the completed function:\n\n```python\n"
_MBPP_EVALPLUS_CUE = (
    "Below is a Python script with a self-contained function that solves the problem"
    " and passes corresponding tests:\n```python"
)

# Three fixed exemplars from the original MBPP "prompt" split (matching oe_eval's ordering); each prompt shows
# only the first test case.
_OLMES_FEWSHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "text": "Write a function to find the similar elements from the given two tuple lists.",
        "code": (
            "def similar_elements(test_tup1, test_tup2):\n"
            "  res = tuple(set(test_tup1) & set(test_tup2))\n  return (res)"
        ),
        "test_list": [
            "assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)",
            "assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)",
            "assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)",
        ],
    },
    {
        "text": "Write a python function to identify non-prime numbers.",
        "code": (
            "import math\ndef is_not_prime(n):\n    result = False\n"
            "    for i in range(2,int(math.sqrt(n)) + 1):\n"
            "        if n % i == 0:\n            result = True\n    return result"
        ),
        "test_list": [
            "assert is_not_prime(2) == False",
            "assert is_not_prime(10) == True",
            "assert is_not_prime(35) == True",
        ],
    },
    {
        "text": (
            "Write a function to find the largest integers from a given list of numbers using heap queue algorithm."
        ),
        "code": (
            "import heapq as hq\ndef heap_queue_largest(nums,n):\n"
            "  largest_nums = hq.nlargest(n, nums)\n  return largest_nums"
        ),
        "test_list": [
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75] ",
            "assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]",
        ],
    },
]


def mbpp_bpb(dataset: DatasetPolicy | None = None) -> Benchmark:
    # The demonstration shows the fenced solution (```python\n…\n```), while the scored completion is the
    # space-prefixed bare solution — so the few-shot target is hand-written rather than the styler's cue+gold.
    def demo(row: dict[str, Any]) -> FewshotExample:
        return FewshotExample(prompt=_mbpp_instruction(row), answer=f"{BEGIN}\n{row['code']}\n{END}")

    return bpb(
        "MBPPBPB",
        dataset_path=MBPP_DATASET_PATH,
        reader=SingleGoldReader(question=_mbpp_instruction, gold=lambda item: item["code"]),
        styler=BPBStyle(question_prefix="", cue_text=BEGIN, trailing_newline=False),
        fewshot=FewShot(FewShotSplit("train"), FunctionRenderer(demo)),
        subjects=ListOfSubjects(["full"]),
        dataset=dataset,
    )


def mbpp_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return execution(
        "MBPP_OLMES",
        dataset_path=MBPP_DATASET_PATH,
        instruction=_evalplus_instruction,
        cue=_MBPP_OLMES_CUE,
        fewshot_target=lambda demo: demo["code"] + "\n",
        answer=ReconstructProgram(olmes_reconstruct, stop_sequences=OLMES_STOPS),
        fewshot_source=Predefined(_OLMES_FEWSHOT_EXAMPLES, count=3, label="MBPP"),
        subjects=ListOfSubjects(["full"]),
        dataset=dataset,
    )


def mbpp_evalplus(dataset: DatasetPolicy | None = None) -> Benchmark:
    return execution(
        "MBPP_EvalPlus",
        dataset_path=MBPP_DATASET_PATH,
        instruction=_evalplus_instruction,
        cue=_MBPP_EVALPLUS_CUE,
        fewshot_target=lambda demo: _MBPP_EVALPLUS_CUE + "\n" + demo["code"] + "\n```",
        answer=ReconstructProgram(evalplus_reconstruct, stop_sequences=["```", "\nassert", "\nPlease provide"]),
        fewshot_source=Predefined(_OLMES_FEWSHOT_EXAMPLES, count=3, label="MBPP"),
        subjects=ListOfSubjects(["full"]),
        dataset=dataset,
    )


def mbpp_bpb_evalplus(dataset: DatasetPolicy | None = None) -> Benchmark:
    reader = SingleGoldReader(question=_evalplus_instruction, gold=lambda item: "\n" + item["code"] + "\n" + END)
    styler = BPBStyle(
        question_prefix="", cue_text=_MBPP_EVALPLUS_CUE, trailing_newline=False, leading_space_continuations=False
    )
    return bpb(
        "MBPP_BPB_EvalPlus",
        dataset_path=MBPP_DATASET_PATH,
        reader=reader,
        styler=styler,
        fewshot=FewShot(Predefined(_OLMES_FEWSHOT_EXAMPLES, count=3, label="MBPP"), ChoiceRenderer(reader, styler)),
        subjects=ListOfSubjects(["full"]),
        dataset=dataset,
    )


MBPP_BENCHMARKS: list[Benchmark] = [mbpp_bpb(), mbpp_olmes(), mbpp_evalplus(), mbpp_bpb_evalplus()]
