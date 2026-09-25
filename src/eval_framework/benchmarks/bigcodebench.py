"""BigCodeBench: https://huggingface.co/datasets/bigcode/bigcodebench

The model completes a self-contained function; the ``CodeExecutionPassAtOne`` metric merges the generated
snippet with the problem's unittest harness (via functions carried, serialized, in the sample's context) and
runs it. Only the OLMES 3-shot variant is registered.
"""

from typing import Any, final, override

from eval_framework.answer import ReconstructProgram
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Generative
from eval_framework.fewshot import FewShot, FewshotExample, FunctionRenderer, SampleSplit
from eval_framework.metrics.completion.code_execution_pass_at_one import (
    CodeExecutionPassAtOneContext,
    CodeExecutionPassAtOneWithCodebench,
)
from eval_framework.shared.types import BaseMetricContext
from eval_framework.subjects import Subject, Subjects, SubjectsSelector
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.utils import (
    BIG_CODE_BENCH_PACKAGE_MAPPING,
    CallableSerializer,
    _parse_unittest_output,
    unittest_merge_snippets,
)
from template_formatting.formatter import Message

BIGCODEBENCH_DATASET_PATH = "bigcode/bigcodebench"
_SAMPLE_SPLIT = "v0.1.2"

# Instruction/target match oe_eval bigcodebench:3shot::olmo3:v2 (complete variant).
_PROMPT_INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
)
_STOP_SEQUENCES = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "\nif __name__",
    "\ndef main(",
    "\nprint(",
    "\ndef ",
    "\nclass ",
    "\nimport ",
    "\nfrom ",
    "\nassert ",
    "\nPlease",
]

# NOTE: must be the same serializer class the metric uses to decode.
_SERIALIZER = CallableSerializer()


@final
class _OlmesSubjects(SubjectsSelector):
    """The OLMES variant carries two subject labels, ``original`` and ``calibrated``, that both load the default
    config and (unlike the base task's other variants, which OLMES overrides) produce identical prompts. Kept
    for parity with the BaseTask task rather than collapsed to one slice."""

    _NAMES = ("original", "calibrated")

    @override
    def select(self, tokens: list[str]) -> Subjects:
        if tokens and tokens != ["*"]:
            unknown = [token for token in tokens if token not in self._NAMES]
            if unknown:
                raise ValueError(f"Unknown subject(s) {unknown}; this task's subjects are {list(self._NAMES)}.")
            names = [name for name in self._NAMES if name in tokens]
        else:
            names = list(self._NAMES)
        return tuple(Subject(load_key=None, label=name) for name in names)


def _instruction(item: dict[str, Any]) -> str:
    return _PROMPT_INSTRUCTION + "\n```\n" + item["complete_prompt"].strip() + "\n"


def _fewshot_target(item: dict[str, Any]) -> str:
    return item["canonical_solution"] + "\n```"


def _context(item: dict[str, Any]) -> CodeExecutionPassAtOneContext:
    return CodeExecutionPassAtOneContext(
        run_env="python:3.12",
        code_prompt=item["code_prompt"],
        test_code=item["test"],
        snippet_merge_fn=_SERIALIZER.encode(unittest_merge_snippets),
        output_parse_fn=_SERIALIZER.encode(_parse_unittest_output),
        package_downloads=BIG_CODE_BENCH_PACKAGE_MAPPING,
    )


def _reconstruct(
    completion_text: str,
    *,
    context: BaseMetricContext | list[BaseMetricContext] | None,
    ground_truth: str | list[str] | None,
    messages: list[Message],
) -> str:
    # The scored answer is the code prompt plus the (un-fenced) generated body; the metric then merges it with
    # the unittest harness and runs it.
    assert isinstance(context, CodeExecutionPassAtOneContext)
    return context.code_prompt + completion_text.replace("```python", "").replace("```", "")


def bigcodebench_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.compose(
        id="BigCodeBench_OLMES",
        kind=Generative(
            build_prompt=_instruction,
            cue="",  # OLMES uses no assistant cue
            ground_truth=lambda item: item["canonical_solution"],  # unused by the test-based metric; recorded gold
            metrics=[CodeExecutionPassAtOneWithCodebench],
            context=_context,
        ),
        answer=ReconstructProgram(_reconstruct, stop_sequences=_STOP_SEQUENCES),
        sample_split=_SAMPLE_SPLIT,
        fewshot=FewShot(
            SampleSplit(),  # no dedicated few-shot split; draw (leak-safe) from the eval split
            FunctionRenderer(lambda row: FewshotExample(prompt=_instruction(row), answer=_fewshot_target(row))),
        ),
        subjects=_OlmesSubjects(),
        dataset_policy=dataset if dataset is not None else pinned_by_framework(BIGCODEBENCH_DATASET_PATH),
        language=Language.ENG,
    )


BIGCODEBENCH_BENCHMARKS: list[Benchmark] = [bigcodebench_olmes()]
