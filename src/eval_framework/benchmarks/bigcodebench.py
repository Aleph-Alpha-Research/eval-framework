"""BigCodeBench: https://huggingface.co/datasets/bigcode/bigcodebench

The model completes a self-contained function; the ``CodeExecutionPassAtOne`` metric merges the generated
snippet with the problem's unittest harness (via functions carried, serialized, in the sample's context) and
runs it. Only the OLMES 3-shot variant is registered.
"""

from typing import Any

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
from eval_framework.subjects import ListOfSubjects
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

# These two subjects have no effect: they produce identical prompts and scoring, and both load the same data.
# They select different response formats in other (non-registered) variants, but not here; kept as-is, faithfully
# ported from an earlier version. The data is a single fixed config regardless of subject (see with_config below).
_SUBJECTS = ListOfSubjects(["original", "calibrated"])


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
    kind = Generative(
        build_prompt=_instruction,
        cue="",  # OLMES uses no assistant cue
        ground_truth=lambda item: item["canonical_solution"],  # unused by the test-based metric; recorded gold
        metrics=[CodeExecutionPassAtOneWithCodebench],
        context=_context,
    )
    fewshot = FewShot(
        SampleSplit(),  # no dedicated few-shot split; draw (leak-safe) from the eval split
        FunctionRenderer(lambda row: FewshotExample(prompt=_instruction(row), answer=_fewshot_target(row))),
    )
    # The subject labels are not HF configs; the data is always the default config.
    dataset_policy = (
        dataset if dataset is not None else pinned_by_framework(BIGCODEBENCH_DATASET_PATH).with_hf_config(None)
    )
    return ComposedBenchmark.compose(
        id="BigCodeBench_OLMES",
        kind=kind,
        answer=ReconstructProgram(_reconstruct, stop_sequences=_STOP_SEQUENCES),
        sample_split=_SAMPLE_SPLIT,
        fewshot=fewshot,
        subjects=_SUBJECTS,
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


BIGCODEBENCH_BENCHMARKS: list[Benchmark] = [bigcodebench_olmes()]
