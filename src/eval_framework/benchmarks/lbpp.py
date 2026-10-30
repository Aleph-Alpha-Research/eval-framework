"""LBPP (Less Basic Python Programming): https://huggingface.co/datasets/CohereLabs/lbpp

161 hand-written Python problems built to be fresh and harder than MBPP, in MBPP's shape: an instruction, a gold
solution, and ``assert`` tests. The paper (Matton et al., 2024, https://arxiv.org/abs/2407.07565) reports zero-shot
pass@1 without publishing its prompt, so both variants reuse MBPP_EvalPlus's template: the instruction and the
first test in a markdown block, answered after MBPP_EvalPlus's cue. ``LBPP_Cohere_EN`` executes the generated
script against the tests; ``LBPP_Cohere_EN_BPB`` scores the bits-per-byte of the gold solution. Demonstrations are
drawn (leak-safe) from the test split, the only one LBPP ships. ``LBPP_Cohere_EN_BPB_{Cpp,Java,JS,Rust,Go}`` score
the same for LBPP v2's other languages (75-84 problems each, mostly translations of the Python ones), with the
language named in the preamble and cue.

The dataset stores its code columns compressed and pickled to resist scraping; they are decoded where read
(accepting plain values only), and the leakage-canary row is dropped. Each test program runs the row's
``test_setup`` (imports and helpers the tests use) without its ``from code import ...`` line, which expects the
solution in a ``code`` module. oe_eval's ``lbpp`` task drops the setup's first line instead, which loses the
imports of the 5 rows whose setup does not start with that line.

Reproduction
------------

- Date: 2026-10-30.
- Reference: OLMo 3 (Olmo et al., 2025, https://arxiv.org/abs/2512.13961), Table 3: ``allenai/Olmo-3-1025-7B``
  (base), 17.1 pass@1; settings from Table 46 (0-shot, T=0.6, top-p=0.6, pass@1 over 32 samples), i.e. oe_eval's
  ``lbpp::olmo3``.
- Settings: ``--model-name allenai/Olmo-3-1025-7B --formatter concat --num-fewshot 0 --temperature 0.6 --top-p 0.6
  --repeats 8``.
- Results (pass@1 in %, ± standard error over the 161 items)::

    Task              Paper   Ours          Gap
    LBPP_Cohere_EN    17.1    18.9 ± 2.6    +1.8

- Details: within one standard error. ``lbpp::olmo3`` differs in its cue (``Here is the completed function:``),
  its extra stop sequences (``'\\n\"\"\"'``, ``'\\n#'``), a randomly chosen shown test, and its first-line setup drop
  (the 5 rows above always fail, the unfiltered canary passes almost for free).
"""

import base64
import io
import json
import pickle
import zlib
from typing import Any, NamedTuple, override

from eval_framework.answer import PickFromCandidates, ReconstructProgram
from eval_framework.benchmarks.mbpp import SingleGoldReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice, Generative
from eval_framework.fewshot import ChoiceRenderer, FewShot, FewshotExample, FunctionRenderer, SampleSplit
from eval_framework.metrics.completion.code_assertion import LBPPCodeCompletionAssertion
from eval_framework.shared.types import BaseMetricContext
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle
from template_formatting.formatter import Message

LBPP_DATASET_PATH = "CohereLabs/lbpp"
_CANARY_TASK_ID = "lbpp/python/042"
_SAMPLE_SPLIT = "test"


class _CodingLanguage(NamedTuple):
    name: str  # as written in the preamble and cue
    fence: str  # the markdown code-fence tag opening the solution
    id_suffix: str  # suffix for the benchmark ID


# LBPP's languages, keyed by their HF config.
# Python has no suffix, since it is the original LBPP language.
_LANGUAGES: dict[str, _CodingLanguage] = {
    "python": _CodingLanguage("Python", "python", ""),
    "cpp": _CodingLanguage("C++", "cpp", "_Cpp"),
    "java": _CodingLanguage("Java", "java", "_Java"),
    "js": _CodingLanguage("JavaScript", "javascript", "_JS"),
    "rust": _CodingLanguage("Rust", "rust", "_Rust"),
    "go": _CodingLanguage("Go", "go", "_Go"),
}


# MBPP_EvalPlus's instruction preamble and cue, verbatim for Python; other languages swap in their name and fence.
def _preamble(language: str) -> str:
    return (
        f"Please provide a self-contained {_LANGUAGES[language].name} script that solves the following problem"
        " in a markdown code block:"
    )


def _cue(language: str) -> str:
    return (
        f"Below is a {_LANGUAGES[language].name} script with a self-contained function that solves the problem"
        f" and passes corresponding tests:\n```{_LANGUAGES[language].fence}"
    )


# Gold solutions run up to ~4k characters; 4096 tokens is oe_eval's lbpp::olmo3 budget.
_MAX_TOKENS = 4096


class _PlainValuesUnpickler(pickle.Unpickler):
    """Unpickles plain values only: refusing every class lookup keeps a tampered dataset from running code."""

    @override
    def find_class(self, module: str, name: str) -> Any:
        raise pickle.UnpicklingError(f"an LBPP field may hold plain values only, not {module}.{name}")


def _decode(field: str) -> Any:
    """Decode an LBPP code column, ``base64(zlib(pickle(json)))``, to its JSON value (a string or a list)."""
    pickled = zlib.decompress(base64.b64decode(field))
    return json.loads(_PlainValuesUnpickler(io.BytesIO(pickled)).load())


def _dataset(dataset: DatasetPolicy | None, language: str) -> DatasetPolicy:
    source = dataset if dataset is not None else pinned_by_framework(LBPP_DATASET_PATH).with_hf_config(language)
    # Drop the fake (leakage) canary task (only the Python config has one).
    return source.subset(
        lambda row: row["task_id"] != _CANARY_TASK_ID,
        f"its problems, without the leakage canary `{_CANARY_TASK_ID}`",
    )


def _instruction(item: dict[str, Any], language: str) -> str:
    return f"{_preamble(language)}\n```\n{item['instruction'].strip()}\n{_decode(item['test_list'])[0]}\n```\n"


def _solution(item: dict[str, Any]) -> str:
    return _decode(item["completion"]).strip()


def _tests(item: dict[str, Any]) -> str:
    """The test program: the setup without its ``from code import ...`` line (the solution is defined in the same
    script), then the tests."""
    setup = [line for line in _decode(item["test_setup"]).splitlines() if not line.startswith("from code import")]
    return "\n".join([*setup, *_decode(item["test_list"])])


def _program(
    completion_text: str,
    *,
    context: BaseMetricContext | list[BaseMetricContext] | None,
    ground_truth: str | list[str] | None,
    messages: list[Message],
) -> str:
    """The generated script, then the tests, printing whether they pass. The tests run through ``exec`` rather
    than indented into a ``try`` block (MBPP's harness): indenting would alter their multi-line string literals."""
    assert isinstance(ground_truth, str)
    return (
        f"{completion_text}\n\n"
        f"try:\n    exec({ground_truth!r})\n    score = True\nexcept:\n    score = False\nprint(score)\n"
    )


def lbpp(dataset: DatasetPolicy | None = None) -> Benchmark:
    """LBPP: write a Python script solving the instruction, scored by executing it against the tests."""
    kind = Generative(
        build_prompt=lambda item: _instruction(item, "python"),
        cue=_cue("python"),
        ground_truth=_tests,  # like MBPP's gold asserts, appended to the generation by the reconstruction
        metrics=[LBPPCodeCompletionAssertion],
    )
    few_shot_prompt = FunctionRenderer(
        lambda item: FewshotExample(
            prompt=_instruction(item, "python"), answer=f"{_cue('python')}\n{_solution(item)}\n```"
        )
    )
    return ComposedBenchmark.compose(
        id="LBPP_Cohere_EN",
        kind=kind,
        answer=ReconstructProgram(
            _program, stop_sequences=["```", "\nassert", "\nPlease provide"], max_tokens=_MAX_TOKENS
        ),
        sample_split=_SAMPLE_SPLIT,
        fewshot=FewShot(SampleSplit(), few_shot_prompt),
        dataset_policy=_dataset(dataset, "python"),
        language=Language.ENG,
    )


def lbpp_bpb(dataset: DatasetPolicy | None = None, language: str = "python") -> Benchmark:
    """LBPP BPB: the bits-per-byte of the gold solution, closing its code fence, after the same prompt and cue."""
    reader = SingleGoldReader(
        question=lambda item: _instruction(item, language), gold=lambda item: f"\n{_solution(item)}\n```"
    )
    styler = BPBStyle(
        question_prefix="", cue_text=_cue(language), trailing_newline=False, leading_space_continuations=False
    )
    return ComposedBenchmark.compose(
        id=f"LBPP_Cohere_EN_BPB{_LANGUAGES[language].id_suffix}",
        kind=Choice(reader, styler),
        answer=PickFromCandidates(),
        sample_split=_SAMPLE_SPLIT,
        fewshot=FewShot(SampleSplit(), ChoiceRenderer(reader, styler)),
        dataset_policy=_dataset(dataset, language),
        language=Language.ENG,
    )


LBPP_BENCHMARKS: list[Benchmark] = [lbpp(), *(lbpp_bpb(language=language) for language in _LANGUAGES)]
