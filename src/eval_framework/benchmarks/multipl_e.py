"""MultiPL-E: translations of HumanEval and MBPP into 6 programming languages (nuprl/MultiPL-E).

Corresponds to the OLMES suites ``multipl_e_{humaneval,mbpp}:{cpp,java,js,php,rs,sh}::olmo3:n32:v2``. Each of
the 12 variants loads one language config (``<humaneval|mbpp>-<lang>``), is 0-shot only (there are no gold
examples to draw from), and prompts with the target-language function stub verbatim. Grading is entirely
test-based via ``MultiPLECodeAssertion``; the generation is trimmed at language-specific stop tokens.

Recommended run settings for OLMES parity: 0-shot, temperature=0.6, top_p=0.6, repeats=32, max_tokens=1024.
Paper: https://ieeexplore.ieee.org/abstract/document/10103177
"""

from typing import Any

from eval_framework.answer import ExtractFromCompletion, Extractor
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Generative
from eval_framework.fewshot import NoFewShot
from eval_framework.metrics.completion.multipl_e_assertion import MultiPLECodeAssertion, MultiPLEMetricContext
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework

MULTIPL_E_DATASET_PATH = "nuprl/MultiPL-E"
_MAX_TOKENS = 1024

MULTIPL_E_STOP_TOKENS: dict[str, list[str]] = {
    "cpp": ["\n}", "}\n//"],
    "java": ["\n    }\n", "}\n}", "}\n\n", "\n    public static void main", "\n    // Write"],
    "js": ["\nfunction ", "\n/*", "\n//", "\nconsole.log"],
    "php": ["\nfunction", "\n?>", "\n//", "\n#"],
    "rs": ["\n}"],
    "sh": ["}\n", "\n}"],
}


def _context(item: dict[str, Any]) -> MultiPLEMetricContext:
    return MultiPLEMetricContext(prompt=item["prompt"], tests=item["tests"], language=item["language"])


def _trim_at_stops(stop_sequences: list[str]) -> Extractor:
    """The scored answer is the raw continuation trimmed at the first language-specific stop token (the model's
    generation stops there too, but a re-trim keeps the answer clean); the metric then runs it against tests."""

    def extract(completion_text: str) -> str:
        for stop in stop_sequences:
            if stop in completion_text:
                completion_text = completion_text.split(stop)[0]
        return completion_text

    return extract


def _multipl_e(id: str, *, prefix: str, lang: str, dataset: DatasetPolicy | None = None) -> Benchmark:
    stop_sequences = MULTIPL_E_STOP_TOKENS[lang]
    return ComposedBenchmark.compose(
        id=id,
        kind=Generative(
            build_prompt=lambda item: item["prompt"],  # the target-language function stub, verbatim
            cue="",
            ground_truth=lambda item: None,  # test-based; no gold string
            metrics=[MultiPLECodeAssertion],
            context=_context,
        ),
        answer=ExtractFromCompletion(_trim_at_stops(stop_sequences), stop_sequences, max_tokens=_MAX_TOKENS),
        sample_split="test",
        fewshot=NoFewShot(),  # 0-shot only; MultiPL-E has no gold examples to draw from
        # No subjects; each variant loads its one fixed language config from the dataset (not the default one).
        dataset_policy=dataset
        if dataset is not None
        else pinned_by_framework(MULTIPL_E_DATASET_PATH).with_hf_config(f"{prefix}-{lang}"),
        language=Language.ENG,
    )


# (id, HF-config prefix, language code) for each registered variant.
_VARIANTS: list[tuple[str, str, str]] = [
    ("MultiPLEHumanEvalCpp", "humaneval", "cpp"),
    ("MultiPLEHumanEvalJava", "humaneval", "java"),
    ("MultiPLEHumanEvalJs", "humaneval", "js"),
    ("MultiPLEHumanEvalPhp", "humaneval", "php"),
    ("MultiPLEHumanEvalRs", "humaneval", "rs"),
    ("MultiPLEHumanEvalSh", "humaneval", "sh"),
    ("MultiPLEMBPPCpp", "mbpp", "cpp"),
    ("MultiPLEMBPPJava", "mbpp", "java"),
    ("MultiPLEMBPPJs", "mbpp", "js"),
    ("MultiPLEMBPPPhp", "mbpp", "php"),
    ("MultiPLEMBPPRs", "mbpp", "rs"),
    ("MultiPLEMBPPSh", "mbpp", "sh"),
]

MULTIPL_E_BENCHMARKS: list[Benchmark] = [_multipl_e(id, prefix=prefix, lang=lang) for id, prefix, lang in _VARIANTS]
