"""GPQA (Graduate-level Google-Proof Q&A): https://huggingface.co/datasets/Idavidrein/gpqa

Gated, expert-written multiple-choice science questions. Each item has one ``Correct Answer`` and three
``Incorrect Answer N`` distractors; the reader shuffles them together, seeded from the option texts so an
item's option order is stable across runs. The registered variants:

- ``GPQA_OLMES``: OLMES-style loglikelihood over the full ``gpqa_extended`` config.
- ``GPQA_DIAMOND_COT`` / ``_V2``: chain-of-thought completion over the harder ``gpqa_diamond`` config; they
  share one prompt and differ only in how the concluding answer letter is extracted.

One question carries a raw sequence far too long for the prompt budget and is dropped from every config.
"""

import hashlib
import random
import re
from typing import Any, final, override

from eval_framework.answer import ExtractedAnswer
from eval_framework.benchmarks.cot import Cot, tulu3_cot_prompt
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.fewshot import NoFewShot
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import MCStyle

GPQA_DATASET_PATH = "Idavidrein/gpqa"

# One question embeds a raw DNA/protein sequence long enough to blow the prompt budget; it is excluded.
_OVERLONG_QUESTION = (
    "Hello, you are embarking on a new project. You need to produce the HP1alpha protein in E. coli. "
    "Which of these plasmids will you choose?"
)

_COT_ANSWER_RE = re.compile(r"Therefore, the answer is \(([ABCDEFGHIJ])\)")
_COT_V2_ANSWER_RE = re.compile(r"\banswer\s+is:?\s*\(?([A-D])\b\)?", re.IGNORECASE)


def _gpqa_preamble(subject_label: str) -> str:
    # GPQA shows a fixed expert-framing preamble; the subject (a config name) is not part of it.
    return (
        "Here are some example questions from experts. An explanation is given before the final answer. "
        "Answer the final question yourself, giving your reasoning beforehand."
    )


def _preprocess(text: str | None) -> str:
    """Clean an answer option: GPQA options carry stray ``[title]`` / ``[...]`` markup and doubled spaces."""
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


@final
class GpqaReader(ChoiceReader):
    """Reads a GPQA item: three ``Incorrect Answer N`` distractors and the ``Correct Answer``, shuffled
    together. The shuffle is seeded from the (preprocessed) option texts, so the same item always yields the
    same option order regardless of the order in which items are evaluated."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        choices = [_preprocess(item[f"Incorrect Answer {i}"]) for i in range(1, 4)]
        correct = _preprocess(item["Correct Answer"])
        seed = int(hashlib.sha256(f"{choices} {correct}".encode()).hexdigest(), 16)
        rng = random.Random(seed)
        rng.shuffle(choices)
        correct_index = rng.randint(0, 3)
        choices.insert(correct_index, correct)
        return ChoiceFields(raw_question=item["Question"].strip(), choices=choices, correct_index=correct_index)


def _gpqa_cot_candidates(keys: list[str]) -> list[str]:
    # COT is free-form, so these letter labels are not scored; they are carried alongside the sample to keep
    # it identical to the loglikelihood-shaped baseline (and thus its formatter hash).
    return [f" ({key})" for key in keys]


def _gpqa_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return (
        dataset
        if dataset is not None
        else pinned_by_framework(GPQA_DATASET_PATH).subset(
            lambda row: row["Question"] != _OVERLONG_QUESTION, description="excluding one over-long question"
        )
    )


def gpqa_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = MCStyle(
        question_prefix="Question: ",
        cue_text="Answer:",
        space_prefixed_labels=True,
        initial_prompt=_gpqa_preamble,
    )
    return ComposedBenchmark.choice(
        id="GPQA_OLMES",
        reader=GpqaReader(),
        styler=styler,
        sample_split="train",
        fewshot_split="train",
        subjects=ListOfSubjects(["gpqa_extended"]),
        dataset_policy=_gpqa_dataset(dataset),
        language=Language.ENG,
    )


def _gpqa_diamond_cot(id: str, answer: ExtractedAnswer, dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        kind=Cot(GpqaReader(), build_prompt=tulu3_cot_prompt, candidates=_gpqa_cot_candidates),
        answer=answer,
        sample_split="train",
        fewshot=NoFewShot(),
        subjects=ListOfSubjects(["gpqa_diamond"]),
        dataset_policy=_gpqa_dataset(dataset),
        language=Language.ENG,
    )


def gpqa_diamond_cot(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_diamond_cot("GPQA_DIAMOND_COT", ExtractedAnswer(_COT_ANSWER_RE, ["Question:"]), dataset)


def gpqa_diamond_cot_v2(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_diamond_cot("GPQA_DIAMOND_COT_V2", ExtractedAnswer(_COT_V2_ANSWER_RE, [], last_match=True), dataset)


GPQA_BENCHMARKS: list[Benchmark] = [
    gpqa_olmes(),
    gpqa_diamond_cot(),
    gpqa_diamond_cot_v2(),
]
