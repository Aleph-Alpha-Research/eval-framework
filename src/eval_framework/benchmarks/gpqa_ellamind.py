"""German GPQA (Graduate-level Professional QA, EllaMind) tasks.

https://huggingface.co/datasets/ellamind/gpqa-multilingual

GPQA uses a single distractor set (``incorrect_answers``). The diamond variants restrict evaluation to
the diamond subset — the 198 hardest questions (``is_diamond``) from the original GPQA-Diamond benchmark.
The COT variant has the model reason in German and conclude with the answer letter, which is
leniently regex-extracted from the generation (free-form completion, 0-shot).
"""

import re
from typing import Any, final, override

from eval_framework.answer import ExtractFromCompletion, last_match
from eval_framework.benchmarks.cot import Cot
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.fewshot import NoFewShot
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler, shuffle_correct_with_distractors
from eval_framework.tasks.utils import get_n_letters

GPQA_ELLAMIND_DATASET_PATH = "ellamind/gpqa-multilingual"


@final
class GpqaReader(ChoiceReader):
    """Reads a GPQA item: a single ``incorrect_answers`` distractor set, shuffled in with the correct answer."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["correct_answer"],
            distractors=item["incorrect_answers"],
            seed_text=item["question"] + item["correct_answer"],
        )
        return ChoiceFields(raw_question=item["question"], choices=choices, correct_index=correct_index)


def _gpqa_ellamind_benchmark(id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None) -> Benchmark:
    dataset_policy = dataset if dataset is not None else pinned_by_framework(GPQA_ELLAMIND_DATASET_PATH)
    return ComposedBenchmark.choice(
        id=id,
        reader=GpqaReader(),
        styler=styler,
        sample_split="train",
        fewshot_split="train",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=dataset_policy,
        language=Language.DEU,
    )


def _diamond_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    # The diamond variants keep only the ``is_diamond`` rows of the full dataset.
    if dataset is not None:
        return dataset
    return pinned_by_framework(GPQA_ELLAMIND_DATASET_PATH).subset(
        lambda row: row["is_diamond"], description="the diamond subset"
    )


def _gpqa_ellamind_diamond_benchmark(id: str, styler: TaskStyler, dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_benchmark(id, styler, _diamond_dataset(dataset))


_ANSWER_PHRASE = '"Daher ist die Antwort (ANTWORTBUCHSTABE)"'
_LETTER_CLAUSE = "wobei (ANTWORTBUCHSTABE) einer von (A), (B), (C), (D), (E) usw. ist."


def tulu3_cot_prompt_de(raw_question: str, choices: list[str]) -> str:
    """German translation of ``tulu3_cot_prompt`` (Figure 44 of the Tülu 3 paper,
    https://arxiv.org/pdf/2411.15124): the model reasons briefly and concludes with "Daher ist die Antwort
    (X)". The answer format is stated twice — once before the question and once as a reminder after it."""
    keys = get_n_letters(len(choices))
    options = "\n".join(f"({key}) {choice}" for key, choice in zip(keys, choices))
    return (
        "Beantworte die folgende Multiple-Choice-Frage, indem du den Buchstaben der richtigen "
        "Antwort in Klammern angibst. Begründe deine Antwort KURZ und beende deine Antwort "
        f"unbedingt mit {_ANSWER_PHRASE}, {_LETTER_CLAUSE}"
        f"\n\nFrage: {raw_question}\n{options}"
        "\n\nBeantworte die obige Frage und DENKE DARAN, deine Antwort mit genau dem Satz "
        f"{_ANSWER_PHRASE} abzuschließen, {_LETTER_CLAUSE}"
    )


def tulu_answer_de() -> ExtractFromCompletion:
    """Extracts the letter that ``tulu3_cot_prompt_de`` asks the model to conclude with, as leniently as
    ``tulu_answer_v2``: the last match wins, the parentheses are optional, and there is no stop sequence to
    cut the generation short. It also accepts the English "answer is X", because a model prompted in German
    often still concludes in English. The match is anchored on the answer phrase, so a bare "Antwort D" in the
    reasoning does not count. GPQA always has four options, so only A–D are accepted."""
    return ExtractFromCompletion(
        last_match(
            re.compile(r"\b(?:ist\s+die\s+Antwort|Antwort\s+ist|Antwort:|answer\s+is)\s*\(?([A-D])\b\)?", re.IGNORECASE)
        )
    )


def gpqa_ellamind_mc_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_benchmark("GPQA_ELLAMIND_MC_DE", MCStyle.for_language(Language.DEU), dataset)


def gpqa_ellamind_cloze_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_benchmark("GPQA_ELLAMIND_CLOZE_DE", ClozeStyle.for_language(Language.DEU), dataset)


def gpqa_ellamind_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_benchmark("GPQA_ELLAMIND_BPB_DE", BPBStyle.for_language(Language.DEU), dataset)


def gpqa_ellamind_diamond_mc_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_diamond_benchmark("GPQA_ELLAMIND_DIAMOND_MC_DE", MCStyle.for_language(Language.DEU), dataset)


def gpqa_ellamind_diamond_cloze_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_diamond_benchmark(
        "GPQA_ELLAMIND_DIAMOND_CLOZE_DE", ClozeStyle.for_language(Language.DEU), dataset
    )


def gpqa_ellamind_diamond_bpb_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _gpqa_ellamind_diamond_benchmark(
        "GPQA_ELLAMIND_DIAMOND_BPB_DE", BPBStyle.for_language(Language.DEU), dataset
    )


def gpqa_ellamind_diamond_cot_de(dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.compose(
        id="GPQA_ELLAMIND_DIAMOND_COT_DE",
        kind=Cot(GpqaReader(), build_prompt=tulu3_cot_prompt_de),
        answer=tulu_answer_de(),
        sample_split="train",
        fewshot=NoFewShot(),
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=_diamond_dataset(dataset),
        language=Language.DEU,
    )


GPQA_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    gpqa_ellamind_mc_de(),
    gpqa_ellamind_cloze_de(),
    gpqa_ellamind_diamond_mc_de(),
    gpqa_ellamind_diamond_cloze_de(),
    gpqa_ellamind_bpb_de(),
    gpqa_ellamind_diamond_bpb_de(),
    gpqa_ellamind_diamond_cot_de(),
]
