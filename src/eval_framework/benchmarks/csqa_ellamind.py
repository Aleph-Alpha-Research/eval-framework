"""German CommonsenseQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/csqa-multilingual

CSQA supplies separate easy and hard distractors.
"""

from typing import Any, Literal, final, override

from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler, shuffle_correct_with_distractors


@final
class CsqaReader(ChoiceReader):
    """Reads a CSQA item: easy/hard distractor lists per level, shuffled in with the correct answer."""

    def __init__(self, distractor_level: Literal["easy", "hard"]) -> None:
        self._distractor_level = distractor_level

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        distractors = item["easy_distractors"] if self._distractor_level == "easy" else item["hard_distractors"]
        choices, correct_index = shuffle_correct_with_distractors(
            correct=item["correct_answer"],
            distractors=distractors,
            seed_text=item["question"] + item["correct_answer"],
        )
        return ChoiceFields(raw_question=item["question"], choices=choices, correct_index=correct_index)


# One styler per format, all sharing the German prefix/cue. The easy/hard distractor axis belongs to
# the reader, so it is orthogonal to the styler choice.
CSQA_ELLAMIND_CLOZE_STYLER = ClozeStyle.for_language(Language.DEU)
CSQA_ELLAMIND_MC_STYLER = MCStyle.for_language(Language.DEU)
CSQA_ELLAMIND_BPB_STYLER = BPBStyle.for_language(Language.DEU)


def _csqa_ellamind_benchmark(id: str, styler: TaskStyler, distractor_level: Literal["easy", "hard"]) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        styler=styler,
        reader=CsqaReader(distractor_level),
        sample_split="validation",
        fewshot_split="validation",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=pinned_by_framework("ellamind/csqa-multilingual"),
        language=Language.DEU,
    )


def csqa_ellamind_benchmarks() -> list[Benchmark]:
    return [
        _csqa_ellamind_benchmark("CSQA_ELLAMIND_MC_EASY_DE", CSQA_ELLAMIND_MC_STYLER, "easy"),
        _csqa_ellamind_benchmark("CSQA_ELLAMIND_MC_HARD_DE", CSQA_ELLAMIND_MC_STYLER, "hard"),
        _csqa_ellamind_benchmark("CSQA_ELLAMIND_CLOZE_EASY_DE", CSQA_ELLAMIND_CLOZE_STYLER, "easy"),
        _csqa_ellamind_benchmark("CSQA_ELLAMIND_CLOZE_HARD_DE", CSQA_ELLAMIND_CLOZE_STYLER, "hard"),
        _csqa_ellamind_benchmark("CSQA_ELLAMIND_BPB_DE", CSQA_ELLAMIND_BPB_STYLER, "easy"),
    ]
