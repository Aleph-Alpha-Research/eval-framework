"""German CommonsenseQA (EllaMind) tasks.

https://huggingface.co/datasets/ellamind/csqa-multilingual

CSQA supplies separate easy and hard distractors.
"""

from typing import Literal

from eval_framework.choices import EllaMindReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, TaskStyler


def csqa_reader(distractor_level: Literal["easy", "hard"]) -> EllaMindReader:
    """Reads CSQA items: easy/hard distractor lists per level, shuffled in with the correct answer."""
    return EllaMindReader(
        distractor_level,
        question_field="question",
        correct_field="correct_answer",
        easy_field="easy_distractors",
        hard_field="hard_distractors",
        singular=False,
    )


# One styler per format, all sharing the German prefix/cue. The easy/hard distractor axis belongs to
# the reader, so it is orthogonal to the styler choice.
CSQA_ELLAMIND_CLOZE_STYLER = ClozeStyle.for_language(Language.DEU)
CSQA_ELLAMIND_MC_STYLER = MCStyle.for_language(Language.DEU)
CSQA_ELLAMIND_BPB_STYLER = BPBStyle.for_language(Language.DEU)


def _csqa_ellamind_benchmark(id: str, styler: TaskStyler, distractor_level: Literal["easy", "hard"]) -> Benchmark:
    return ComposedBenchmark.compose(
        id=id,
        styler=styler,
        reader=csqa_reader(distractor_level),
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
