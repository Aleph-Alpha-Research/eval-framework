"""German GSM8K (EllaMind): https://huggingface.co/datasets/ellamind/gsm8k-platinum-multilingual

The German counterpart of GSM8K (``Frage:`` / ``Antwort:``), with few-shot demonstrations sampled from the
test split. Each item carries a worked ``solution`` and a ``final_answer``; a demonstration ends with the
German final-answer line ``"Daher ist die Antwort N."``. Two registered variants:

- ``GSM8K_Ellamind_DE_Platinum``: free-form completion, scored on the final integer of the generation.
- ``GSM8K_Ellamind_DE_BPB_Platinum``: bits-per-byte of the single gold solution.
"""

import re
from typing import Any, final, override

from eval_framework.answer import ExtractFromCompletion
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Generative
from eval_framework.fewshot import ChoiceRenderer, SampledFewShot
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle

GSM8K_ELLAMIND_DATASET_PATH = "ellamind/gsm8k-platinum-multilingual"
_STOP_SEQUENCES = ["Frage:"]
_MAX_TOKENS = 1600


def _normalize_number(answer: str) -> str:
    """Drop thousands separators ('.' or ',') so a final answer is a bare integer string."""
    return answer.replace(".", "").replace(",", "")


@final
class _GenerativeFewshotReader(ChoiceReader):
    """Renders a demonstration's solution and (normalised) final answer as the shown German answer."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        answer = f"{item['solution']} Daher ist die Antwort {_normalize_number(item['final_answer'])}."
        return ChoiceFields(raw_question=item["question"], choices=[answer], correct_index=0)


@final
class _BpbReader(ChoiceReader):
    """The single scored 'choice' is the gold solution plus its (raw) final-answer line."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        answer = f"{item['solution']} Daher ist die Antwort {item['final_answer']}."
        return ChoiceFields(raw_question=item["question"], choices=[answer], correct_index=0)


def _extract_final_integer(completion_text: str) -> str:
    """The last integer in the generation (EllaMind answers are integers), or ``"[invalid]"``."""
    numbers = re.findall(r"[-+]?\d+", _normalize_number(completion_text))
    return numbers[-1] if numbers else "[invalid]"


def _ellamind_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_by_framework(GSM8K_ELLAMIND_DATASET_PATH)


def gsm8k_ellamind_de_platinum(dataset: DatasetPolicy | None = None) -> Benchmark:
    # Demonstrations are sampled from the test split and rendered in the same German answer format; the eval
    # itself is free-form (the generative kind), so few-shot rendering and eval are separate policies.
    fewshot = SampledFewShot(
        "test", ChoiceRenderer(_GenerativeFewshotReader(), ClozeStyle(question_prefix="Frage: ", cue_text="Antwort:"))
    )
    return ComposedBenchmark.compose(
        id="GSM8K_Ellamind_DE_Platinum",
        kind=Generative(
            build_prompt=lambda item: f"Frage: {item['question']}\n",
            cue="Antwort:",
            ground_truth=lambda item: _normalize_number(item["final_answer"]),
            metrics=[AccuracyCompletion],
        ),
        answer=ExtractFromCompletion(_extract_final_integer, _STOP_SEQUENCES, max_tokens=_MAX_TOKENS),
        sample_split="test",
        fewshot=fewshot,
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=_ellamind_dataset(dataset),
        language=Language.DEU,
    )


def gsm8k_ellamind_de_bpb_platinum(dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.choice(
        id="GSM8K_Ellamind_DE_BPB_Platinum",
        reader=_BpbReader(),
        styler=BPBStyle(question_prefix="Frage: ", cue_text="Antwort:"),
        sample_split="test",
        fewshot_split="test",
        subjects=ListOfSubjects(["deu"]),
        dataset_policy=_ellamind_dataset(dataset),
        language=Language.DEU,
    )


GSM8K_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    gsm8k_ellamind_de_platinum(),
    gsm8k_ellamind_de_bpb_platinum(),
]
