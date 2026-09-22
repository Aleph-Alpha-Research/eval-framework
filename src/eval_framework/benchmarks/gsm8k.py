"""GSM8K: https://huggingface.co/datasets/openai/gsm8k

Grade-school math word problems. Both registered variants use eight fixed, hand-written exemplars (not
sampled from the dataset) and OLMES answer normalisation:

- ``GSM8K_OLMES``: free-form completion, scored on the final number of the generation.
- ``GSM8KBPB``: bits-per-byte of the single normalised gold solution (one forward pass).
"""

import re
from typing import Any, final, override

from eval_framework.answer import ExtractFromCompletion, PickFromCandidates
from eval_framework.choices import ChoiceFields, ChoiceReader
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Choice, Generative
from eval_framework.fewshot import FewshotExample, PredefinedFewShot
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletionOLMES
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle

GSM8K_DATASET_PATH = "openai/gsm8k"
_STOP_SEQUENCES = ["Question:"]
_MAX_TOKENS = 1600
_NUM_FEWSHOT = 8

# Eight fixed exemplars (the standard GSM8K few-shot prompt), used verbatim instead of sampling the dataset.
FEWSHOT_ITEMS = [
    {
        "question": (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
            "After they are done, there will be 21 trees. "
            "How many trees did the grove workers plant today?"
        ),
        "answer": (
            "There are 15 trees originally. Then there were 21 trees after some more were planted. "
            "So there must have been 21 - 15 = 6.\n#### 6"
        ),
    },
    {
        "question": (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
        ),
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
    },
    {
        "question": (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
        ),
        "answer": (
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
            "After eating 35, they had 74 - 35 = 39.\n#### 39"
        ),
    },
    {
        "question": (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
            "How many lollipops did Jason give to Denny?"
        ),
        "answer": (
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
            "So he gave Denny 20 - 12 = 8.\n#### 8"
        ),
    },
    {
        "question": (
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
            "How many toys does he have now?"
        ),
        "answer": (
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
            "5 + 4 = 9.\n#### 9"
        ),
    },
    {
        "question": (
            "There were nine computers in the server room. Five more computers were installed each day, "
            "from monday to thursday. "
            "How many computers are now in the server room?"
        ),
        "answer": (
            "There were originally 9 computers. For each of 4 days, 5 more computers were "
            "added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n#### 29"
        ),
    },
    {
        "question": (
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
            "How many golf balls did he have at the end of wednesday?"
        ),
        "answer": (
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
            "After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33"
        ),
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": (
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
            "So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8"
        ),
    },
]


def _clean_short_answer(continuation: str) -> str:
    """Reduce a solution to its final number, commas removed — the OLMES short-answer form."""
    output = re.sub(r"(\d),(\d)", r"\1\2", continuation)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", output)
    return numbers[-1] if numbers else output


def _add_spaces_around_operators(text: str) -> str:
    operators = {"+", "-", "*", "/", "="}
    result: list[str] = []
    for char in text:
        if char in operators:
            if result and result[-1] != " ":
                result.append(" ")
            result.append(char)
            result.append(" ")
        else:
            result.append(char)
    return "".join(result).replace("  ", " ")


def _normalize_answer_str(item: dict[str, Any]) -> str:
    """OLMES reformatting of a worked solution into a natural sentence (improves BPB scoring)."""
    answer = item["answer"]
    short_answer = _clean_short_answer(answer.split("####")[-1].strip())
    answer = re.sub(r"<<.*?>>", "", answer)
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = re.split(r"####", answer)[0]
    answer = answer[0].capitalize() + answer[1:] if answer else answer
    answer = answer.strip()
    if not answer.endswith("."):
        answer += "."
    answer = answer + f" So the answer is {short_answer}."
    answer = _add_spaces_around_operators(answer)
    return " " + answer


@final
class _Gsm8kBpbReader(ChoiceReader):
    """The single 'choice' is the normalised gold solution; BPB scores the model's likelihood of it."""

    @override
    def read(self, item: dict[str, Any]) -> ChoiceFields:
        return ChoiceFields(raw_question=item["question"], choices=[_normalize_answer_str(item)], correct_index=0)


def _generative_demo(demo: dict[str, Any]) -> FewshotExample:
    return FewshotExample(prompt=f"Question: {demo['question']}\nAnswer:", answer=_normalize_answer_str(demo))


def _bpb_demo(demo: dict[str, Any]) -> FewshotExample:
    return FewshotExample(prompt=f"Question: {demo['question']}\n", answer=f"Answer:{_normalize_answer_str(demo)}")


def _gsm8k_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_by_framework(GSM8K_DATASET_PATH)


def gsm8k_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return ComposedBenchmark.compose(
        id="GSM8K_OLMES",
        kind=Generative(
            build_prompt=lambda item: f"Question: {item['question']}\nAnswer:",
            cue="",  # no assistant cue — the model continues the answer
            ground_truth=lambda item: _clean_short_answer(item["answer"]),
            metrics=[AccuracyCompletionOLMES],
        ),
        answer=ExtractFromCompletion(_clean_short_answer, _STOP_SEQUENCES, max_tokens=_MAX_TOKENS),
        sample_split="test",
        fewshot=PredefinedFewShot(FEWSHOT_ITEMS, _generative_demo, count=_NUM_FEWSHOT, label="GSM8K"),
        subjects=ListOfSubjects(["main"]),
        dataset_policy=_gsm8k_dataset(dataset),
        language=Language.ENG,
    )


def gsm8k_bpb(dataset: DatasetPolicy | None = None) -> Benchmark:
    styler = BPBStyle(cue_text="Answer:", leading_space_continuations=False)
    return ComposedBenchmark.compose(
        id="GSM8KBPB",
        kind=Choice(_Gsm8kBpbReader(), styler),
        answer=PickFromCandidates(),
        sample_split="test",
        fewshot=PredefinedFewShot(FEWSHOT_ITEMS, _bpb_demo, count=_NUM_FEWSHOT, label="GSM8K"),
        subjects=ListOfSubjects(["main"]),
        dataset_policy=_gsm8k_dataset(dataset),
        language=Language.ENG,
    )


GSM8K_BENCHMARKS: list[Benchmark] = [gsm8k_olmes(), gsm8k_bpb()]
