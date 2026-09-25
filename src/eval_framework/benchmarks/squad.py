"""SQuAD reading comprehension: a passage and a question whose answer is a span of the passage (or, in v2,
unanswerable). Answers come as several equally-correct annotator spans, scored by (SQuAD-normalised) F1.

- ``SQuAD_OLMES``: v1 (rajpurkar/squad), OLMES Title/Background/Question layout, F1 on the raw generation.
- ``SQuAD2_MA`` / ``SQuAD2_MA_NO_SYSPROMPT``: v2 (rajpurkar/squad_v2), the MA-training prompt; the model is
  told to begin with "Final answer:", which is stripped back off before scoring. The two differ only in
  whether the MA system prompt is present.
"""

from typing import Any

from eval_framework.answer import ExtractFromCompletion
from eval_framework.composed import ComposedBenchmark
from eval_framework.contract import Benchmark
from eval_framework.eval_kind import Generative, ItemText
from eval_framework.fewshot import FewShot, FewshotExample, FewShotSplit, FunctionRenderer
from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.f1 import F1, F1SquadNormalized
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework

SQUAD_V1_DATASET_PATH = "rajpurkar/squad"
SQUAD_V2_DATASET_PATH = "rajpurkar/squad_v2"

_UNANSWERABLE = "unanswerable"


# --- SQuAD_OLMES (v1) -------------------------------------------------------------------------------------

_OLMES_PREAMBLE = (
    "The following are reading comprehension questions, "
    "where the answer to each question is a segment of text from the corresponding background text."
)


def _olmes_prompt(item: dict[str, Any]) -> str:
    return f"Title: {item['title']}\nBackground: {item['context']}\nQuestion: {item['question']}\n"


def _olmes_ground_truth(item: dict[str, Any]) -> list[str]:
    return [f" {a}" for a in item["answers"]["text"]]


def _olmes_demo(item: dict[str, Any]) -> FewshotExample:
    return FewshotExample(prompt=_olmes_prompt(item), answer=f"Answer:{_olmes_ground_truth(item)[0]}")


def squad_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    kind = Generative(
        build_prompt=_olmes_prompt,
        cue="Answer:",  # the model continues after the cue
        ground_truth=_olmes_ground_truth,
        metrics=[F1SquadNormalized],
        initial_prompt=_OLMES_PREAMBLE,
    )
    # F1 scores the whole generation; nothing is extracted
    answer = ExtractFromCompletion(lambda completion_text: completion_text, ["Title:", "\n\n"], max_tokens=50)
    dataset_policy = dataset if dataset is not None else pinned_by_framework(SQUAD_V1_DATASET_PATH)
    return ComposedBenchmark.compose(
        id="SQuAD_OLMES",
        kind=kind,
        answer=answer,
        sample_split="validation",
        fewshot=FewShot(FewShotSplit("train"), FunctionRenderer(_olmes_demo)),
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


# --- SQuAD2_MA (v2) ---------------------------------------------------------------------------------------

_MA_SYSTEM_PROMPT = (
    "You are a helpful assistant and will answer the user's questions carefully, "
    "logically, accurately and well-reasoned.\n"
    "Use the given context to answer the question faithfully. Answer only if the "
    f"answer is present in the given context, otherwise respond with '{_UNANSWERABLE}' "
    "if the answer is not present in the context."
    "Always begin your answer with 'Final answer:'"
)


def _ma_prompt(item: dict[str, Any]) -> str:
    return f"Context:\n{item['context']}\n\nQuestion:\n{item['question']}\n"


def _ma_ground_truth(item: dict[str, Any]) -> list[str]:
    # An unanswerable v2 question has no gold spans; accept the several spellings the model might produce.
    text = item["answers"]["text"]
    return text if text else [_UNANSWERABLE, _UNANSWERABLE + " ", _UNANSWERABLE.capitalize()]


def _ma_demo(item: dict[str, Any]) -> FewshotExample:
    return FewshotExample(prompt=_ma_prompt(item), answer=_ma_ground_truth(item)[0])


def _strip_answer_prefix(completion_text: str) -> str:
    # The MA prompt asks the model to begin with "Final answer:"; take only what follows the last such prefix.
    cleaned = completion_text.strip()
    prefixes = ["Answer", "Final answer"]
    prefixes.extend([f"**{prefix}**" for prefix in prefixes])
    prefixes.reverse()
    for prefix in prefixes:
        idx = cleaned.rfind(prefix + ":")
        if idx != -1:
            cleaned = cleaned[idx + len(prefix) + 1 :].strip()
            break
    return cleaned


def _fixed_system_prompt(text: str) -> ItemText:
    # SQuAD2_MA uses one fixed MA system prompt for every item; wrap it as the per-item callable Generative wants.
    return lambda item: text


def _squad2_ma(id: str, *, system_prompt: ItemText | None, dataset: DatasetPolicy | None) -> Benchmark:
    kind = Generative(
        build_prompt=_ma_prompt,
        cue="",  # no assistant cue; the model answers (beginning with "Final answer:")
        ground_truth=_ma_ground_truth,
        metrics=[AccuracyCompletion, F1, F1SquadNormalized],
        system_prompt=system_prompt,
    )
    answer = ExtractFromCompletion(_strip_answer_prefix, [], max_tokens=10_000)
    dataset_policy = dataset if dataset is not None else pinned_by_framework(SQUAD_V2_DATASET_PATH)
    return ComposedBenchmark.compose(
        id=id,
        kind=kind,
        answer=answer,
        sample_split="validation",
        fewshot=FewShot(FewShotSplit("train"), FunctionRenderer(_ma_demo)),
        dataset_policy=dataset_policy,
        language=Language.ENG,
    )


def squad2_ma(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _squad2_ma("SQuAD2_MA", system_prompt=_fixed_system_prompt(_MA_SYSTEM_PROMPT), dataset=dataset)


def squad2_ma_no_sysprompt(dataset: DatasetPolicy | None = None) -> Benchmark:
    return _squad2_ma("SQuAD2_MA_NO_SYSPROMPT", system_prompt=None, dataset=dataset)


SQUAD_BENCHMARKS: list[Benchmark] = [squad_olmes(), squad2_ma(), squad2_ma_no_sysprompt()]
