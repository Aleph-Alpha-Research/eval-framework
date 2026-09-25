"""German MBPP (EllaMind): https://huggingface.co/datasets/ellamind/mbpp-multilingual

The German counterpart of MBPP — the dataset mirrors the ``text`` / ``code`` / ``test_list`` schema, so these
reuse the builders and reconstruct functions from the ``mbpp`` module with a German instruction wrapper. Unlike
the English variants (fixed English 3-shot block), the German ones sample demonstrations from the (German) eval
split. ``MBPPDEEvalPlusInstruct`` asks for the function in a markdown block and extracts it from the response.
"""

from typing import Any

from eval_framework.answer import ReconstructProgram
from eval_framework.benchmarks.mbpp import (
    END,
    OLMES_STOPS,
    SingleGoldReader,
    bpb,
    evalplus_reconstruct,
    execution,
    instruct_reconstruct,
    olmes_reconstruct,
)
from eval_framework.contract import Benchmark
from eval_framework.fewshot import ChoiceRenderer, FewShot, SampleSplit
from eval_framework.subjects import ListOfSubjects
from eval_framework.tasks.base import Language
from eval_framework.tasks.dataset_loading import DatasetPolicy
from eval_framework.tasks.dataset_revisions import pinned_by_framework
from eval_framework.tasks.task_style import BPBStyle

MBPP_ELLAMIND_DATASET_PATH = "ellamind/mbpp-multilingual"
_SUBJECTS = ListOfSubjects(["deu"])

_DE_PREAMBLE = (
    "Bitte erstelle ein in sich geschlossenes Python-Skript, "
    "das das folgende Problem in einem Markdown-Code-Block löst:"
)
_DE_OLMES_CUE = "Hier ist die fertige Funktion:\n\n```python\n"
_DE_EVALPLUS_CUE = (
    "Hier ist ein Python-Skript mit einer in sich geschlossenen Funktion, "
    "die das Problem löst und die entsprechenden Tests besteht:\n```python"
)


def _de_dataset(dataset: DatasetPolicy | None) -> DatasetPolicy:
    return dataset if dataset is not None else pinned_by_framework(MBPP_ELLAMIND_DATASET_PATH)


def _text(item: dict[str, Any]) -> str:
    return item["text"] if "text" in item else item["prompt"]


def _de_olmes_instruction(item: dict[str, Any]) -> str:
    return f"{_DE_PREAMBLE}\n```\n{_text(item).strip()}\n{item['test_list'][0]}\n```\n"


def _de_evalplus_instruction(item: dict[str, Any]) -> str:
    # EvalPlus additionally normalises Windows line endings in the problem text.
    text = _text(item).replace("\r\n", "\n").strip()
    return f"{_DE_PREAMBLE}\n```\n{text}\n{item['test_list'][0]}\n```\n"


def _norm_code(item: dict[str, Any]) -> str:
    return (item["code"] or "").replace("\r\n", "\n").strip()


def mbpp_de_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    return execution(
        "MBPPDE_OLMES",
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        instruction=_de_olmes_instruction,
        cue=_DE_OLMES_CUE,
        fewshot_target=lambda demo: demo["code"] + "\n",
        answer=ReconstructProgram(olmes_reconstruct, stop_sequences=OLMES_STOPS),
        fewshot_source=SampleSplit(),  # EllaMind ships only a test split; sample (leak-safe) from it
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


def mbpp_de_bpb_olmes(dataset: DatasetPolicy | None = None) -> Benchmark:
    reader = SingleGoldReader(
        question=lambda item: _text(item).replace("\r\n", "\n").strip() + "\n```python\n",
        gold=lambda item: _norm_code(item) + "\n```",
    )
    styler = BPBStyle(question_prefix="", cue_text="", trailing_newline=False, leading_space_continuations=False)
    return bpb(
        "MBPPDE_BPB_OLMES",
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        reader=reader,
        styler=styler,
        fewshot=FewShot(SampleSplit(), ChoiceRenderer(reader, styler)),
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


def mbpp_de_evalplus(dataset: DatasetPolicy | None = None) -> Benchmark:
    return execution(
        "MBPPDE_EvalPlus",
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        instruction=_de_evalplus_instruction,
        cue=_DE_EVALPLUS_CUE,
        fewshot_target=lambda demo: _DE_EVALPLUS_CUE + "\n" + _norm_code(demo) + "\n```",
        answer=ReconstructProgram(evalplus_reconstruct, stop_sequences=["```", "\nassert", "\nBitte erstelle"]),
        fewshot_source=SampleSplit(),
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


def mbpp_de_bpb_evalplus(dataset: DatasetPolicy | None = None) -> Benchmark:
    reader = SingleGoldReader(
        question=_de_evalplus_instruction,
        gold=lambda item: "\n" + _norm_code(item) + "\n" + END,
    )
    styler = BPBStyle(
        question_prefix="", cue_text=_DE_EVALPLUS_CUE, trailing_newline=False, leading_space_continuations=False
    )
    return bpb(
        "MBPPDE_BPB_EvalPlus",
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        reader=reader,
        styler=styler,
        fewshot=FewShot(SampleSplit(), ChoiceRenderer(reader, styler)),
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


def mbpp_de_evalplus_instruct(dataset: DatasetPolicy | None = None) -> Benchmark:
    return execution(
        "MBPPDEEvalPlusInstruct",
        dataset_path=MBPP_ELLAMIND_DATASET_PATH,
        instruction=_de_evalplus_instruction,
        cue="",  # no forced preamble; the code block is extracted from the free-form response
        fewshot_target=lambda demo: f"```python\n{_norm_code(demo)}\n```",
        answer=ReconstructProgram(instruct_reconstruct, stop_sequences=[]),
        fewshot_source=SampleSplit(),
        subjects=_SUBJECTS,
        language=Language.DEU,
        dataset=_de_dataset(dataset),
    )


MBPP_ELLAMIND_BENCHMARKS: list[Benchmark] = [
    mbpp_de_olmes(),
    mbpp_de_bpb_olmes(),
    mbpp_de_evalplus(),
    mbpp_de_bpb_evalplus(),
    mbpp_de_evalplus_instruct(),
]
