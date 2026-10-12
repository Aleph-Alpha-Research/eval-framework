"""Specification of the MMLU-ProX tasks.

- the reader turns a raw row into ChoiceFields, dropping padded options
- the dataset loader filters both splits to the requested category
- prompt spec: build the real benchmarks (via their ``mmlu_prox_*`` constructors) over fictional rows
  and assert the assembled messages, ground truth, and scored completions
- CoT variants: prompt spec over the same fictional rows, plus answer extraction through the public
  completion boundary

All of the above are offline (no dataset download). ``test_formatter_hash`` is the exception: it
loads the real pinned Hugging Face dataset.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from eval_framework.benchmarks.mmlu_prox import (
    MMLU_PROX_BENCHMARKS,
    MMLUProXDatasetLoader,
    MMLUProXReader,
    mmlu_prox_lilab_de_bpb,
    mmlu_prox_lilab_de_cloze,
    mmlu_prox_lilab_de_cot,
    mmlu_prox_lilab_de_cot_v2,
    mmlu_prox_lilab_de_mc,
    mmlu_prox_lilab_de_mc_olmes,
)
from eval_framework.contract import Benchmark
from template_formatting.formatter import BaseFormatter, ConcatFormatter, Llama3Formatter, Message, Role
from tests.tests_eval_framework.benchmarks.utils import (
    DatasetStub,
    StubLlm,
    assert_benchmark_formatter_hash,
    first_sample,
)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize(
    "benchmark", [b for b in MMLU_PROX_BENCHMARKS if not b.id().endswith("_V2")], ids=lambda b: b.id()
)
def test_formatter_hash(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=1)


@pytest.mark.formatter_hash
@pytest.mark.parametrize("formatter_cls", [Llama3Formatter, ConcatFormatter])
@pytest.mark.parametrize("benchmark", [b for b in MMLU_PROX_BENCHMARKS if b.id().endswith("_V2")], ids=lambda b: b.id())
def test_formatter_hash_zeroshot_only(benchmark: Benchmark, formatter_cls: type[BaseFormatter]) -> None:
    assert_benchmark_formatter_hash(benchmark, formatter_cls, num_fewshot=0)


# ---------------------------------------------------------------------------
# Fictional rows in the MMLU-ProX format. NOT real dataset examples.
# Both rows share one category so the category-restricted prompt spec below finds them.
# ---------------------------------------------------------------------------

# Test-split style: unused option slots are None.
_EVAL_ROW: dict[str, Any] = {
    "question_id": 0,
    "question": "Was ist die Hauptstadt von Frankreich?",
    "option_0": "Berlin",
    "option_1": "Paris",
    "option_2": "London",
    "option_3": "Madrid",
    "option_4": "Rom",
    "option_5": None,
    "option_6": None,
    "option_7": None,
    "option_8": None,
    "option_9": None,
    "answer": "B",
    "answer_index": 1,
    "cot_content": "",
    "category": "history",
}

# Validation-split style: unused option slots are the literal string "N/A".
_FEWSHOT_ROW: dict[str, Any] = {
    "question_id": 1,
    "question": "Was ist 1+1?",
    "option_0": "1",
    "option_1": "2",
    "option_2": "3",
    "option_3": "N/A",
    "option_4": "N/A",
    "option_5": "N/A",
    "option_6": "N/A",
    "option_7": "N/A",
    "option_8": "N/A",
    "option_9": "N/A",
    "answer": "B",
    "answer_index": 1,
    # Real rows have no whitespace after the "A:" cue; keep that quirk in the fictional row.
    "cot_content": "A: Denken wir Schritt für Schritt nach.Eins plus eins ergibt zwei.Die Antwort ist (B)",
    "category": "history",
}


# ---------------------------------------------------------------------------
# Reader tests
# ---------------------------------------------------------------------------


def test_reader_drops_trailing_none_padding() -> None:
    # When
    fields = MMLUProXReader().read(_EVAL_ROW)
    # Then
    assert fields.raw_question == "Was ist die Hauptstadt von Frankreich?"
    assert fields.choices == ["Berlin", "Paris", "London", "Madrid", "Rom"]
    assert fields.correct_index == 1


def test_reader_drops_trailing_na_padding() -> None:
    # When
    fields = MMLUProXReader().read(_FEWSHOT_ROW)
    # Then
    assert fields.raw_question == "Was ist 1+1?"
    assert fields.choices == ["1", "2", "3"]
    assert fields.correct_index == 1


def test_reader_rejects_non_trailing_padding() -> None:
    # Given a malformed row with padding in the middle (would shift answer_index off the answer)
    row = {**_EVAL_ROW, "option_2": None}
    # When / Then
    with pytest.raises(AssertionError, match="Non-trailing"):
        MMLUProXReader().read(row)


def test_reader_rejects_answer_index_pointing_at_padding() -> None:
    # Given a malformed row whose answer_index points beyond the real options
    row = {**_EVAL_ROW, "answer_index": 7}
    # When / Then
    with pytest.raises(AssertionError, match="answer_index"):
        MMLUProXReader().read(row)


# ---------------------------------------------------------------------------
# Dataset loader tests
# ---------------------------------------------------------------------------


def test_loader_filters_both_splits_by_category() -> None:
    # Given a stub with rows from two categories in each split
    inner = DatasetStub(
        {
            "test": [_EVAL_ROW, {**_EVAL_ROW, "question_id": 2, "category": "law"}],
            "validation": [_FEWSHOT_ROW],
        }
    ).loader(None)
    loader = MMLUProXDatasetLoader("de", inner)
    # When
    dataset = loader.load("history")
    # Then only the "history" rows survive, in both splits
    assert [row["question_id"] for row in dataset["test"]] == [0]
    assert len(dataset["validation"]) == 1


def test_loader_requires_a_category() -> None:
    loader = MMLUProXDatasetLoader("de", DatasetStub({"test": [_EVAL_ROW]}).loader(None))
    with pytest.raises(AssertionError, match="category must be selected"):
        loader.load(None)


# ---------------------------------------------------------------------------
# Prompt spec: build the real benchmark over fictional rows, assert the assembled messages
# ---------------------------------------------------------------------------

_EVAL_Q = "Frage: Was ist die Hauptstadt von Frankreich?"

# The initial prompt depends on the variant (Cloze/BPB show no options, so their preamble must not
# claim multiple choice) and on the subject; the fictional rows are in "history" -> "Geschichte".
_MC_PREAMBLE = "Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über Geschichte."
_QA_PREAMBLE = "Die folgenden sind Fragen (mit Antworten) über Geschichte."


def _with_preamble(preamble: str, messages: list[Message]) -> list[Message]:
    """The initial prompt lands once, at the top of the prompt's first message."""
    first = messages[0]
    return [Message(role=first.role, content=f"{preamble}\n\n{first.content}"), *messages[1:]]


@dataclass(frozen=True)
class _ExpectedPrompt:
    messages: list[Message]
    ground_truth: str
    possible_completions: list[str]


# --- Zero-shot: the preamble sits on the eval question itself ---
_MC_EVAL_MESSAGES = [
    Message(role=Role.USER, content=f"{_EVAL_Q}\nA. Berlin\nB. Paris\nC. London\nD. Madrid\nE. Rom\n"),
    Message(role=Role.ASSISTANT, content="Antwort:"),
]
_MC = _ExpectedPrompt(
    messages=_with_preamble(_MC_PREAMBLE, _MC_EVAL_MESSAGES),
    ground_truth=" B",
    possible_completions=[" A", " B", " C", " D", " E"],
)
_MC_OLMES_EVAL_MESSAGES = [
    Message(role=Role.USER, content=f"{_EVAL_Q}\n A. Berlin\n B. Paris\n C. London\n D. Madrid\n E. Rom\n"),
    Message(role=Role.ASSISTANT, content="Antwort:"),
]
_MC_OLMES = _ExpectedPrompt(
    messages=_with_preamble(_MC_PREAMBLE, _MC_OLMES_EVAL_MESSAGES),
    ground_truth=" B",
    possible_completions=[" A", " B", " C", " D", " E"],
)
# Cloze/BPB show no options, so the assembled messages are identical; only the scored completions differ.
_CLOZE_EVAL_MESSAGES = [
    Message(role=Role.USER, content=f"{_EVAL_Q}\n"),
    Message(role=Role.ASSISTANT, content="Antwort:"),
]
_CLOZE = _ExpectedPrompt(
    messages=_with_preamble(_QA_PREAMBLE, _CLOZE_EVAL_MESSAGES),
    ground_truth=" Paris",
    possible_completions=[" Berlin", " Paris", " London", " Madrid", " Rom"],
)
_BPB = _ExpectedPrompt(
    messages=_CLOZE.messages,
    ground_truth=" Paris",
    possible_completions=[" Paris"],  # BPB scores only the gold continuation
)


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(mmlu_prox_lilab_de_mc, _MC, id="mc_de"),
        pytest.param(mmlu_prox_lilab_de_mc_olmes, _MC_OLMES, id="mc_olmes_de"),
        pytest.param(mmlu_prox_lilab_de_cloze, _CLOZE, id="cloze_de"),
        pytest.param(mmlu_prox_lilab_de_bpb, _BPB, id="bpb_de"),
    ],
)
def test_mmlu_prox_zeroshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"test": [_EVAL_ROW]}))
    sample = first_sample(benchmark, num_fewshot=0, custom_subjects=["history"])
    assert sample.subject == "history"
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


# --- One-shot: the preamble moves to the first fewshot example; the eval question stays bare ---
_MC_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content="Frage: Was ist 1+1?\nA. 1\nB. 2\nC. 3\n"),
    Message(role=Role.ASSISTANT, content="Antwort: B"),
]
_MC_OLMES_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content="Frage: Was ist 1+1?\n A. 1\n B. 2\n C. 3\n"),
    Message(role=Role.ASSISTANT, content="Antwort: B"),
]
_CLOZE_FEWSHOT_MESSAGES = [
    Message(role=Role.USER, content="Frage: Was ist 1+1?\n"),
    Message(role=Role.ASSISTANT, content="Antwort: 2"),
]


def _oneshot(
    preamble: str,
    fewshot_messages: list[Message],
    eval_messages: list[Message],
    zeroshot: _ExpectedPrompt,
) -> _ExpectedPrompt:
    return _ExpectedPrompt(
        messages=[*_with_preamble(preamble, fewshot_messages), *eval_messages],
        ground_truth=zeroshot.ground_truth,
        possible_completions=zeroshot.possible_completions,
    )


@pytest.mark.parametrize(
    "make_benchmark, expected",
    [
        pytest.param(
            mmlu_prox_lilab_de_mc,
            _oneshot(_MC_PREAMBLE, _MC_FEWSHOT_MESSAGES, _MC_EVAL_MESSAGES, _MC),
            id="mc_de",
        ),
        pytest.param(
            mmlu_prox_lilab_de_mc_olmes,
            _oneshot(_MC_PREAMBLE, _MC_OLMES_FEWSHOT_MESSAGES, _MC_OLMES_EVAL_MESSAGES, _MC_OLMES),
            id="mc_olmes_de",
        ),
        pytest.param(
            mmlu_prox_lilab_de_cloze,
            _oneshot(_QA_PREAMBLE, _CLOZE_FEWSHOT_MESSAGES, _CLOZE_EVAL_MESSAGES, _CLOZE),
            id="cloze_de",
        ),
        pytest.param(
            mmlu_prox_lilab_de_bpb,
            _oneshot(_QA_PREAMBLE, _CLOZE_FEWSHOT_MESSAGES, _CLOZE_EVAL_MESSAGES, _BPB),
            id="bpb_de",
        ),
    ],
)
def test_mmlu_prox_oneshot_prompt(make_benchmark: Callable[..., Benchmark], expected: _ExpectedPrompt) -> None:
    benchmark = make_benchmark(dataset=DatasetStub({"test": [_EVAL_ROW], "validation": [_FEWSHOT_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1, custom_subjects=["history"])
    assert sample.messages == expected.messages
    assert sample.ground_truth == expected.ground_truth
    assert sample.possible_completions == expected.possible_completions


def test_mc_de_oneshot_flat_prompt() -> None:
    """The complete 1-shot MC prompt as a single string, for visual inspection."""
    benchmark = mmlu_prox_lilab_de_mc(dataset=DatasetStub({"test": [_EVAL_ROW], "validation": [_FEWSHOT_ROW]}))
    sample = first_sample(benchmark, num_fewshot=1, custom_subjects=["history"])
    assert ConcatFormatter().format(sample.messages, output_mode="string") == (
        """Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über Geschichte.

Frage: Was ist 1+1?
A. 1
B. 2
C. 3
Antwort: B

Frage: Was ist die Hauptstadt von Frankreich?
A. Berlin
B. Paris
C. London
D. Madrid
E. Rom
Antwort:"""
    )


# ---------------------------------------------------------------------------
# CoT variant: prompt spec and answer extraction
# ---------------------------------------------------------------------------

_COT_PREAMBLE = (
    "Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über Geschichte. Denken Sie Schritt für Schritt "
    'nach und beenden Sie Ihre Antwort mit "Die Antwort ist (X)", wobei X der richtige Buchstabe ist.'
)
_COT_PREAMBLE_V2 = (
    "Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über Geschichte. Geben Sie eine KNAPPE Begründung "
    'und beenden Sie Ihre Antwort mit "Die Antwort ist (X)", wobei X der richtige Buchstabe ist.'
)
_COT_CUE = "Antwort: Denken wir Schritt für Schritt nach."
_COT_EVAL_Q = (
    "Frage:\nWas ist die Hauptstadt von Frankreich?\nOptionen:\nA. Berlin\nB. Paris\nC. London\nD. Madrid\nE. Rom\n"
)
_COT_EVAL_Q_V2 = (
    "Frage:\nWas ist die Hauptstadt von Frankreich?\nOptionen:\n"
    "(A) Berlin\n(B) Paris\n(C) London\n(D) Madrid\n(E) Rom\n"
)
_COT_FEWSHOT_Q = "Frage:\nWas ist 1+1?\nOptionen:\nA. 1\nB. 2\nC. 3\n"
_COT_FEWSHOT_TARGET = f"{_COT_CUE}Eins plus eins ergibt zwei.Die Antwort ist (B)"

_COT_STUB_SPLITS = {"test": [_EVAL_ROW], "validation": [_FEWSHOT_ROW]}


def test_cot_zeroshot_prompt() -> None:
    # Given one test row
    dataset = DatasetStub(_COT_STUB_SPLITS)
    # When the first sample is assembled
    sample = first_sample(mmlu_prox_lilab_de_cot(dataset=dataset), num_fewshot=0, custom_subjects=["history"])
    # Then the preamble tops the question and the cue primes the reasoning as an assistant turn
    assert sample.messages == [
        Message(role=Role.USER, content=f"{_COT_PREAMBLE}\n\n{_COT_EVAL_Q}"),
        Message(role=Role.ASSISTANT, content=_COT_CUE),
    ]
    assert ConcatFormatter().format(sample.messages, output_mode="string") == (
        f"{_COT_PREAMBLE}\n\n{_COT_EVAL_Q}{_COT_CUE}"
    )
    assert sample.ground_truth == "B"


def test_cot_oneshot_prompt() -> None:
    """Apart from the preamble wording and the blank line after it, the flat string is lm-eval's."""
    # Given one row per split
    dataset = DatasetStub(_COT_STUB_SPLITS)
    # When the first sample is assembled with one demonstration
    sample = first_sample(mmlu_prox_lilab_de_cot(dataset=dataset), num_fewshot=1, custom_subjects=["history"])
    # Then the demonstration answers with its worked reasoning, the cue replacing the dataset's "A:" prefix
    assert sample.messages == [
        Message(role=Role.USER, content=f"{_COT_PREAMBLE}\n\n{_COT_FEWSHOT_Q}"),
        Message(role=Role.ASSISTANT, content=_COT_FEWSHOT_TARGET),
        Message(role=Role.USER, content=_COT_EVAL_Q),
        Message(role=Role.ASSISTANT, content=_COT_CUE),
    ]
    assert ConcatFormatter().format(sample.messages, output_mode="string") == (
        """Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über Geschichte. Denken Sie Schritt für Schritt nach und beenden Sie Ihre Antwort mit "Die Antwort ist (X)", wobei X der richtige Buchstabe ist.

Frage:
Was ist 1+1?
Optionen:
A. 1
B. 2
C. 3
Antwort: Denken wir Schritt für Schritt nach.Eins plus eins ergibt zwei.Die Antwort ist (B)

Frage:
Was ist die Hauptstadt von Frankreich?
Optionen:
A. Berlin
B. Paris
C. London
D. Madrid
E. Rom
Antwort: Denken wir Schritt für Schritt nach."""  # noqa: E501 (lm-eval-shaped flat prompt)
    )
    assert sample.ground_truth == "B"


def _extracted(benchmark: Benchmark, model_output: str) -> str:
    evaluation = benchmark.create(num_fewshot=0, custom_subjects=["history"], custom_hf_revision=None)
    sample = next(iter(evaluation.iterate_samples(1)))
    (completion,) = evaluation.generate_completions(StubLlm(model_output), [sample])
    return completion.completion


@pytest.mark.parametrize(
    "completion, expected",
    [
        pytest.param("Paris ist die Hauptstadt. Die Antwort ist (B).", "B", id="parenthesized"),
        pytest.param("Die Antwort ist B", "B", id="bare_letter"),
        pytest.param("Die Antwort ist (B\n\nFrage:\nNoch eine? Die Antwort ist (C)", "B", id="unclosed_parenthesis"),
        pytest.param("Die Antwort ist (B). Oder doch: Die Antwort ist (C).", "B", id="first_match_wins"),
        pytest.param("Ich weiß es nicht.", "[invalid]", id="no_match"),
        pytest.param("die antwort ist (b)", "[invalid]", id="case_sensitive_like_lm_eval"),
    ],
)
def test_cot_answer_extraction(completion: str, expected: str) -> None:
    benchmark = mmlu_prox_lilab_de_cot(dataset=DatasetStub(_COT_STUB_SPLITS))
    assert _extracted(benchmark, completion) == expected


def test_cot_v2_zeroshot_prompt_has_no_assistant_cue() -> None:
    # Given one test row
    dataset = DatasetStub(_COT_STUB_SPLITS)
    # When the first sample is assembled
    sample = first_sample(mmlu_prox_lilab_de_cot_v2(dataset=dataset), num_fewshot=0, custom_subjects=["history"])
    # Then the prompt is a single user message with "(A)"-labelled options; the model opens its own turn
    assert sample.messages == [Message(role=Role.USER, content=f"{_COT_PREAMBLE_V2}\n\n{_COT_EVAL_Q_V2}")]
    # ConcatFormatter strips the trailing newline
    assert ConcatFormatter().format(sample.messages, output_mode="string") == (
        f"{_COT_PREAMBLE_V2}\n\n{_COT_EVAL_Q_V2}".rstrip()
    )
    assert sample.ground_truth == "B"


@pytest.mark.parametrize(
    "completion, expected",
    [
        pytest.param("Paris ist die Hauptstadt. Die Antwort ist (B).", "B", id="parenthesized"),
        pytest.param("Die Antwort ist B", "B", id="bare_letter"),
        pytest.param("Die Antwort ist: (B)", "B", id="colon"),
        pytest.param("die antwort ist (b)", "B", id="case_insensitive"),
        pytest.param("Die Antwort ist (B). Oder doch: Die Antwort ist (C).", "C", id="last_match_wins"),
        pytest.param("Therefore, the answer is (B)", "B", id="english_phrase"),
        pytest.param("Die Antwort ist eindeutig Paris.", "[invalid]", id="word_not_letter"),
        pytest.param("Ich weiß es nicht.", "[invalid]", id="no_match"),
    ],
)
def test_cot_v2_answer_extraction(completion: str, expected: str) -> None:
    benchmark = mmlu_prox_lilab_de_cot_v2(dataset=DatasetStub(_COT_STUB_SPLITS))
    assert _extracted(benchmark, completion) == expected
