"""Tests for the choice readers — what ``EllaMindReader`` extracts from a raw item."""

from typing import Any, Literal

from eval_framework.choices import EllaMindReader

# A synthetic ellamind-shaped row (NOT a real dataset). Field names are arbitrary here to prove the
# reader uses its configured fields rather than hardcoded ones.
_PLURAL_ROW: dict[str, Any] = {
    "q": "Was essen Pandas am liebsten?",
    "answer": "Bambus",
    "easy": ["Pizza", "Eis", "Schokolade"],
    "hard": ["Blätter", "Gräser", "Kräuter"],
}


def _plural_reader(level: Literal["easy", "hard"]) -> EllaMindReader:
    return EllaMindReader(
        level, question_field="q", correct_field="answer", easy_field="easy", hard_field="hard", singular=False
    )


def test_ellamind_reader_reads_configured_fields_and_shuffles_correct_among_distractors() -> None:
    # When reading with the easy distractor set
    fields = _plural_reader("easy").read(_PLURAL_ROW)

    # Then the raw question comes from the configured field, the correct answer is present and
    # correct_index points to it, and the choices are exactly the correct answer + its distractors
    assert fields.raw_question == "Was essen Pandas am liebsten?"
    assert fields.choices[fields.correct_index] == "Bambus"
    assert set(fields.choices) == {"Bambus", "Pizza", "Eis", "Schokolade"}


def test_ellamind_reader_is_deterministic_for_identical_input() -> None:
    assert _plural_reader("easy").read(_PLURAL_ROW) == _plural_reader("easy").read(_PLURAL_ROW)


def test_ellamind_reader_easy_and_hard_draw_from_different_distractor_sets() -> None:
    easy = _plural_reader("easy").read(_PLURAL_ROW)
    hard = _plural_reader("hard").read(_PLURAL_ROW)
    assert set(easy.choices) == {"Bambus", "Pizza", "Eis", "Schokolade"}
    assert set(hard.choices) == {"Bambus", "Blätter", "Gräser", "Kräuter"}


def test_ellamind_reader_wraps_a_singular_distractor_field() -> None:
    # Some datasets give a single distractor per level (a string, not a list); singular=True wraps it
    row: dict[str, Any] = {"q": "x", "answer": "correct", "easy": "the-distractor", "hard": "other"}
    reader = EllaMindReader(
        "easy", question_field="q", correct_field="answer", easy_field="easy", hard_field="hard", singular=True
    )
    fields = reader.read(row)
    assert set(fields.choices) == {"correct", "the-distractor"}
    assert fields.choices[fields.correct_index] == "correct"
