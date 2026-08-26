"""Tests for the Subjects selectors: what ``--task-subjects`` selectors resolve to."""

import pytest

from eval_framework.subjects import ListOfSubjects, NoSubject, Subject


def test_no_subject_selects_the_single_unnamed_slice() -> None:
    # A task with no subjects: no selection yields one slice with no config to load
    assert NoSubject().select([]) == (Subject(load_key=None, label="no_subject"),)


def test_no_subject_wildcard_yields_the_single_slice() -> None:
    # "*" means "all", which for a subjectless task is the one slice
    assert NoSubject().select(["*"]) == (Subject(load_key=None, label="no_subject"),)


def test_no_subject_rejects_a_specific_selector() -> None:
    # A no-subject task cannot select a named subject
    with pytest.raises(ValueError, match="no subjects"):
        NoSubject().select(["deu"])


def test_list_of_subjects_no_selection_yields_all_in_declared_order() -> None:
    assert [s.label for s in ListOfSubjects(["a", "b", "c"]).select([])] == ["a", "b", "c"]


def test_list_of_subjects_each_name_is_both_load_key_and_label() -> None:
    (subject,) = ListOfSubjects(["a"]).select([])
    assert subject == Subject(load_key="a", label="a")


def test_list_of_subjects_wildcard_yields_all() -> None:
    assert [s.label for s in ListOfSubjects(["a", "b"]).select(["*"])] == ["a", "b"]


def test_list_of_subjects_selection_follows_declared_order_and_dedupes() -> None:
    # Tokens in any order, repeated, are resolved to the task's declared order, once each
    assert [s.label for s in ListOfSubjects(["a", "b", "c"]).select(["c", "a", "a"])] == ["a", "c"]


def test_list_of_subjects_rejects_an_unknown_selector() -> None:
    with pytest.raises(ValueError, match="Unknown subject"):
        ListOfSubjects(["a", "b"]).select(["x"])
