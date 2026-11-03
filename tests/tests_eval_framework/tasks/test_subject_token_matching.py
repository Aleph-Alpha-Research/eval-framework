import pytest

from eval_framework.tasks.base import resolve_overwrite_subjects

# "No tokens = all subjects" is the callers' short-circuit (e.g. ``Benchmark.subjects``), so every
# row here carries at least one token.


@pytest.mark.parametrize(
    "subjects,custom_subjects,expected_value",
    [
        (["subject1", "subject2", "subject3"], ["subject1", "subject3"], ["subject1", "subject3"]),
        # result follows SUBJECTS' declared order, not the CLI argument order, and dedupes repeats --
        # matches come from `accepted_subjects`, not from echoing `custom_subjects` back verbatim.
        (["subject1", "subject2", "subject3"], ["subject3", "subject1"], ["subject1", "subject3"]),
        (["subject1", "subject2"], ["subject1", "subject1"], ["subject1"]),
        # "*" matches any scalar subject too, consistent with "*" already being a wildcard position
        # within a tuple subject. Redundant with passing no tokens for the top-level case, but
        # the matching function treats scalar and tuple subjects the same way, so this falls out for free.
        (["subject1", "subject2", "subject3"], ["*"], ["subject1", "subject2", "subject3"]),
        ([("EN_US", "topic1"), ("EN_US", "topic2"), ("DE_DE", "topic1")], ["EN_US,topic1"], [("EN_US", "topic1")]),
        (
            [("EN_US", "topic1"), ("EN_US", "topic2"), ("DE_DE", "topic1")],
            ["EN_US,*"],
            [("EN_US", "topic1"), ("EN_US", "topic2")],
        ),
        (
            [
                ("EN_US", "topic1", "subtopic1"),
                ("EN_US", "topic1", "subtopic2"),
                ("EN_US", "topic2", "subtopic1"),
                ("DE_DE", "topic1", "subtopic1"),
            ],
            ["EN_US,topic1,*"],
            [("EN_US", "topic1", "subtopic1"), ("EN_US", "topic1", "subtopic2")],
        ),
        (
            [
                ("EN_US", "topic1", "subtopic1"),
                ("EN_US", "topic1", "subtopic2"),
                ("EN_US", "topic2", "subtopic1"),
                ("DE_DE", "topic1", "subtopic1"),
            ],
            ["*,topic1,*"],
            [
                ("EN_US", "topic1", "subtopic1"),
                ("EN_US", "topic1", "subtopic2"),
                ("DE_DE", "topic1", "subtopic1"),
            ],
        ),
        (
            [("EN_US", "topic1"), ("EN_US", "topic2"), ("DE_DE", "topic1")],
            ["EN_US,topic1", "DE_DE,topic1"],
            [("EN_US", "topic1"), ("DE_DE", "topic1")],
        ),
        # mixed-type tuple subjects, tuple[str, int, str]
        (
            [("ctx1", 4096, "single"), ("ctx1", 8192, "multi"), ("ctx2", 4096, "single")],
            ["ctx1,4096,single"],
            [("ctx1", 4096, "single")],
        ),
        (
            [("ctx1", 4096, "single"), ("ctx1", 8192, "multi"), ("ctx2", 4096, "single")],
            ["ctx1,*,*"],
            [("ctx1", 4096, "single"), ("ctx1", 8192, "multi")],
        ),
        (["subject1", "subject2"], ["invalid_subject"], "ValueError"),
        ([("EN_US", "topic1"), ("EN_US", "topic2")], ["EN_US,invalid_topic"], "ValueError"),
        ([("ctx1", 4096, "single"), ("ctx1", 8192, "multi")], ["ctx1,9999,single"], "ValueError"),
        # matching compares stringified subject fields, not a parsed native value, so a non-numeric
        # part at an int position is just another "not a legal value" case, not a separate parse error.
        ([("ctx1", 4096, "single"), ("ctx1", 8192, "multi")], ["ctx1,abc,single"], "ValueError"),
    ],
)
def test_subject_tokens_select_subjects_part_by_part(
    subjects: list[str] | list[tuple], custom_subjects: list[str], expected_value: list[str] | list[tuple] | str
) -> None:
    if expected_value == "ValueError":
        with pytest.raises(ValueError):
            resolve_overwrite_subjects(custom_subjects, subjects, "MyTask")
    else:
        assert resolve_overwrite_subjects(custom_subjects, subjects, "MyTask") == expected_value


def test_a_task_without_subjects_is_rejected() -> None:
    with pytest.raises(ValueError, match="no SUBJECTS"):
        resolve_overwrite_subjects(["anything"], [], "MyTask")
