from enum import Enum
from typing import Any

import iso639

RANDOM_SEED = 42


class TaskStyle(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    CLOZE = "cloze"
    BPB = "bpb"


class Language(Enum):
    ENG = "English"
    DEU = "German"
    FRA = "French"
    ITA = "Italian"
    SPA = "Spanish"
    POR = "Portuguese"
    NLD = "Dutch"
    CES = "Czech"
    DAN = "Danish"
    FIN = "Finnish"
    SWE = "Swedish"
    ARB = "Arabic"
    POL = "Polish"
    RUS = "Russian"
    UKR = "Ukrainian"
    HRV = "Croatian"
    SRP = "Serbian"

    @classmethod
    def add_members(cls, new_members: dict[str, Any]) -> type["Language"]:
        members = {member.name: member.value for member in cls}
        for name, value in new_members.items():
            if name not in members:
                members[name] = value
        return Enum(cls.__name__, members)  # type: ignore[return-value]


languages: dict[str, str] = {}
for language in iso639.ALL_LANGUAGES:
    enum_name = language.part3.upper()
    languages[enum_name] = language.name

Language: type[Enum] = Language.add_members(languages)  # type: ignore[no-redef]


def _subject_parts(subject: object) -> tuple[str, ...]:
    """A subject as its stringified parts: tuple subjects keep their fields, scalars are a single part.

    Comparing string forms means tuple fields of any type work (e.g. tuple[str, int, str]) without
    having to parse the CLI token into each field's native type.
    """
    return tuple(str(field) for field in subject) if isinstance(subject, tuple) else (str(subject),)


def resolve_overwrite_subjects[SubjectType](
    custom_subjects: list[str], accepted_subjects: list[SubjectType], task_name: str
) -> list[SubjectType]:
    """Restrict `accepted_subjects` to the ones requested via --task-subjects.

    A subject matches a token when their parts agree position by position, where "*" matches any one
    part -- so "DE_DE,*" selects every German subject and "*" selects every scalar subject.
    """
    if not accepted_subjects:
        raise ValueError(f"Task {task_name} has no SUBJECTS defined")

    filters = {token: tuple(part.strip() for part in token.split(",")) for token in custom_subjects}

    # Select and validate in one pass. Every filter is tried against every subject (no early exit), so
    # a filter still counts as used when its subjects were already selected by an earlier filter --
    # e.g. "a,*" after "a,1" is valid, not unused.
    chosen_subjects: list[SubjectType] = []
    used_filters: set[str] = set()
    for subject in accepted_subjects:
        fields = _subject_parts(subject)
        matching = [
            token
            for token, parts in filters.items()
            if len(parts) == len(fields) and all(p in ("*", f) for p, f in zip(parts, fields))
        ]
        if matching:
            chosen_subjects.append(subject)
            used_filters.update(matching)

    for token in custom_subjects:
        if token not in used_filters:
            raise ValueError(
                f"Subject '{token}' not found in task {task_name}. Subjects are matched by their "
                f"string form, so check number and enum formatting."
            )

    return chosen_subjects
