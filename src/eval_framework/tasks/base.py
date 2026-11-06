# The companion still imports ``Language`` from here; the alias form keeps it a re-export
# under ``no_implicit_reexport``.
from eval_framework.tasks.task_style import Language as Language

RANDOM_SEED = 42


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
