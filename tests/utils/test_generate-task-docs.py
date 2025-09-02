from pathlib import Path

# from eval_framework.utils.generate_task_docs import generate_task_docs


def test_task_docs_are_up_to_date(tmp_path: Path) -> None:
    """
    Test that all tasks docs have been generated and are up to date. If not they are updated locally and the test fails
    on the CI until it gets commited. Checks that no documentation still remain from removed tasks.
    """
    raise NotImplementedError


def test_generate_task_docs(tmp_path: Path) -> None:
    """
    Test that the task documentation generation script runs without errors on a dummy task class.
    """
    raise NotImplementedError
