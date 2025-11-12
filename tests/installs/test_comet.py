def test_comet_import() -> None:
    from eval_framework.tasks.task_names import registered_tasks_iter

    # Just iterate through registered tasks to ensure they are importable
    for _ in registered_tasks_iter():
        pass
