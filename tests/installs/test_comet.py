from eval_framework.tasks.task_names import registered_tasks_iter


def test_comet_import() -> None:
    # Just iterate through registered tasks to ensure they are importable
    for _ in registered_tasks_iter():
        pass


def main() -> None:
    test_comet_import()


if __name__ == "__main__":
    main()
