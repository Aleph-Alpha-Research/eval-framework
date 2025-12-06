def test_determined_import() -> None:
    from eval_framework.context.determined import DeterminedContext

    assert DeterminedContext is not None
