def test_accelerate_import() -> None:
    import accelerate

    assert accelerate.__version__ is not None
