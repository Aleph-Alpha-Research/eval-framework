def test_optional_import() -> None:
    import transformers

    assert transformers.__version__ is not None
    from jinja2 import Template

    assert Template is not None
