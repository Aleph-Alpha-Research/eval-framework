from jinja2 import Template
import transformers

def test_transformers_import() -> None:
    # Check that the transformers version is available
    version = transformers.__version__
    assert version is not None

def test_jinja2_import() -> None:
    # Simple test to ensure Jinja2 is importable and functional
    template = Template("Hello {{ name }}!")
    rendered = template.render(name="World")
    assert rendered == "Hello World!"

def main() -> None:
    test_jinja2_import()
    test_transformers_import()

if __name__ == "__main__":
    main()