from eval_framework.llm.base import BaseLLM
from eval_framework.tasks.base import BaseTask
from template_formatting.formatter import BaseFormatter


def test_core_import() -> None:
    # Just instantiate core components to ensure they are importable
    assert BaseLLM is not None
    assert BaseFormatter is not None
    assert BaseTask is not None


def main() -> None:
    test_core_import()


if __name__ == "__main__":
    main()
