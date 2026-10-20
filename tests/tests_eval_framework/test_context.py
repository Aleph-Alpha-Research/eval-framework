from pathlib import Path

import pytest

from eval_framework.context.local import LocalContext
from eval_framework.llm.openai import OpenAIModel


def test_local_context() -> None:
    with LocalContext(
        num_samples=10,
        num_fewshot=0,
        task_name="ARC",
        output_dir=Path("dummy"),
        hf_upload_dir="dummy22",
        llm_args={"model_name": "m"},
        judge_model_args={"model_name": "judge"},
        batch_size=1,
        repeats=1,
    ) as ctx:
        assert ctx is not None
        assert ctx.config is not None
        assert ctx.config.llm_class.__name__ == OpenAIModel.__name__
        assert ctx.config.num_samples == 10
        assert ctx.config.num_fewshot == 0
        assert ctx.config.task_name == "ARC"
        assert ctx.config.output_dir == Path("dummy")
        assert ctx.config.hf_upload_dir == "dummy22"
        assert ctx.config.llm_args == {"model_name": "m"}
        assert ctx.config.llm_judge_class is not None
        assert ctx.config.llm_judge_class.__name__ == OpenAIModel.__name__
        assert ctx.config.judge_model_args is not None
        assert ctx.config.repeats == 1


def test_local_context_without_judge_model_args() -> None:
    with LocalContext(
        task_name="ARC", num_fewshot=0, batch_size=1, output_dir=Path("dummy"), llm_args={"model_name": "m"}
    ) as ctx:
        assert ctx.config is not None
        assert ctx.config.llm_judge_class is None


@pytest.mark.parametrize(
    ("llm_args", "judge_model_args", "match"),
    [
        ({}, {}, "llm_args must set model_name"),
        ({"model_name": "m"}, {"temperature": "0"}, "judge_model_args must set model_name"),
    ],
)
def test_local_context_requires_model_names(
    llm_args: dict[str, str], judge_model_args: dict[str, str], match: str
) -> None:
    context = LocalContext(task_name="ARC", llm_args=llm_args, judge_model_args=judge_model_args)
    with pytest.raises(ValueError, match=match):
        context.__enter__()
