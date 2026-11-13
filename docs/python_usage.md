# Python Usage

The CLI is a thin wrapper around `eval_framework.main.main`. Calling it from Python gives access to everything the
CLI cannot express, such as a custom `BaseLLM` implementation or a prompt formatter for loglikelihood tasks.

```python
from pathlib import Path

from eval_framework.llm.openai import OpenAIModel
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig

config = EvalConfig(
    llm_class=OpenAIModel,
    llm_args={"model_name": "gpt-4o-mini"},  # plus e.g. "base_url" for any OpenAI-compatible endpoint
    task_name="MMLU",
    task_subjects=["abstract_algebra"],
    num_fewshot=5,
    num_samples=10,  # per subject; None evaluates the whole split
    output_dir=Path("outputs"),
)

results = main(llm=config.llm_class(**config.llm_args), config=config)
```

`llm_args` takes every keyword argument of the model class; see the [Model Arguments guide](model_arguments.md).
