from pathlib import Path

from eval_framework.llm.openai import OpenAIModel
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.task_loader import load_extra_tasks

SCRIPT_DIR = Path(__file__).parent.resolve()

if __name__ == "__main__":
    # Using an OpenAI model; any OpenAI-compatible server can be targeted by passing base_url
    # This requires an OPENAI_API_KEY to be set in the .env file
    llm_args = {"model_name": "gpt-4o-mini"}
    llm = OpenAIModel(**llm_args)

    # Load custom tasks from the specified directory or file:
    load_extra_tasks(SCRIPT_DIR / "custom_tasks")

    # Running evaluation on COPA task using 2 few-shot examples and 5 samples
    config = EvalConfig(
        task_name="COPA",
        num_fewshot=2,
        num_samples=5,
        output_dir=Path("eval-results"),
        llm_class=OpenAIModel,
        llm_args=llm_args,
    )

    # Generating samples, running evaluation and aggregating results
    results = main(llm=llm, config=config)
