from pathlib import Path

from eval_framework.llm.models import Llama31_8B_Instruct_API
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig

# from eval_framework.tasks.task_loader import load_extra_tasks

if __name__ == "__main__":
    # Using Llama 3.1 8B Instruct hosted on Aleph Alpha Research API
    # This requires an AA_TOKEN to be set in the .env file
    llm = Llama31_8B_Instruct_API()

    # Load custom tasks from the specified directory or file:
    # load_extra_tasks(["examples/custom_tasks/"])

    # Running evaluation on COPA task using 2 few-shot examples and 5 samples
    config = EvalConfig(
        task_name="COPA",
        num_fewshot=2,
        num_samples=5,
        output_dir=Path("eval-results"),
        llm_class=Llama31_8B_Instruct_API,
    )

    # Generating samples, running evaluation and aggregating results
    results = main(llm=llm, config=config)
