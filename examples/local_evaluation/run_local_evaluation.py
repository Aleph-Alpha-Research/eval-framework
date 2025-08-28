from pathlib import Path

from eval_framework.llm.huggingface_llm import HFLLM
from eval_framework.main import main
from eval_framework.task_loader import load_extra_tasks
from eval_framework.tasks.eval_config import EvalConfig
from template_formatting.formatter import HFFormatter


# Define your model if needed
class MyHuggingFaceModel(HFLLM):
    LLM_NAME = "microsoft/DialoGPT-medium"
    DEFAULT_FORMATTER = HFFormatter("microsoft/DialoGPT-medium")


if __name__ == "__main__":
    # Initialize your model
    llm = MyHuggingFaceModel()

    # Using Llama 3.1 8B Instruct hosted on Aleph Alpha Research API
    # This requires an AA_TOKEN to be set in the .env file
    # llm = Llama31_8B_Instruct_API()

    # Load custom tasks from the specified directory or file:
    load_extra_tasks(
        ["/home/gregory.schott/repos/eval-framework-companion-bis/src/eval_framework_companion/tasks/benchmarks"]
    )

    # Running evaluation on COPA task using 2 few-shot examples and 5 samples
    config = EvalConfig(
        task_name="COPA",
        num_fewshot=2,
        num_samples=1,
        output_dir=Path("eval-results"),
        llm_class=MyHuggingFaceModel,
        # llm_class=Llama31_8B_Instruct_API,
    )

    # Generating samples, running evaluation and aggregating results
    results = main(llm=llm, config=config)
