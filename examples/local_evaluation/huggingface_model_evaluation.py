from pathlib import Path

from eval_framework.llm.huggingface import HFLLM
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig
from template_formatting.formatter import HFFormatter


# Define your model
class MyHuggingFaceModel(HFLLM):
    LLM_NAME = "microsoft/DialoGPT-medium"
    DEFAULT_FORMATTER = HFFormatter("microsoft/DialoGPT-medium")


if __name__ == "__main__":
    # Initialize your model
    llm = MyHuggingFaceModel()

    # Configure evaluation
    config = EvalConfig(
        task_name="ARC",
        num_fewshot=3,
        num_samples=100,
        output_dir=Path("./eval_results"),
        llm_class=MyHuggingFaceModel,
    )

    # Run evaluation
    results = main(llm=llm, config=config)
