"""
Example script to run the English to German Translation benchmark.

This demonstrates how to evaluate a model on EN→DE translation.
"""

from pathlib import Path
from functools import partial

from eval_framework.llm.huggingface import HFLLM
from eval_framework.main import main
from eval_framework.tasks.eval_config import EvalConfig
from eval_framework.tasks.registry import register_task
from template_formatting.formatter import HFFormatter

# Import your custom task
from en_de_translation import EnglishToGermanTranslation

# Register the task
register_task(EnglishToGermanTranslation)


# Define your model
class MyHuggingFaceModel(HFLLM):
    LLM_NAME = "HuggingFaceTB/SmolLM-360M-Instruct"
    DEFAULT_FORMATTER = partial(HFFormatter, "HuggingFaceTB/SmolLM-360M-Instruct")


if __name__ == "__main__":
    # Initialize your model
    llm = MyHuggingFaceModel()

    # Configure evaluation
    config = EvalConfig(
        output_dir=Path("./eval_results_en_de_translation__NEW_llama"),
        num_fewshot=3,  # Use 3 translation examples
        num_samples=30,  # Evaluate on 30 test samples
        task_name="EnglishToGermanTranslation",
        llm_class=MyHuggingFaceModel,
    )

    # Run evaluation and get results
    print("Starting English to German translation evaluation...")
    results = main(llm=llm, config=config)
    print(f"\nResults saved to: {config.output_dir}")
    print(f"\nBLEU Score and other metrics: {results}")