from pathlib import Path
from typing import Any

from datasets import load_dataset

from eval_framework.metrics.completion.bleu import BLEU
from eval_framework.metrics.completion.chrf import CHRF
from eval_framework.metrics.completion.ter import TER
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample


class EnglishToGermanTranslation(BaseTask[str]):
    NAME = "EnglishToGermanTranslation"
    DATASET_PATH = "en_de_translation"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [BLEU, CHRF, TER]
    SUBJECTS = ["main"]
    LANGUAGE = {"main": (Language.ENG, Language.DEU)}

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["\n\n", "\nEnglish phrase:", "English phrase:"]
        self.max_tokens = 150

    def _load_hf_dataset(self, **kwargs: Any) -> Any:
        """Load translation pairs from local JSONL files."""
        current_dir = Path(__file__).parent
        data_dir = current_dir / "data" / "en_de_translation"
        
        train_file = data_dir / "en_de_translation_train.jsonl"
        test_file = data_dir / "en_de_translation_test.jsonl"
        
        if not train_file.exists() or not test_file.exists():
            raise FileNotFoundError(
                f"Dataset files not found!\n"
                f"Expected:\n  - {train_file}\n  - {test_file}"
            )
        
        return load_dataset(
            "json",
            data_files={"train": str(train_file), "test": str(test_file)},
        )

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        """Format the translation prompt."""
        return f"English phrase: {item['source']}\nGerman phrase:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str:
        """Return the reference German translation."""
        return item["target"]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        """Format few-shot example targets."""
        return f" {item['target']}"

    def post_process_generated_completion(
        self, completion_text: str, sample: Sample | None = None
    ) -> str:
        """Clean the model output before evaluation."""
        for stop_seq in self.stop_sequences:
            if stop_seq in completion_text:
                completion_text = completion_text.split(stop_seq)[0]
        return completion_text.strip()