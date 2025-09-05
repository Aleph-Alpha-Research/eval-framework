from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample


class PAWSX(BaseTask[str]):
    """PAWSX dataset: https://huggingface.co/datasets/google-research-datasets/paws-x
    used in the way suggested in PARAPHRASUS benchmark (https://arxiv.org/pdf/2409.12060)."""

    NAME = "PAWS-X"
    DATASET_PATH = "google-research-datasets/paws-x"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.COMPLETION  # LOGLIKELIHOODS would also make sense but staying true to PARAPHRASUS
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["en", "de"]  # ["es", "fr", "ja", "ko", "zh"] -- disabled as irrelevant for the time being
    PERTURBATION_UNMODIFIABLE_WORDS = ["Ja", "Nein", "Paraphrasen", "Yes", "No", "paraphrases"]
    LANGUAGE = {"en": Language.ENG, "de": Language.DEU}

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        self.num_fewshot = num_fewshot

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        # PARAPHRASUS seems to use English prompt for all languages but that's a bit weird, let's do it properly.
        match item["subject"]:
            case "de":
                return (
                    "Sind die folgenden Sätze Paraphrasen?\n"
                    f"Satz 1: {item['sentence1']}\n"
                    f"Satz 2: {item['sentence2']}\n"
                    "Antworte mit 'Ja' oder 'Nein'.\n"
                )
            case _:
                # Please translate to other language as necessary
                return (
                    "Are the following sentences paraphrases?\n"
                    f"Sentence 1: {item['sentence1']}\n"
                    f"Sentence 2: {item['sentence2']}\n"
                    "Answer with 'Yes' or 'No'.\n"
                )

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        match item["subject"]:
            case "de":
                return "Ja" if item["label"] == "1" else "Nein"
            case _:
                # Please translate to other language as necessary
                return "Yes" if item["label"] == "1" else "No"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        return completion_text.strip().strip("\"'.")

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        # Note that this, together with BaseTask._get_messages(), produces a different prompt structure than
        # what PARAPHRASUS suggests in Figure 4. But both seem approaches are somehow valid...
        examples: list[dict] = []
        for _ in range(1000):
            example = self.rnd.choice(self.dataset[self.FEWSHOT_SPLIT])
            # Ensure half of the examples is negative and half positive.
            if example["label"] == (len(examples) % 2) and example not in examples:
                examples.append(example)
            if len(examples) >= self.num_fewshot:
                break
        return examples
