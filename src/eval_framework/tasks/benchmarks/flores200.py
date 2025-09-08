from typing import Any

import pycountry

from eval_framework.metrics.completion.bleu import BLEU
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample

FLORES_LANGUAGES = [
    "deu_Latn",
    "eng_Latn",
    "fin_Latn",
    "fra_Latn",
    "nld_Latn",
]  # Note: there are many more languages in the dataset, but we only consider these for now


class Flores200(BaseTask[str]):
    """QMSum dataset: https://huggingface.co/datasets/facebook/flores"""

    NAME = "FLoRes-200"
    DATASET_PATH = "facebook/flores"
    SAMPLE_SPLIT = "devtest"
    FEWSHOT_SPLIT = "dev"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [BLEU]
    SUBJECTS = [f"{s}-{t}" for s in FLORES_LANGUAGES for t in FLORES_LANGUAGES if s != t]
    PERTURBATION_UNMODIFIABLE_WORDS = ["sentence"]
    LANGUAGE = {
        "deu_Latn": Language.DEU,
        "eng_Latn": Language.ENG,
        "fin_Latn": Language.FIN,
        "fra_Latn": Language.FRA,
        "nld_Latn": Language.NLD,
    }

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

        self.stop_sequences = ["\n"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        source_key = item["subject"].split("-")[0]
        source_language = pycountry.languages.get(alpha_3=source_key.split("_")[0]).name
        source = item[f"sentence_{source_key}"]
        instruction = f"{source_language} sentence: {source}\n"
        target_key = item["subject"].split("-")[1]
        target_language = pycountry.languages.get(alpha_3=target_key.split("_")[0]).name

        return f"{instruction}{target_language} sentence:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        target_key = item["subject"].split("-")[1]
        return item[f"sentence_{target_key}"]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = f" {self._get_ground_truth(item)}"
        assert target is not None
        assert isinstance(target, str)
        return target

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        return completion_text.strip()
