from typing import Any

from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType

PALOMA_SOURCES = [
    "4chan_meta_sep",
    "c4_100_domains",
    "c4_en",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "dolma-v1_5",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
]

class GenericPaloma(BaseTask[str]):
    """
    Paloma perplexity benchmark over many domains.

    This mirrors oe_eval's GenericPaloma task and uses the allenai/paloma dataset.
    We score the full document as a single completion: the model is asked for
    log p(document), and BitsPerByteLoglikelihood computes -log p(text) / bytes(text),
    i.e. bits per byte, matching oe_eval's primary_metric "bits_per_byte".

    Note: This dataset is gated on Hugging Face.
    """

    NAME = "Paloma"
    DATASET_PATH = "allenai/paloma"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [BitsPerByteLoglikelihood]
    SUBJECTS = PALOMA_SOURCES
    PERTURBATION_UNMODIFIABLE_WORDS: list[str] | None = None
    LANGUAGE = Language.ENG

    def _load_dataset(self, subject: str) -> None:  # type: ignore[override]
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH)
        self.dataset = {"validation": list(hf_dataset["validation"])}

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return ""

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return item["text"]

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [item["text"]]

