import random
from itertools import product
from typing import Any

from eval_framework.metrics.completion.bleu import BLEU
from eval_framework.metrics.completion.chrf import CHRF
from eval_framework.metrics.completion.comet import COMET
from eval_framework.shared.types import BaseMetricContext, UntemplatedPrompt
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample

LANG_MAP = {
    "deu_Latn": "German",
    "eng_Latn": "English",
    "fra_Latn": "French",
    "ita_Latn": "Italian",
    "nld_Latn": "Dutch",
    "pol_Latn": "Polish",
    "rus_Cyrl": "Russian",
    "spa_Latn": "Spanish",
    "ukr_Cyrl": "Ukrainian",
}


class FloresPlus(BaseTask[str]):
    """Flores-Plus dataset: https://huggingface.co/datasets/openlanguagedata/flores_plus"""

    NAME = "Flores-Plus"
    DATASET_PATH = "openlanguagedata/flores_plus"
    SAMPLE_SPLIT = "dev"
    FEWSHOT_SPLIT = "devtest"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [BLEU, CHRF, COMET]
    SUBJECTS = [f"{s}-{t}" for s, t in product(LANG_MAP, LANG_MAP) if s != t]
    PERTURBATION_UNMODIFIABLE_WORDS = ["sentence"]
    LANGUAGE = {
        "deu_Latn": Language.DEU,
        "eng_Latn": Language.ENG,
        "fra_Latn": Language.FRA,
        "ita_Latn": Language.ITA,
        "nld_Latn": Language.NLD,
        "pol_Latn": Language.POL,
        "rus_Cyrl": Language.RUS,
        "spa_Latn": Language.SPA,
        "ukr_Cyrl": Language.UKR,
    }

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences = ["\n"]

    def _load_dataset(self, subject: str) -> None:
        hf_dataset_src = self._load_hf_dataset(path=self.DATASET_PATH, name=subject.split("-")[0])
        hf_dataset_tgt = self._load_hf_dataset(path=self.DATASET_PATH, name=subject.split("-")[1])
        self.dataset = {}
        self.rnd = random.Random(42)

        for split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
            data_src = hf_dataset_src[split]
            data_tgt = hf_dataset_tgt[split]
            data_list = []
            for item_src, item_tgt in zip(data_src, data_tgt):
                assert item_src["id"] == item_tgt["id"]
                iso_src = f"{item_src['iso_639_3']}_{item_src['iso_15924']}"
                iso_tgt = f"{item_tgt['iso_639_3']}_{item_tgt['iso_15924']}"
                text_src = item_src["text"]
                text_tgt = item_tgt["text"]
                data_list.append({"iso_source": iso_src, "iso_target": iso_tgt, "source": text_src, "target": text_tgt})
            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)
            self.dataset[split] = data_list

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        target_language = LANG_MAP[item["iso_target"]]
        instruction = f"Translate the following text into {target_language}:\n{item['source']}"
        return instruction

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return item["target"]

    def _get_context(self, item: dict[str, Any]) -> BaseMetricContext | list[BaseMetricContext] | None:
        return UntemplatedPrompt(untemplated_prompt=item["source"])

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        return completion_text.strip()
