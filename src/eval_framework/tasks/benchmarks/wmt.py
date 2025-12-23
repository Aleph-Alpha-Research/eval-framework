import random
from abc import ABC
from typing import Any

import pycountry

from eval_framework.metrics.completion.bleu import LINEWISE_BLEU
from eval_framework.metrics.completion.chrf import LINEWISE_CHRF
from eval_framework.metrics.completion.ter import LINEWISE_TER
from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Language, ResponseType, Sample


class WMT(BaseTask[str], ABC):
    """WMT dataset: https://huggingface.co/datasets/wmt

    Uses HuggingFace datasets for deterministic data loading.
    """

    NAME = "WMT"
    DATASET_PATH = ""
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [LINEWISE_BLEU, LINEWISE_CHRF, LINEWISE_TER]
    PERTURBATION_UNMODIFIABLE_WORDS = ["phrase"]

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences: list[str] = [".\n", " phrase: ", "phrase:", "phrase: ", " phrase:", "\n\n"]

    def _load_dataset(self, subject: str | None) -> None:
        """Load WMT dataset from HuggingFace.

        The HuggingFace WMT datasets use the structure:
        {
            "translation": {
                "src_lang": "Source text",
                "tgt_lang": "Target text"
            }
        }

        Args:
            subject: Language pair in format "src-tgt" (e.g., "de-en", "en-fr")
        """
        # Load from HuggingFace - subject is the language pair like "de-en"
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=subject)

        # Parse the language pair to get source and target language codes
        if subject is None:
            raise ValueError("subject is required for WMT dataset")
        src_lang, tgt_lang = subject.split("-")

        self.dataset = {}
        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            if split not in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                continue

            # Transform HuggingFace format to expected format
            data_list = []
            for item in data:
                translation = item["translation"]
                data_list.append(
                    {
                        "source": translation[src_lang],
                        "target": translation[tgt_lang],
                        "subject": subject,
                    }
                )

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            self.dataset[split] = data_list

    def _code_to_language(self, code: str) -> str:
        # key is alpha_2 or alpha_3 depending on the code length
        key = f"alpha_{len(code)}"
        language_tuple = pycountry.languages.get(**{key: code})
        return language_tuple.name

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        language_codes = item["subject"].split("-")
        src_lang = self._code_to_language(language_codes[0])

        language_codes = item["subject"].split("-")
        tar_lang = self._code_to_language(language_codes[1])
        cue = f"{tar_lang} phrase:"

        return f"{src_lang} phrase: {item['source']}\n{cue}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return item["target"] if isinstance(item["target"], str) else item["target"][0]

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = self._get_ground_truth(item)
        assert target is not None
        assert isinstance(target, str)
        return f" {target}"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return completion_text.strip()


class WMT14(WMT):
    NAME = "WMT14"
    DATASET_PATH = "wmt/wmt14"
    SUBJECTS = ["fr-en", "en-fr"]
    LANGUAGE = {
        "en-fr": (Language["ENG"], Language["FRA"]),
        "fr-en": (Language["FRA"], Language["ENG"]),
    }


class WMT16(WMT):
    NAME = "WMT16"
    DATASET_PATH = "wmt/wmt16"
    SUBJECTS = ["de-en", "en-de"]
    LANGUAGE = {
        "de-en": (Language["DEU"], Language["ENG"]),
        "en-de": (Language["ENG"], Language["DEU"]),
    }


class WMT20(WMT):
    NAME = "WMT20"
    DATASET_PATH = "wmt/wmt20"
    SUBJECTS = ["de-en", "de-fr", "en-de", "fr-de"]
    LANGUAGE = {
        "de-en": (Language["DEU"], Language["ENG"]),
        "de-fr": (Language["DEU"], Language["FRA"]),
        "en-de": (Language["ENG"], Language["DEU"]),
        "fr-de": (Language["FRA"], Language["DEU"]),
    }


class WMT_INSTRUCT(WMT):
    PERTURBATION_UNMODIFIABLE_WORDS = ["Please", "translate"]
    COMPLETION_PREFIX = "This is the translation:"

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.stop_sequences: list[str] = ["Please translate"]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        src_lang, tar_lang = map(self._code_to_language, item["subject"].split("-"))
        return f"Please translate from {src_lang} to {tar_lang}: {item['source']}"

    def _get_cue(self, item: dict[str, Any]) -> str:
        return self.COMPLETION_PREFIX

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        target = self._get_ground_truth(item)
        assert target is not None
        return f" {target}"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        completion_text = completion_text.removeprefix(self.COMPLETION_PREFIX)
        completion_text = completion_text.strip()
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return completion_text


class WMT14_INSTRUCT(WMT_INSTRUCT):
    NAME = "WMT14 Instruct"
    DATASET_PATH = "wmt/wmt14"
    SUBJECTS = ["fr-en", "en-fr"]
    LANGUAGE = {
        "en-fr": (Language["ENG"], Language["FRA"]),
        "fr-en": (Language["FRA"], Language["ENG"]),
    }


class WMT16_INSTRUCT(WMT_INSTRUCT):
    NAME = "WMT16 Instruct"
    DATASET_PATH = "wmt/wmt16"
    SUBJECTS = ["de-en", "en-de"]
    LANGUAGE = {
        "de-en": (Language["DEU"], Language["ENG"]),
        "en-de": (Language["ENG"], Language["DEU"]),
    }


class WMT20_INSTRUCT(WMT_INSTRUCT):
    NAME = "WMT20 Instruct"
    DATASET_PATH = "wmt/wmt20"
    SUBJECTS = ["de-en", "de-fr", "en-de", "fr-de"]
    LANGUAGE = {
        "de-en": (Language["DEU"], Language["ENG"]),
        "de-fr": (Language["DEU"], Language["FRA"]),
        "en-de": (Language["ENG"], Language["DEU"]),
        "fr-de": (Language["FRA"], Language["DEU"]),
    }
