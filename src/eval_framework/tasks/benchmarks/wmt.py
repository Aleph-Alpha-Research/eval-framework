# import hashlib
# import random
# from abc import ABC
# from typing import Any

# import pycountry
# import sacrebleu

# from eval_framework.metrics.completion.bleu import LINEWISE_BLEU
# from eval_framework.metrics.completion.chrf import LINEWISE_CHRF
# from eval_framework.metrics.completion.ter import LINEWISE_TER
# from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Language, ResponseType, Sample


# class WMT(BaseTask[str], ABC):
#     """WMT dataset:"""

#     NAME = "WMT"
#     DATASET_PATH = ""
#     SAMPLE_SPLIT = "test"
#     FEWSHOT_SPLIT = "test"
#     RESPONSE_TYPE = ResponseType.COMPLETION
#     METRICS = [LINEWISE_BLEU, LINEWISE_CHRF, LINEWISE_TER]
#     PERTURBATION_UNMODIFIABLE_WORDS = ["phrase"]

#     def __init__(self, num_fewshot: int = 0) -> None:
#         super().__init__(num_fewshot)
#         self.stop_sequences: list[str] = [".\n", " phrase: ", "phrase:", "phrase: ", " phrase:", "\n\n"]

#     def _load_dataset(self, subject: str | None) -> None:
#         src_file, ref_files, _, _, _ = sacrebleu.download_test_set(test_set=self.DATASET_PATH, langpair=subject)

#         # Handle multiple reference files - use first non-empty one
#         if isinstance(ref_files, list):
#             ref_files = sorted(ref_files)
#             ref_data = None
#             for candidate_ref in ref_files:
#                 try:
#                     with sacrebleu.smart_open(candidate_ref) as f:
#                         test_data = [line.rstrip() for line in f]
#                         if test_data:
#                             ref_data = test_data
#                             break
#                 except Exception:
#                     continue

#             if ref_data is None:
#                 raise ValueError(f"No valid reference file found for {self.DATASET_PATH}/{subject}")
#         else:
#             with sacrebleu.smart_open(ref_files) as f:
#                 ref_data = [line.rstrip() for line in f]

#         # Read source file
#         with sacrebleu.smart_open(src_file) as f:
#             src_data = [line.rstrip() for line in f]

#         assert len(src_data) == len(ref_data), f"Mismatch: {len(src_data)} vs {len(ref_data)} lines"

#         # Create indexed pairs for deterministic sorting
#         indexed_pairs = list(enumerate(zip(src_data, ref_data)))

#         # Sort by content hash (primary) + original index (tiebreaker)
#         def sort_key(item: tuple[int, tuple[str, str]]) -> tuple[str, int]:
#             idx, (src, ref) = item
#             content = f"{src}\x00{ref}"
#             content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
#             return (content_hash, idx)

#         indexed_pairs.sort(key=sort_key)

#         # Create final data list
#         data_list = [{"source": src, "target": ref, "subject": subject} for _, (src, ref) in indexed_pairs]

#         # Shuffle with fixed seed for reproducibility
#         self.rnd = random.Random(RANDOM_SEED)
#         self.rnd.shuffle(data_list)
#         self.dataset = {"test": data_list}

#     def _code_to_language(self, code: str) -> str:
#         # key is alpha_2 or alpha_3 depending on the code length
#         key = f"alpha_{len(code)}"
#         language_tuple = pycountry.languages.get(**{key: code})
#         return language_tuple.name

#     def _get_instruction_text(self, item: dict[str, Any]) -> str:
#         language_codes = item["subject"].split("-")
#         src_lang = self._code_to_language(language_codes[0])

#         language_codes = item["subject"].split("-")
#         tar_lang = self._code_to_language(language_codes[1])
#         cue = f"{tar_lang} phrase:"

#         return f"{src_lang} phrase: {item['source']}\n{cue}"

#     def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
#         return item["target"] if isinstance(item["target"], str) else item["target"][0]

#     def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
#         target = self._get_ground_truth(item)
#         assert target is not None
#         assert isinstance(target, str)
#         return f" {target}"

#     def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
#         for stop_sequence in self.stop_sequences:
#             if stop_sequence in completion_text:
#                 completion_text = completion_text.split(stop_sequence)[0]
#         return completion_text.strip()


# class WMT14(WMT):
#     NAME = "WMT14"
#     DATASET_PATH = "wmt14"
#     SUBJECTS = ["en-fr", "fr-en"]
#     LANGUAGE = {
#         "en-fr": (Language["ENG"], Language["FRA"]),
#         "fr-en": (Language["FRA"], Language["ENG"]),
#     }


# class WMT16(WMT):
#     NAME = "WMT16"
#     DATASET_PATH = "wmt16"
#     SUBJECTS = ["de-en", "en-de"]
#     LANGUAGE = {
#         "de-en": (Language["DEU"], Language["ENG"]),
#         "en-de": (Language["ENG"], Language["DEU"]),
#     }


# class WMT20(WMT):
#     NAME = "WMT20"
#     DATASET_PATH = "wmt20"
#     SUBJECTS = ["de-en", "de-fr", "en-de", "fr-de"]
#     LANGUAGE = {
#         "de-en": (Language["DEU"], Language["ENG"]),
#         "de-fr": (Language["DEU"], Language["FRA"]),
#         "en-de": (Language["ENG"], Language["DEU"]),
#         "fr-de": (Language["FRA"], Language["DEU"]),
#     }


# class WMT_INSTRUCT(WMT):
#     PERTURBATION_UNMODIFIABLE_WORDS = ["Please", "translate"]
#     COMPLETION_PREFIX = "This is the translation:"

#     def __init__(self, num_fewshot: int = 0) -> None:
#         super().__init__(num_fewshot)
#         self.stop_sequences: list[str] = ["Please translate"]

#     def _get_instruction_text(self, item: dict[str, Any]) -> str:
#         src_lang, tar_lang = map(self._code_to_language, item["subject"].split("-"))
#         return f"Please translate from {src_lang} to {tar_lang}: {item['source']}"

#     def _get_cue(self, item: dict[str, Any]) -> str:
#         return self.COMPLETION_PREFIX

#     def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
#         target = self._get_ground_truth(item)
#         assert target is not None
#         return f" {target}"

#     def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
#         completion_text = completion_text.removeprefix(self.COMPLETION_PREFIX)
#         completion_text = completion_text.strip()
#         for stop_sequence in self.stop_sequences:
#             if stop_sequence in completion_text:
#                 completion_text = completion_text.split(stop_sequence)[0]
#         return completion_text


# class WMT14_INSTRUCT(WMT_INSTRUCT):
#     NAME = "WMT14 Instruct"
#     DATASET_PATH = "wmt14"
#     SUBJECTS = ["en-fr", "fr-en"]
#     LANGUAGE = {
#         "en-fr": (Language["ENG"], Language["FRA"]),
#         "fr-en": (Language["FRA"], Language["ENG"]),
#     }


# class WMT16_INSTRUCT(WMT_INSTRUCT):
#     NAME = "WMT16 Instruct"
#     DATASET_PATH = "wmt16"
#     SUBJECTS = ["de-en", "en-de"]
#     LANGUAGE = {
#         "de-en": (Language["DEU"], Language["ENG"]),
#         "en-de": (Language["ENG"], Language["DEU"]),
#     }


# class WMT20_INSTRUCT(WMT_INSTRUCT):
#     NAME = "WMT20 Instruct"
#     DATASET_PATH = "wmt20"
#     SUBJECTS = ["de-en", "de-fr", "en-de", "fr-de"]
#     LANGUAGE = {
#         "de-en": (Language["DEU"], Language["ENG"]),
#         "de-fr": (Language["DEU"], Language["FRA"]),
#         "en-de": (Language["ENG"], Language["DEU"]),
#         "fr-de": (Language["FRA"], Language["DEU"]),
#     }

import random
from abc import ABC
from typing import Any

import pycountry
import sacrebleu

from eval_framework.metrics.completion.bleu import LINEWISE_BLEU
from eval_framework.metrics.completion.chrf import LINEWISE_CHRF
from eval_framework.metrics.completion.ter import LINEWISE_TER
from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Language, ResponseType, Sample


class WMT(BaseTask[str], ABC):
    """WMT dataset:"""

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
        src_file, ref_file, _, _, _ = sacrebleu.download_test_set(test_set=self.DATASET_PATH, langpair=subject)
        src_data, ref_data = [[line.rstrip() for line in sacrebleu.smart_open(file)] for file in (src_file, ref_file)]

        data_list = [{"source": src, "target": ref, "subject": subject} for src, ref in zip(src_data, ref_data)]
        self.rnd = random.Random(RANDOM_SEED)
        self.rnd.shuffle(data_list)
        self.dataset = {"test": data_list}

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
    DATASET_PATH = "wmt14"
    SUBJECTS = ["en-fr", "fr-en"]
    LANGUAGE = {
        "en-fr": (Language["ENG"], Language["FRA"]),
        "fr-en": (Language["FRA"], Language["ENG"]),
    }


class WMT16(WMT):
    NAME = "WMT16"
    DATASET_PATH = "wmt16"
    SUBJECTS = ["de-en", "en-de"]
    LANGUAGE = {
        "de-en": (Language["DEU"], Language["ENG"]),
        "en-de": (Language["ENG"], Language["DEU"]),
    }


class WMT20(WMT):
    NAME = "WMT20"
    DATASET_PATH = "wmt20"
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
    DATASET_PATH = "wmt14"
    SUBJECTS = ["en-fr", "fr-en"]
    LANGUAGE = {
        "en-fr": (Language["ENG"], Language["FRA"]),
        "fr-en": (Language["FRA"], Language["ENG"]),
    }


class WMT16_INSTRUCT(WMT_INSTRUCT):
    NAME = "WMT16 Instruct"
    DATASET_PATH = "wmt16"
    SUBJECTS = ["de-en", "en-de"]
    LANGUAGE = {
        "de-en": (Language["DEU"], Language["ENG"]),
        "en-de": (Language["ENG"], Language["DEU"]),
    }


class WMT20_INSTRUCT(WMT_INSTRUCT):
    NAME = "WMT20 Instruct"
    DATASET_PATH = "wmt20"
    SUBJECTS = ["de-en", "de-fr", "en-de", "fr-de"]
    LANGUAGE = {
        "de-en": (Language["DEU"], Language["ENG"]),
        "de-fr": (Language["DEU"], Language["FRA"]),
        "en-de": (Language["ENG"], Language["DEU"]),
        "fr-de": (Language["FRA"], Language["DEU"]),
    }