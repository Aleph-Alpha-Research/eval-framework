from typing import Any

from eval_framework.tasks.base import Language
from eval_framework.tasks.benchmarks.winogrande import WINOGRANDE

ANSWER_STR_TO_NUM = {"1": 0, "2": 1}


class WINOX(WINOGRANDE):
    """
    Wino-X is a parallel dataset of German, French, and Russian Winograd schemas, aligned with their English
    counterparts, used to examine whether neural machine translation models can perform coreference resolution that
    requires commonsense knowledge, and whether multilingual language models are capable of commonsense reasoning
    across multiple languages.

    Winogrande: https://arxiv.org/abs/1907.10641
    Wino-X: https://github.com/demelin/Wino-X
    Wino-X: https://huggingface.co/datasets/demelin/wino_x
    """

    DATASET_PATH = "demelin/wino_x"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    LANGUAGE_SHORT_CODE = ""

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        choices = self._extract_choices(item)
        # in winogrande answer is a string but in wino_x it is an int
        return f" {choices[ANSWER_STR_TO_NUM[str(item['answer'])]]}"

    def _extract_question(self, item: dict) -> str:
        question, _ = item[f"context_{self.LANGUAGE_SHORT_CODE}"].split("_")
        question = question.replace("  ", " ")
        return question.strip()

    def _extract_choices(self, item: dict) -> list[str]:
        _, choice_suffix = item[f"context_{self.LANGUAGE_SHORT_CODE}"].split("_")
        choice_suffix = choice_suffix.replace("  ", " ")
        choices = [
            choice + choice_suffix
            for choice in [item[f"option1_{self.LANGUAGE_SHORT_CODE}"], item[f"option2_{self.LANGUAGE_SHORT_CODE}"]]
        ]
        return choices


class WINOX_DE(WINOX):
    NAME = "WINOX_DE"
    SUBJECTS = ["lm_en_de"]
    LANGUAGE = Language.DEU
    LANGUAGE_SHORT_CODE = "de"


class WINOX_FR(WINOX):
    NAME = "WINOX_FR"
    SUBJECTS = ["lm_en_fr"]
    LANGUAGE = Language.FRA
    LANGUAGE_SHORT_CODE = "fr"
