import re
import unicodedata

from eval_framework.metrics.base import (
    BaseMetric,
    MetricResult,
)
from eval_framework.shared.types import Completion, Error, LanguageMetricContext, extract_context_metric

# Dictionary of "none" words in different languages
NONE_DICT = {
    "en": ["none"],
    "ko": ["없음"],
    "pl": ["brak"],
    "zh": ["无"],
    "vi": ["Không có"],
    "ja": ["なし", "数字はありません"],
    "ta": ["ஏதுமில்லை"],
    "hu": ["nincs"],
    "fr": ["aucun"],
    "no": ["ingen"],
    "uk": ["немає", "Нема"],
    "ru": ["нет"],
    "de": ["Keine vorhanden"],
    "es": ["ninguno"],
    "sv": ["inga"],
    "fi": ["ei mikään"],
    "cs": ["žádné", "žádná"],
    "sr": ["nema"],
    "pt": ["nenhum"],
    "it": ["nessuno"],
    "fa": ["هیچ کدام"],
    "sw": ["hakuna"],
    "nl": ["geen"],
    "st": ["ha ho letho"],
    "hi": ["कोई नहीं"],
    "da": ["ingen"],
}


def clean_text(text: str) -> str:
    """Clean text by removing spaces and normalizing"""
    return text.strip().lower().replace("\u200c", "").replace(" ", "")


class NIAHAccuracy(BaseMetric[Completion]):
    """Metric for Needle in a Haystack tasks"""

    NAME = "NIAHAccuracy"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, LanguageMetricContext)

        ground_truths = [gt for gt in response.ground_truth_list if gt is not None]

        try:
            # Extract task and language from metadata
            assert response.context is not None
            language = context.language

            # Get model's answer
            model_answer = response.completion

            # Determine which comparison function to use based on the task
            none_values = set(v for values in NONE_DICT.values() for v in values)
            if ground_truths[0] in none_values:
                is_correct = self._compare_none(language, model_answer)
            else:
                is_correct = self._compare_numbers(language, ground_truths, model_answer)

            return [
                MetricResult(
                    metric_name=self.NAME, value=float(is_correct), higher_is_better=True, error=response.error
                )
            ]
        except Exception as e:
            error = Error(error_class=e.__class__.__name__, message=str(e), traceback="")
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=error)]

    def _compare_numbers(self, lang: str, correct_answer: list[str], model_answer: str) -> bool:
        """Compare numbers for regular NIAH tasks"""
        if "-" in lang:
            inst_lang = lang.split("-")[1]
        else:
            inst_lang = lang

        if not model_answer:
            return False

        processed_model_answer = unicodedata.normalize("NFKC", model_answer)

        none_words = NONE_DICT.get(inst_lang, ["none"])
        # Check if any word in none_words is present in the processed answer; if yes, auto-fail
        for word in none_words:
            if word in processed_model_answer or clean_text(word) in processed_model_answer:
                return False

        # Extract all numeric substrings from the processed answer
        numeric_strings = re.findall(r"\d+", processed_model_answer)

        # Remove numbers that consist of a single digit
        numeric_strings = [num for num in numeric_strings if len(num) > 1]

        # Remove duplicates while preserving the original order
        numeric_strings = list(dict.fromkeys(numeric_strings))

        # If no numerics are found after processing, return False
        if not numeric_strings:
            return False

        # Convert the extracted number strings to integers
        try:
            extracted_numbers = [int(num) for num in numeric_strings]
        except Exception:
            return False

        # Convert correct_answers elements to integers to ensure numeric comparison
        try:
            correct_converted = [int(item) for item in correct_answer]
        except Exception:
            return False

        # Check that the number of extracted numbers matches the length of correct_answers
        if len(extracted_numbers) != len(correct_converted):
            return False

        # Compare the extracted numbers with the correct answers
        if set(extracted_numbers) == set(correct_converted):
            return True
        else:
            return False

    def _compare_none(self, lang: str, model_answer: str) -> bool:
        """Compare for NIAH none tasks"""
        # Lower-case all inputs for consistent, case-insensitive processing
        if "-" in lang:
            inst_lang = lang.split("-")[1]
        else:
            inst_lang = lang

        processed_model_answer = clean_text(unicodedata.normalize("NFKC", model_answer))
        none_words = [clean_text(word) for word in NONE_DICT[inst_lang]]

        # Remove single digit numbers from the processed answer
        processed_model_answer = re.sub(r"\b\d\b", "", processed_model_answer)

        # Extract all multi-digit numeric substrings from the processed answer
        numeric_strings = re.findall(r"\d\d+", processed_model_answer)

        # If any multi-digit numbers are found, return False
        if numeric_strings:
            return False

        # Check if any of the words in none_words are present
        for word in none_words:
            if word in processed_model_answer:
                return True

        # If none of the none_words are found, return False
        return False
