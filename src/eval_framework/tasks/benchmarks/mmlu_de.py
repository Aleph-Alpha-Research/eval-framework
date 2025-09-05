from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.utils import get_n_letters

MMLU_SUBJECTS_TRANSLATION = {
    "abstract_algebra": "Abstrakte Algebra",
    "anatomy": "Anatomie",
    "astronomy": "Astronomie",
    "business_ethics": "Wirtschaftsethik",
    "clinical_knowledge": "Klinisches Wissen",
    "college_biology": "Hochschulbiologie",
    "college_chemistry": "Hochschulchemie",
    "college_computer_science": "Hochschulinformatik",
    "college_mathematics": "Hochschulmathematik",
    "college_medicine": "Hochschulmedizin",
    "college_physics": "Hochschulphysik",
    "computer_security": "Computersicherheit",
    "conceptual_physics": "Konzeptuelle Physik",
    "econometrics": "Ökonometrie",
    "electrical_engineering": "Elektrotechnik",
    "elementary_mathematics": "Elementarmathematik",
    "formal_logic": "Formale Logik",
    "global_facts": "Globale Fakten",
    "high_school_biology": "Gymnasialbiologie",
    "high_school_chemistry": "Gymnasialchemie",
    "high_school_computer_science": "Gymnasiale Informatik",
    "high_school_european_history": "Gymnasiale Europäische Geschichte",
    "high_school_geography": "Gymnasiale Geographie",
    "high_school_government_and_politics": "Gymnasiale Regierung und Politik",
    "high_school_macroeconomics": "Gymnasiale Makroökonomie",
    "high_school_mathematics": "Gymnasialmathematik",
    "high_school_microeconomics": "Gymnasiale Mikroökonomie",
    "high_school_physics": "Gymnasialphysik",
    "high_school_psychology": "Gymnasialpsychologie",
    "high_school_statistics": "Gymnasialstatistik",
    "high_school_us_history": "Gymnasiale US-Geschichte",
    "high_school_world_history": "Gymnasiale Weltgeschichte",
    "human_aging": "Menschliches Altern",
    "human_sexuality": "Menschliche Sexualität",
    "international_law": "Internationales Recht",
    "jurisprudence": "Rechtswissenschaft",
    "logical_fallacies": "Logische Fehlschlüsse",
    "machine_learning": "Maschinelles Lernen",
    "management": "Management",
    "marketing": "Marketing",
    "medical_genetics": "Medizinische Genetik",
    "miscellaneous": "Verschiedenes",
    "moral_disputes": "Moralische Streitfragen",
    "moral_scenarios": "Moralische Szenarien",
    "nutrition": "Ernährung",
    "philosophy": "Philosophie",
    "prehistory": "Urgeschichte",
    "professional_accounting": "Berufliche Buchhaltung",
    "professional_law": "Berufliches Recht",
    "professional_medicine": "Berufliche Medizin",
    "professional_psychology": "Berufliche Psychologie",
    "public_relations": "Öffentlichkeitsarbeit",
    "security_studies": "Sicherheitsstudien",
    "sociology": "Soziologie",
    "us_foreign_policy": "US-Außenpolitik",
    "virology": "Virologie",
    "world_religions": "Weltreligionen",
}


class MMLU_DE(BaseTask[str]):
    """MMLU DE dataset: https://huggingface.co/datasets/LeoLM/MMLU_de"""

    NAME = "MMLU_DE"
    DATASET_PATH = "LeoLM/MMLU_de"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "validation"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = list(MMLU_SUBJECTS_TRANSLATION.keys())
    PERTURBATION_UNMODIFIABLE_WORDS = ["Frage"] + get_n_letters(4)
    LANGUAGE = Language.DEU

    def __init__(self, num_fewshot: int, custom_subjects: list[str] | None, custom_hf_revision: str | None) -> None:
        super().__init__(num_fewshot, custom_subjects, custom_hf_revision)

        self.keys = get_n_letters(4)

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return f"Die folgenden sind Multiple Choice Fragen (mit Antworten) über {MMLU_SUBJECTS_TRANSLATION[item['subject']]}."  # noqa: E501

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question_de"].strip()
        choices = "".join([f"{key}. {choice}\n" for key, choice in zip(self.keys, item["choices_de"])])
        return f"Frage: {question}\n{choices}"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Antwort:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {self.keys[item['answer']]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]
