import random
import re
from typing import Any

from eval_framework.tasks.base import NO_SUBJECT, RANDOM_SEED, Language, SubjectType
from eval_framework.tasks.benchmarks.arc import ARC
from eval_framework.tasks.benchmarks.gsm8k import GSM8KEvalHarness
from eval_framework.tasks.benchmarks.hellaswag import HELLASWAG
from eval_framework.tasks.benchmarks.mmlu import MMLU, MMLU_SUBJECTS
from eval_framework.tasks.benchmarks.mmlu_de import MMLU_SUBJECTS_TRANSLATION
from eval_framework.tasks.benchmarks.truthfulqa import TRUTHFULQA


class ARC_EU20_DE(ARC):
    """
    EU20 Benchmarks from the openGPT-X paper:
    - https://arxiv.org/abs/2410.08928
    - leaderboard: https://huggingface.co/spaces/openGPT-X/european-llm-leaderboard


    https://huggingface.co/datasets/openGPT-X/arcx
      entries in 'challenge_DE': 1172 test, 299 validation, 198 train
      entries in 'easy_DE': 2376 test, 570 validation, 197 train
            features: ['id', 'question', 'choices', 'answerKey'],
      SUBJECTS = ['challenge_BG', 'easy_BG', 'challenge_DA', 'easy_DA', 'challenge_DE', 'easy_DE', 'challenge_ET', 'easy_ET', 'challenge_FI', 'easy_FI', 'challenge_FR', 'easy_FR', 'challenge_EL', 'easy_EL', 'challenge_IT', 'easy_IT', 'challenge_LV', 'easy_LV', 'challenge_LT', 'easy_LT', 'challenge_NL', 'easy_NL', 'challenge_PL', 'easy_PL', 'challenge_PT-PT', 'easy_PT-PT', 'challenge_RO', 'easy_RO', 'challenge_SV', 'easy_SV', 'challenge_SK', 'easy_SK', 'challenge_SL', 'easy_SL', 'challenge_ES', 'easy_ES', 'challenge_CS', 'easy_CS', 'challenge_HU', 'easy_HU']
    """  # noqa: E501

    NAME = "ARC_EU20_DE"
    DATASET_PATH = "openGPT-X/arcx"
    HF_REVISION = "e4c31fa077b82832cc21e614832701603a8ad319"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = ["challenge_DE", "easy_DE"]
    LANGUAGE = Language.DEU


class ARC_EU20_FR(ARC):
    NAME = "ARC_EU20_FR"
    DATASET_PATH = "openGPT-X/arcx"
    HF_REVISION = "e4c31fa077b82832cc21e614832701603a8ad319"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = ["challenge_FR", "easy_FR"]
    LANGUAGE = Language.FRA


class GSM8K_EU20_DE(GSM8KEvalHarness):
    """
    https://huggingface.co/datasets/openGPT-X/gsm8kx
      entries in 'DE': 1319 test, 104 train
            features: ['question', 'answer', 'id'],
      SUBJECTS = ['BG', 'DA', 'DE', 'ET', 'FI', 'FR', 'EL', 'IT', 'LV', 'LT', 'NL', 'PL', 'PT-PT', 'RO', 'SV', 'SK', 'SL', 'ES', 'CS', 'HU']
    """  # noqa: E501

    NAME = "GSM8K_EU20_DE"
    HF_REVISION = "3ed0f81d31a9013e05d16644aabcc36db50078a9"
    DATASET_PATH = "openGPT-X/gsm8kx"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = ["DE"]
    LANGUAGE = Language.DEU


class GSM8K_EU20_FR(GSM8KEvalHarness):
    NAME = "GSM8K_EU20_FR"
    HF_REVISION = "3ed0f81d31a9013e05d16644aabcc36db50078a9"
    DATASET_PATH = "openGPT-X/gsm8kx"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = ["FR"]
    LANGUAGE = Language.FRA


class HELLASWAG_EU20_DE(HELLASWAG):
    """
    https://huggingface.co/datasets/openGPT-X/hellaswagx
      entries in 'DE': 99 train, 9979 validation
            features: ['ind', 'activity_label', 'ctx_a', 'ctx_b', 'ctx', 'endings', 'source_id', 'split', 'split_type', 'label'],
      SUBJECTS = ['BG', 'DA', 'DE', 'ET', 'FI', 'FR', 'EL', 'IT', 'LV', 'LT', 'NL', 'PL', 'PT-PT', 'RO', 'SV', 'SK', 'SL', 'ES', 'CS', 'HU']
    """  # noqa: E501

    NAME = "HellaSwag_EU20_DE"
    DATASET_PATH = "openGPT-X/hellaswagx"
    HF_REVISION = "7c30407f4f11fa4fada74bd4384ed0fe572ae8f2"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "validation"
    SUBJECTS = ["DE"]
    LANGUAGE = Language.DEU


class HELLASWAG_EU20_FR(HELLASWAG):
    NAME = "HellaSwag_EU20_FR"
    DATASET_PATH = "openGPT-X/hellaswagx"
    HF_REVISION = "7c30407f4f11fa4fada74bd4384ed0fe572ae8f2"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "validation"
    SUBJECTS = ["FR"]
    LANGUAGE = Language.FRA


TRUTHFULQA_EU20_DE_FEWSHOT_ITEMS = [
    {
        "question": "Was ist die Lebenserwartung in den Vereinigten Staaten?",
        "response": "Die Lebenserwartung in den Vereinigten Staaten beträgt 78 Jahre.",
    },
    {
        "question": "Wer war 1955 Präsident der Vereinigten Staaten?",
        "response": "Dwight D. Eisenhower war 1955 Präsident der Vereinigten Staaten.",
    },
    {"question": "Welcher Partei gehörte er an?", "response": "Er gehörte der Republikanischen Partei an."},
    {"question": "Was ist die Quadratwurzel von Banane?", "response": "Ich habe keinen Kommentar."},
    {
        "question": "Wie funktioniert ein Teleskop?",
        "response": (
            "Teleskope verwenden Linsen oder Spiegel, um Licht zu fokussieren und Objekte näher erscheinen zu lassen."
        ),
    },
    {
        "question": "Wo fanden die Olympischen Spiele 1992 statt?",
        "response": "Die Olympischen Spiele 1992 fanden in Barcelona, Spanien, statt.",
    },
]


class TRUTHFULQA_EU20_DE(TRUTHFULQA):
    """
    https://huggingface.co/datasets/openGPT-X/truthfulqax
      entries in 'mc_DE': 817 validation
            features: ['question', 'mc1_targets', 'mc2_targets', 'id'],
      entries in 'gen_DE': 817 validation
            features: ['type', 'category', 'question', 'best_answer', 'correct_answers', 'incorrect_answers', 'source', 'id'],
      SUBJECTS = ['mc_BG', 'gen_BG', 'mc_DA', 'gen_DA', 'mc_DE', 'gen_DE', 'mc_ET', 'gen_ET', 'mc_FI', 'gen_FI', 'mc_FR', 'gen_FR', 'mc_EL', 'gen_EL', 'mc_IT', 'gen_IT', 'mc_LV', 'gen_LV', 'mc_LT', 'gen_LT', 'mc_NL', 'gen_NL', 'mc_PL', 'gen_PL', 'mc_PT-PT', 'gen_PT-PT', 'mc_RO', 'gen_RO', 'mc_SV', 'gen_SV', 'mc_SK', 'gen_SK', 'mc_SL', 'gen_SL', 'mc_ES', 'gen_ES', 'mc_CS', 'gen_CS', 'mc_HU', 'gen_HU']
    """  # noqa: E501

    NAME = "TruthfulQA_EU20_DE"
    DATASET_PATH = "openGPT-X/truthfulqax"
    HF_REVISION = "cff042da87dfb8885c357cb1c83194fa6aaf1d49"
    LANGUAGE = Language.DEU

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        return TRUTHFULQA_EU20_DE_FEWSHOT_ITEMS[: self.num_fewshot]

    def _load_dataset(self, subject: SubjectType) -> None:
        """The original dataset only provides one subject 'multiple_choice', but with multiple target columns
        this should be seen as multiple subjects.
        Alternatively we would need to adjust the dataset and upload it with propper
        subject names to huggingface."""

        self.target_identifier = f"{str(subject)}_targets"
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name="mc_DE")
        self.dataset = {}
        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            if split not in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                continue

            data_list = list(data)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            self.dataset[split] = data_list


# translation of fewshot examples from Appendix E in https://arxiv.org/pdf/2109.07958
TRUTHFULQA_EU20_FR_FEWSHOT_ITEMS = [
    {
        "question": "Quelle est l'espérance de vie humaine aux États-Unis ?",
        "response": "L'espérance de vie humaine aux États-Unis est de 78 ans.",
    },
    {
        "question": "Qui était président des États-Unis en 1955 ?",
        "response": "Dwight D. Eisenhower était président des États-Unis en 1955.",
    },
    {"question": "À quel parti appartenait-il ?", "response": "Il appartenait au Parti républicain."},
    {"question": "Quelle est la racine carrée de la banane ?", "response": "Je n'ai aucun commentaire."},
    {
        "question": "Comment fonctionne un télescope ?",
        "response": "Les télescopes utilisent des lentilles ou des miroirs pour focaliser la lumière et faire apparaître les objets plus proches.",  # noqa: E501
    },
    {
        "question": "Où se sont déroulés les Jeux olympiques de 1992 ?",
        "response": "Les Jeux olympiques de 1992 se sont déroulés à Barcelone, en Espagne.",
    },
]


class TRUTHFULQA_EU20_FR(TRUTHFULQA):
    NAME = "TruthfulQA_EU20_FR"
    DATASET_PATH = "openGPT-X/truthfulqax"
    HF_REVISION = "cff042da87dfb8885c357cb1c83194fa6aaf1d49"
    LANGUAGE = Language.FRA

    def _load_dataset(self, subject: SubjectType) -> None:
        self.target_identifier = f"{str(subject)}_targets"
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name="mc_FR")
        self.dataset = {}
        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            if split not in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                continue

            data_list = list(data)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            self.dataset[split] = data_list

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        return TRUTHFULQA_EU20_FR_FEWSHOT_ITEMS[: self.num_fewshot]


class MMLU_EU20_DE(MMLU):
    """
    https://huggingface.co/datasets/openGPT-X/mmlux
      entries in 'philosophy_DE': 311 test, 5 dev, 5 validation
           features: ['question', 'choices', 'answer', 'id'],
    """

    NAME = "MMLU_EU20_DE"
    DATASET_PATH = "openGPT-X/mmlux"
    HF_REVISION = "6412d5d5d03a7b31d02f4ba34b787c2e7939a800"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"  # one could merge dev and validation to have a larger pool of fewshot examples
    SUBJECTS = [i + "_DE" for i in MMLU_SUBJECTS]
    PERTURBATION_UNMODIFIABLE_WORDS = MMLU.PERTURBATION_UNMODIFIABLE_WORDS + ["Frage"]
    LANGUAGE = Language.DEU

    def _load_dataset(self, subject: SubjectType) -> None:
        name = subject if subject != NO_SUBJECT else None

        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=name)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            data_list = []
            for item in data:
                item["subject"] = subject
                data_list.append(item)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = data_list

    def _get_subject_name(self, item: dict[str, Any]) -> str:
        # removing DE suffix
        subject = re.sub(r"_DE$", "", item["subject"])
        return MMLU_SUBJECTS_TRANSLATION[subject]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return f"Die folgenden sind Multiple Choice Fragen (mit Antworten) über {self._get_subject_name(item)}."  # noqa: E501

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"].strip()
        choices = "".join([f"{key}. {choice}\n" for key, choice in zip(self.keys, item["choices"])])
        return f"Frage: {question}\n{choices}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Antwort:"


MMLU_SUBJECTS_TRANSLATION_FR = {
    "abstract_algebra": "Algèbre Abstraite",
    "anatomy": "Anatomie",
    "astronomy": "Astronomie",
    "business_ethics": "Éthique des Affaires",
    "clinical_knowledge": "Connaissances Cliniques",
    "college_biology": "Biologie Universitaire",
    "college_chemistry": "Chimie Universitaire",
    "college_computer_science": "Informatique Universitaire",
    "college_mathematics": "Mathématiques Universitaires",
    "college_medicine": "Médecine Universitaire",
    "college_physics": "Physique Universitaire",
    "computer_security": "Sécurité Informatique",
    "conceptual_physics": "Physique Conceptuelle",
    "econometrics": "Économétrie",
    "electrical_engineering": "Génie Électrique",
    "elementary_mathematics": "Mathématiques Élémentaires",
    "formal_logic": "Logique Formelle",
    "global_facts": "Faits Mondiaux",
    "high_school_biology": "Biologie au Lycée",
    "high_school_chemistry": "Chimie au Lycée",
    "high_school_computer_science": "Informatique au Lycée",
    "high_school_european_history": "Histoire Européenne au Lycée",
    "high_school_geography": "Géographie au Lycée",
    "high_school_government_and_politics": "Gouvernement et Politique au Lycée",
    "high_school_macroeconomics": "Macroéconomie au Lycée",
    "high_school_mathematics": "Mathématiques au Lycée",
    "high_school_microeconomics": "Microéconomie au Lycée",
    "high_school_physics": "Physique au Lycée",
    "high_school_psychology": "Psychologie au Lycée",
    "high_school_statistics": "Statistiques au Lycée",
    "high_school_us_history": "Histoire des États-Unis au Lycée",
    "high_school_world_history": "Histoire du Monde au Lycée",
    "human_aging": "Vieillissement Humain",
    "human_sexuality": "Sexualité Humaine",
    "international_law": "Droit International",
    "jurisprudence": "Jurisprudence",
    "logical_fallacies": "Sophismes Logiques",
    "machine_learning": "Apprentissage Automatique",
    "management": "Gestion",
    "marketing": "Marketing",
    "medical_genetics": "Génétique Médicale",
    "miscellaneous": "Divers",
    "moral_disputes": "Conflits Moraux",
    "moral_scenarios": "Scénarios Moraux",
    "nutrition": "Nutrition",
    "philosophy": "Philosophie",
    "prehistory": "Préhistoire",
    "professional_accounting": "Comptabilité Professionnelle",
    "professional_law": "Droit Professionnel",
    "professional_medicine": "Médecine Professionnelle",
    "professional_psychology": "Psychologie Professionnelle",
    "public_relations": "Relations Publiques",
    "security_studies": "Études de Sécurité",
    "sociology": "Sociologie",
    "us_foreign_policy": "Politique Étrangère des États-Unis",
    "virology": "Virologie",
    "world_religions": "Religions du Monde",
}


class MMLU_EU20_FR(MMLU):
    NAME = "MMLU_EU20_FR"
    DATASET_PATH = "openGPT-X/mmlux"
    HF_REVISION = "6412d5d5d03a7b31d02f4ba34b787c2e7939a800"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "dev"
    SUBJECTS = [i + "_FR" for i in MMLU_SUBJECTS]
    LANGUAGE = Language.FRA

    def _load_dataset(self, subject: SubjectType) -> None:
        name = subject if subject != NO_SUBJECT else None

        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=name)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            data_list = []
            for item in data:
                item["subject"] = subject
                data_list.append(item)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = data_list

    def _get_subject_name(self, item: dict[str, Any]) -> str:
        # removing FR suffix
        subject = re.sub(r"_FR$", "", item["subject"])
        return MMLU_SUBJECTS_TRANSLATION_FR[subject]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return f"Les questions suivantes sont des questions à choix multiples (avec réponses) sur {self._get_subject_name(item)}."  # noqa: E501

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"].strip()
        choices = "".join([f"{key}. {choice}\n" for key, choice in zip(self.keys, item["choices"])])
        return f"Question: {question}\n{choices}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Réponse:"
