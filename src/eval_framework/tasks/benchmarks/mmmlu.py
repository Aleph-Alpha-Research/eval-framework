import random
import re
from itertools import product
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.metrics.completion.language_checker import GermanCompletionChecker
from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.benchmarks.mmlu import MMLU_SUBJECTS
from eval_framework.tasks.utils import get_n_letters

MMMLU_LANGS = ["FR_FR", "DE_DE", "ES_LA", "IT_IT", "PT_BR", "AR_XY"]
MMLU_SUBJECTS_DE = {
    "abstract_algebra": "Abstrakte Algebra",
    "anatomy": "Anatomie",
    "astronomy": "Astronomie",
    "business_ethics": "Wirtschaftsethik",
    "clinical_knowledge": "Klinisches Wissen",
    "college_biology": "Biologie (Universität)",
    "college_chemistry": "Chemie (Universität)",
    "college_computer_science": "Informatik (Universität)",
    "college_mathematics": "Mathematik (Universität)",
    "college_medicine": "Medizin (Universität)",
    "college_physics": "Physik (Universität)",
    "computer_security": "IT-Sicherheit",
    "conceptual_physics": "Konzeptuelle Physik",
    "econometrics": "Ökonometrie",
    "electrical_engineering": "Elektrotechnik",
    "elementary_mathematics": "Elementarmathematik",
    "formal_logic": "Formale Logik",
    "global_facts": "Weltwissen",
    "high_school_biology": "Biologie (Gymnasium)",
    "high_school_chemistry": "Chemie (Gymnasium)",
    "high_school_computer_science": "Informatik (Gymnasium)",
    "high_school_european_history": "Europäische Geschichte (Gymnasium)",
    "high_school_geography": "Geografie (Gymnasium)",
    "high_school_government_and_politics": "Politik und Regierung (Gymnasium)",
    "high_school_macroeconomics": "Makroökonomie (Gymnasium)",
    "high_school_mathematics": "Mathematik (Gymnasium)",
    "high_school_microeconomics": "Mikroökonomie (Gymnasium)",
    "high_school_physics": "Physik (Gymnasium)",
    "high_school_psychology": "Psychologie (Gymnasium)",
    "high_school_statistics": "Statistik (Gymnasium)",
    "high_school_us_history": "US-Geschichte (Gymnasium)",
    "high_school_world_history": "Weltgeschichte (Gymnasium)",
    "human_aging": "Altern des Menschen",
    "human_sexuality": "Menschliche Sexualität",
    "international_law": "Völkerrecht",
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
    "professional_accounting": "Berufsbezogene Buchhaltung",
    "professional_law": "Berufsbezogenes Recht",
    "professional_medicine": "Berufsbezogene Medizin",
    "professional_psychology": "Berufsbezogene Psychologie",
    "public_relations": "Öffentlichkeitsarbeit",
    "security_studies": "Sicherheitsstudien",
    "sociology": "Soziologie",
    "us_foreign_policy": "US-Außenpolitik",
    "virology": "Virologie",
    "world_religions": "Weltreligionen",
}
MMLU_SUBJECTS_FR = {
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
    "high_school_world_history": "Histoire Mondiale au Lycée",
    "human_aging": "Vieillissement Humain",
    "human_sexuality": "Sexualité Humaine",
    "international_law": "Droit International",
    "jurisprudence": "Jurisprudence",
    "logical_fallacies": "Fautes de Logique",
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
MMLU_SUBJECTS_ES = {
    "abstract_algebra": "Álgebra Abstracta",
    "anatomy": "Anatomía",
    "astronomy": "Astronomía",
    "business_ethics": "Ética Empresarial",
    "clinical_knowledge": "Conocimientos Clínicos",
    "college_biology": "Biología Universitaria",
    "college_chemistry": "Química Universitaria",
    "college_computer_science": "Informática Universitaria",
    "college_mathematics": "Matemáticas Universitarias",
    "college_medicine": "Medicina Universitaria",
    "college_physics": "Física Universitaria",
    "computer_security": "Seguridad Informática",
    "conceptual_physics": "Física Conceptual",
    "econometrics": "Econometría",
    "electrical_engineering": "Ingeniería Eléctrica",
    "elementary_mathematics": "Matemáticas Elementales",
    "formal_logic": "Lógica Formal",
    "global_facts": "Datos Globales",
    "high_school_biology": "Biología de Secundaria",
    "high_school_chemistry": "Química de Secundaria",
    "high_school_computer_science": "Informática de Secundaria",
    "high_school_european_history": "Historia Europea de Secundaria",
    "high_school_geography": "Geografía de Secundaria",
    "high_school_government_and_politics": "Gobierno y Política de Secundaria",
    "high_school_macroeconomics": "Macroeconomía de Secundaria",
    "high_school_mathematics": "Matemáticas de Secundaria",
    "high_school_microeconomics": "Microeconomía de Secundaria",
    "high_school_physics": "Física de Secundaria",
    "high_school_psychology": "Psicología de Secundaria",
    "high_school_statistics": "Estadística de Secundaria",
    "high_school_us_history": "Historia de EE. UU. de Secundaria",
    "high_school_world_history": "Historia Mundial de Secundaria",
    "human_aging": "Envejecimiento Humano",
    "human_sexuality": "Sexualidad Humana",
    "international_law": "Derecho Internacional",
    "jurisprudence": "Jurisprudencia",
    "logical_fallacies": "Falacias Lógicas",
    "machine_learning": "Aprendizaje Automático",
    "management": "Administración",
    "marketing": "Mercadotecnia",
    "medical_genetics": "Genética Médica",
    "miscellaneous": "Misceláneos",
    "moral_disputes": "Disputas Morales",
    "moral_scenarios": "Escenarios Morales",
    "nutrition": "Nutrición",
    "philosophy": "Filosofía",
    "prehistory": "Prehistoria",
    "professional_accounting": "Contabilidad Profesional",
    "professional_law": "Derecho Profesional",
    "professional_medicine": "Medicina Profesional",
    "professional_psychology": "Psicología Profesional",
    "public_relations": "Relaciones Públicas",
    "security_studies": "Estudios de Seguridad",
    "sociology": "Sociología",
    "us_foreign_policy": "Política Exterior de EE. UU.",
    "virology": "Virología",
    "world_religions": "Religiones del Mundo",
}
MMLU_SUBJECTS_IT = {
    "abstract_algebra": "Algebra Astratta",
    "anatomy": "Anatomia",
    "astronomy": "Astronomia",
    "business_ethics": "Etica Aziendale",
    "clinical_knowledge": "Conoscenza Clinica",
    "college_biology": "Biologia Universitaria",
    "college_chemistry": "Chimica Universitaria",
    "college_computer_science": "Informatica Universitaria",
    "college_mathematics": "Matematica Universitaria",
    "college_medicine": "Medicina Universitaria",
    "college_physics": "Fisica Universitaria",
    "computer_security": "Sicurezza Informatica",
    "conceptual_physics": "Fisica Concettuale",
    "econometrics": "Econometria",
    "electrical_engineering": "Ingegneria Elettrica",
    "elementary_mathematics": "Matematica Elementare",
    "formal_logic": "Logica Formale",
    "global_facts": "Fatti Globali",
    "high_school_biology": "Biologia Liceale",
    "high_school_chemistry": "Chimica Liceale",
    "high_school_computer_science": "Informatica Liceale",
    "high_school_european_history": "Storia Europea Liceale",
    "high_school_geography": "Geografia Liceale",
    "high_school_government_and_politics": "Governo e Politica Liceale",
    "high_school_macroeconomics": "Macroeconomia Liceale",
    "high_school_mathematics": "Matematica Liceale",
    "high_school_microeconomics": "Microeconomia Liceale",
    "high_school_physics": "Fisica Liceale",
    "high_school_psychology": "Psicologia Liceale",
    "high_school_statistics": "Statistica Liceale",
    "high_school_us_history": "Storia Americana Liceale",
    "high_school_world_history": "Storia Mondiale Liceale",
    "human_aging": "Invecchiamento Umano",
    "human_sexuality": "Sessualità Umana",
    "international_law": "Diritto Internazionale",
    "jurisprudence": "Giurisprudenza",
    "logical_fallacies": "Fallacie Logiche",
    "machine_learning": "Apprendimento Automatico",
    "management": "Gestione",
    "marketing": "Marketing",
    "medical_genetics": "Genetica Medica",
    "miscellaneous": "Varie",
    "moral_disputes": "Controversie Morali",
    "moral_scenarios": "Scenari Morali",
    "nutrition": "Nutrizione",
    "philosophy": "Filosofia",
    "prehistory": "Preistoria",
    "professional_accounting": "Contabilità Professionale",
    "professional_law": "Diritto Professionale",
    "professional_medicine": "Medicina Professionale",
    "professional_psychology": "Psicologia Professionale",
    "public_relations": "Relazioni Pubbliche",
    "security_studies": "Studi sulla Sicurezza",
    "sociology": "Sociologia",
    "us_foreign_policy": "Politica Estera degli Stati Uniti",
    "virology": "Virologia",
    "world_religions": "Religioni del Mondo",
}
MMLU_SUBJECTS_PT = {
    "abstract_algebra": "Álgebra Abstrata",
    "anatomy": "Anatomia",
    "astronomy": "Astronomia",
    "business_ethics": "Ética Empresarial",
    "clinical_knowledge": "Conhecimento Clínico",
    "college_biology": "Biologia Universitária",
    "college_chemistry": "Química Universitária",
    "college_computer_science": "Ciência da Computação Universitária",
    "college_mathematics": "Matemática Universitária",
    "college_medicine": "Medicina Universitária",
    "college_physics": "Física Universitária",
    "computer_security": "Segurança da Computação",
    "conceptual_physics": "Física Conceitual",
    "econometrics": "Econometria",
    "electrical_engineering": "Engenharia Elétrica",
    "elementary_mathematics": "Matemática Elementar",
    "formal_logic": "Lógica Formal",
    "global_facts": "Fatos Globais",
    "high_school_biology": "Biologia do Ensino Médio",
    "high_school_chemistry": "Química do Ensino Médio",
    "high_school_computer_science": "Ciência da Computação do Ensino Médio",
    "high_school_european_history": "História Europeia do Ensino Médio",
    "high_school_geography": "Geografia do Ensino Médio",
    "high_school_government_and_politics": "Governo e Política do Ensino Médio",
    "high_school_macroeconomics": "Macroeconomia do Ensino Médio",
    "high_school_mathematics": "Matemática do Ensino Médio",
    "high_school_microeconomics": "Microeconomia do Ensino Médio",
    "high_school_physics": "Física do Ensino Médio",
    "high_school_psychology": "Psicologia do Ensino Médio",
    "high_school_statistics": "Estatística do Ensino Médio",
    "high_school_us_history": "História dos EUA do Ensino Médio",
    "high_school_world_history": "História Mundial do Ensino Médio",
    "human_aging": "Envelhecimento Humano",
    "human_sexuality": "Sexualidade Humana",
    "international_law": "Direito Internacional",
    "jurisprudence": "Jurisprudência",
    "logical_fallacies": "Falácias Lógicas",
    "machine_learning": "Aprendizado de Máquina",
    "management": "Administração",
    "marketing": "Marketing",
    "medical_genetics": "Genética Médica",
    "miscellaneous": "Diversos",
    "moral_disputes": "Disputas Morais",
    "moral_scenarios": "Cenários Morais",
    "nutrition": "Nutrição",
    "philosophy": "Filosofia",
    "prehistory": "Pré-História",
    "professional_accounting": "Contabilidade Profissional",
    "professional_law": "Direito Profissional",
    "professional_medicine": "Medicina Profissional",
    "professional_psychology": "Psicologia Profissional",
    "public_relations": "Relações Públicas",
    "security_studies": "Estudos de Segurança",
    "sociology": "Sociologia",
    "us_foreign_policy": "Política Externa dos EUA",
    "virology": "Virologia",
    "world_religions": "Religiões Mundiais",
}
MMLU_SUBJECTS_AR = {
    "abstract_algebra": "الجبر المجرد",
    "anatomy": "علم التشريح",
    "astronomy": "علم الفلك",
    "business_ethics": "أخلاقيات الأعمال",
    "clinical_knowledge": "المعرفة السريرية",
    "college_biology": "أحياء جامعية",
    "college_chemistry": "كيمياء جامعية",
    "college_computer_science": "علوم الحاسوب الجامعية",
    "college_mathematics": "رياضيات جامعية",
    "college_medicine": "طب جامعي",
    "college_physics": "فيزياء جامعية",
    "computer_security": "أمن الحاسوب",
    "conceptual_physics": "الفيزياء المفاهيمية",
    "econometrics": "الاقتصاد القياسي",
    "electrical_engineering": "الهندسة الكهربائية",
    "elementary_mathematics": "الرياضيات الابتدائية",
    "formal_logic": "المنطق الصوري",
    "global_facts": "حقائق عالمية",
    "high_school_biology": "أحياء ثانوية",
    "high_school_chemistry": "كيمياء ثانوية",
    "high_school_computer_science": "علوم الحاسوب الثانوية",
    "high_school_european_history": "تاريخ أوروبا الثانوي",
    "high_school_geography": "جغرافيا ثانوية",
    "high_school_government_and_politics": "الحكومة والسياسة الثانوية",
    "high_school_macroeconomics": "الاقتصاد الكلي الثانوي",
    "high_school_mathematics": "رياضيات ثانوية",
    "high_school_microeconomics": "الاقتصاد الجزئي الثانوي",
    "high_school_physics": "فيزياء ثانوية",
    "high_school_psychology": "علم النفس الثانوي",
    "high_school_statistics": "الإحصاء الثانوي",
    "high_school_us_history": "تاريخ الولايات المتحدة الثانوي",
    "high_school_world_history": "تاريخ العالم الثانوي",
    "human_aging": "شيخوخة الإنسان",
    "human_sexuality": "الجنس البشري",
    "international_law": "القانون الدولي",
    "jurisprudence": "الفقه القانوني",
    "logical_fallacies": "المغالطات المنطقية",
    "machine_learning": "تعلم الآلة",
    "management": "الإدارة",
    "marketing": "التسويق",
    "medical_genetics": "الوراثة الطبية",
    "miscellaneous": "متفرقات",
    "moral_disputes": "الخلافات الأخلاقية",
    "moral_scenarios": "السيناريوهات الأخلاقية",
    "nutrition": "التغذية",
    "philosophy": "الفلسفة",
    "prehistory": "ما قبل التاريخ",
    "professional_accounting": "المحاسبة المهنية",
    "professional_law": "القانون المهني",
    "professional_medicine": "الطب المهني",
    "professional_psychology": "علم النفس المهني",
    "public_relations": "العلاقات العامة",
    "security_studies": "دراسات الأمن",
    "sociology": "علم الاجتماع",
    "us_foreign_policy": "السياسة الخارجية الأمريكية",
    "virology": "علم الفيروسات",
    "world_religions": "الديانات العالمية",
}

LANGUAGE_SUBJECTS_MAP = {
    "FR_FR": MMLU_SUBJECTS_FR,
    "DE_DE": MMLU_SUBJECTS_DE,
    "ES_LA": MMLU_SUBJECTS_ES,
    "IT_IT": MMLU_SUBJECTS_IT,
    "PT_BR": MMLU_SUBJECTS_PT,
    "AR_XY": MMLU_SUBJECTS_AR,
}

LANGUAGE_INITIAL_PROMPT_TEXT_MAP = {
    "FR_FR": "Les questions suivantes sont des questions à choix multiples (avec réponses) sur",
    "DE_DE": "Die folgenden sind Multiple-Choice-Fragen (mit Antworten) über",
    "ES_LA": "Las siguientes son preguntas de opción múltiple (con respuestas) sobre",
    "IT_IT": "Le seguenti sono domande a scelta multipla (con risposte) su",
    "PT_BR": "As seguintes são perguntas de múltipla escolha (com respostas) sobre",
    "AR_XY": "فيما يلي أسئلة اختيار من متعدد (مع الإجابات) حول",
}

LANGUAGE_QUESTION_TEXT_MAP = {
    "FR_FR": "Question",
    "DE_DE": "Frage",
    "ES_LA": "Pregunta",
    "IT_IT": "Domanda",
    "PT_BR": "Pergunta",
    "AR_XY": "السؤال",
}

LANGUAGE_ANSWER_TEXT_MAP = {
    "FR_FR": "Réponse",
    "DE_DE": "Antwort",
    "ES_LA": "Respuesta",
    "IT_IT": "Risposta",
    "PT_BR": "Resposta",
    "AR_XY": "الإجابة",
}

LANGUAGE_NAME_MAP = {
    "FR_FR": Language.FRA,
    "DE_DE": Language.DEU,
    "ES_LA": Language.SPA,
    "IT_IT": Language.ITA,
    "PT_BR": Language.POR,
    "AR_XY": Language.ARB,
}


class MMMLU(BaseTask[tuple[str, str]]):
    """MMMLU dataset: https://huggingface.co/datasets/openai/MMMLU"""

    NAME = "MMMLU"
    DATASET_PATH = "openai/MMMLU"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = list(product(MMMLU_LANGS, MMLU_SUBJECTS))
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"] + get_n_letters(4)
    LANGUAGE = {
        str((lang_code.split("_")[0], subject)): LANGUAGE_NAME_MAP[lang_code]
        for lang_code, subjects in LANGUAGE_SUBJECTS_MAP.items()
        for subject in subjects
    }

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(4)

    def _load_dataset(self, subject: tuple[str, str]) -> None:
        lang, current_subject = subject
        hf_dataset = self._load_hf_dataset(path=self.DATASET_PATH, name=lang)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            data = data.filter(lambda x: x["Subject"] == current_subject)
            data_list = list(data)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = data_list

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        language_key = item["subject"][0]
        subject = LANGUAGE_SUBJECTS_MAP[language_key][item["Subject"]]
        return f"{LANGUAGE_INITIAL_PROMPT_TEXT_MAP[language_key]} {subject}."

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["Question"].strip()
        choices = "".join([f"{key}. {item[key]}\n" for key in self.keys])
        language_key = item["subject"][0]
        return f"{LANGUAGE_QUESTION_TEXT_MAP[language_key]}: {question}\n{choices}"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        language_key = item["subject"][0]
        return f"{LANGUAGE_ANSWER_TEXT_MAP[language_key]}:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return f" {item['Answer']}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {key}" for key in self.keys]


class MMMLU_GERMAN_COT(MMMLU):
    NAME = "MMMLU_GERMAN_COT"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion, GermanCompletionChecker]
    SUBJECTS = [("DE_DE", subject) for subject in MMLU_SUBJECTS]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Frage", "Question", "Answer"] + get_n_letters(4)
    LANGUAGE = {str(s): Language.DEU for s in list(product(["de"], MMLU_SUBJECTS))}

    ANS_RE = re.compile(r"Daher lautet die Antwort: ([ABCD])")

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for MMMLU_GERMAN_COT"
        super().__init__(num_fewshot)
        self.stop_sequences: list[str] = ["Frage:", "Question:"]

    def _extract_answer(self, completion: str) -> str:
        match = self.ANS_RE.search(completion)
        if match:
            match_str = match.group(1)
            return match_str
        else:
            return "[invalid]"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return ""

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return item["Answer"]

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return self._extract_answer(completion_text)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["Question"].strip()
        choices = "\n".join([f"{key}. {item[key]}" for key in self.keys])
        return f"Frage: {question}\n{choices}"

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            f"Die folgende Frage ist eine Multiple-Choice-Frage zum Thema {MMLU_SUBJECTS_DE[item['Subject']]}. "
            'Fasse deine Argumentation zusammen und schließe mit dem Satz: "Daher lautet die Antwort: X" '
            "ab, wobei X eine der Antworten A, B, C oder D ist. Benutze ausschließlich Deutsch."
        )
