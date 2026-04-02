from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import BaseTask, Language, ResponseType
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle, answer_key_to_index
from eval_framework.tasks.utils import get_n_letters

# OLMES fixed fewshot sources, keyed by HF subject name.
# Source: https://github.com/allenai/olmes  (FEWSHOT_SOURCES["OLMES:ARC-*"])
_ARC_FEWSHOT_SOURCES: dict[str, list[dict[str, Any]]] = {
    "ARC-Easy": [
        {
            "id": "MCAS_2007_8_5189",
            "question": "Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply "
            "to the fungi in this symbiotic relationship?",
            "choices": {"text": ["carbon dioxide", "food", "protection", "water"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "id": "Mercury_SC_401169",
            "question": "When a switch is used in an electrical circuit, the switch can",
            "choices": {
                "text": [
                    "cause the charge to build.",
                    "increase and decrease the voltage.",
                    "cause the current to change direction.",
                    "stop and start the flow of current.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "D",
        },
        {
            "id": "MCAS_2004_8_27",
            "question": "Which of the following is an example of an assistive device?",
            "choices": {
                "text": ["contact lens", "motorcycle", "raincoat", "coffee pot"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "NYSEDREGENTS_2006_8_10",
            "question": "Rocks are classified as igneous, metamorphic, or sedimentary according to",
            "choices": {
                "text": ["their color", "their shape", "how they formed", "the minerals they contain"],
                "label": ["1", "2", "3", "4"],
            },
            "answerKey": "3",
        },
        {
            "id": "Mercury_7013388",
            "question": "A chewable calcium carbonate tablet is a common treatment for stomach discomfort. Calcium "
            "carbonate is most likely used as this type of medicine because calcium carbonate",
            "choices": {
                "text": [
                    "has a pleasant flavor.",
                    "is inexpensive to produce.",
                    "neutralizes digestive acid.",
                    "occurs naturally in the body.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
        {
            "id": "Mercury_7179953",
            "question": "Which two body systems are directly involved in movement?",
            "choices": {
                "text": [
                    "muscular and skeletal",
                    "digestive and muscular",
                    "skeletal and respiratory",
                    "respiratory and digestive",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "Mercury_7205118",
            "question": "Which change in the state of water particles causes the particles to become arranged in a"
            " fixed position?",
            "choices": {"text": ["boiling", "melting", "freezing", "evaporating"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        },
        {
            "id": "MCAS_2016_8_13",
            "question": "Earth's core is primarily composed of which of the following materials?",
            "choices": {"text": ["basalt", "iron", "magma", "quartz"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
    ],
    "ARC-Challenge": [
        {
            "id": "Mercury_SC_415702",
            "question": "George wants to warm his hands quickly by rubbing them. Which skin surface will produce the "
            "most heat?",
            "choices": {
                "text": ["dry palms", "wet palms", "palms covered with oil", "palms covered with lotion"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "MCAS_2009_5_6516",
            "question": "Which of the following statements best explains why magnets usually stick to a refrigerator "
            "door?",
            "choices": {
                "text": [
                    "The refrigerator door is smooth.",
                    "The refrigerator door contains iron.",
                    "The refrigerator door is a good conductor.",
                    "The refrigerator door has electric wires in it.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_7233695",
            "question": "A fold observed in layers of sedimentary rock most likely resulted from the",
            "choices": {
                "text": [
                    "cooling of flowing magma.",
                    "converging of crustal plates.",
                    "deposition of river sediments.",
                    "solution of carbonate minerals.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_7041615",
            "question": "Which of these do scientists offer as the most recent explanation as to why many plants and "
            "animals died out at the end of the Mesozoic era?",
            "choices": {
                "text": [
                    "worldwide disease",
                    "global mountain building",
                    "rise of mammals that preyed upon plants and animals",
                    "impact of an asteroid created dust that blocked the sunlight",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "D",
        },
        {
            "id": "MCAS_1998_4_3",
            "question": "Which of the following is a trait that a dog does NOT inherit from its parents?",
            "choices": {
                "text": [
                    "the length of its fur",
                    "the shape of its nose",
                    "the size of its appetite",
                    "the color of its fur",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
        {
            "id": "Mercury_7041860",
            "question": "A boat is acted on by a river current flowing north and by wind blowing on its sails. The boat"
            " travels northeast. In which direction is the wind most likely applying force to the sails of the boat?",
            "choices": {"text": ["west", "east", "north", "south"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "id": "ACTAAP_2013_5_11",
            "question": "As part of an experiment, an astronaut takes a scale to the Moon and weighs himself. The scale"
            " reads 31 pounds. If the astronaut has a mass of about 84 kilograms, which are the approximate weight "
            "and mass of the astronaut when standing on the Earth?",
            "choices": {
                "text": [
                    "31 pounds and 14 kilograms",
                    "31 pounds and 84 kilograms",
                    "186 pounds and 14 kilograms",
                    "186 pounds and 84 kilograms",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "D",
        },
        {
            "id": "MDSA_2008_5_30",
            "question": "On Earth, water can be a solid, a liquid, or a gas. Which energy source has the greatest "
            "influence on the state of matter of water?",
            "choices": {
                "text": ["the sun", "the wind", "ocean currents", "the metal core"],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "A",
        },
        {
            "id": "MEA_2016_8_14",
            "question": "Which statement best compares single-celled and multi-celled organisms?",
            "choices": {
                "text": [
                    "Tissues in a single-celled organism are like the cells in a multi-celled organism.",
                    "The nucleus in a single-celled organism is like the skin of a multi-celled organism.",
                    "Organelles in a single-celled organism are like the organs in a multi-celled organism.",
                    "The cytoplasm in a single-celled organism is like the nervous system in a multi-celled organism.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
        {
            "id": "Mercury_SC_401653",
            "question": "Which land form is the result of the constructive force of a glacier?",
            "choices": {
                "text": [
                    "valleys carved by a moving glacier",
                    "piles of rocks deposited by a melting glacier",
                    "grooves created in a granite surface by a glacier",
                    "bedrock hills roughened by the passing of a glacier",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "B",
        },
        {
            "id": "Mercury_7106908",
            "question": "Hatchling sea turtles are typically dark in color. Occasionally, a sea turtle hatches that "
            "is almost white in color. When crawling from the nest on the beach to the ocean, this light-colored sea "
            "turtle could be at risk for sunburn. The light color of the turtles would most likely",
            "choices": {
                "text": [
                    "help the turtles have better chances at reproducing.",
                    "cause the shell of the sea turtles to become stronger.",
                    "reduce the chances of turtles surviving to reproduce.",
                    "help in the development of a new species of sea turtles.",
                ],
                "label": ["A", "B", "C", "D"],
            },
            "answerKey": "C",
        },
    ],
}  # noqa: E501


class ARC(BaseTask[str]):
    """ARC dataset: https://huggingface.co/datasets/allenai/ai2_arc"""

    NAME = "ARC"
    DATASET_PATH = "allenai/ai2_arc"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
    SUBJECTS = ["ARC-Easy", "ARC-Challenge"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"] + get_n_letters(5)
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        self.keys = get_n_letters(5)  # needs to be 5 because there is one sample with 5 answer possibilities
        self.num_to_letter = {str(i): letter for i, letter in enumerate(self.keys, start=1)}

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\n"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        ground_truth = self._get_ground_truth(item)
        assert ground_truth is not None
        return f"{self._get_cue_text(item)}{ground_truth}"

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "Answer:"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        answer_key = self.num_to_letter.get(item["answerKey"], item["answerKey"])
        return f" {item['choices']['text'][self.keys.index(answer_key)]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {choice}" for choice in item["choices"]["text"]]


class ARC_OLMES(ARC):
    """
    ARC with OLMES-style prompt: options shown with space-prefixed labels (" A.", " B.", ...);
    loglikelihood over " A"/" B"/ etc.
    """

    NAME = "ARC_OLMES"

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        question = item["question"]
        texts = item["choices"]["text"]
        labels = get_n_letters(len(texts))
        options = "\n".join(f" {label}. {t}" for label, t in zip(labels, texts))
        return f"Question: {question}\n{options}\n"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        answer_key = self.num_to_letter.get(item["answerKey"], item["answerKey"])
        idx = self.keys.index(answer_key) if answer_key in self.keys else 0
        labels = get_n_letters(len(item["choices"]["text"]))
        return f" {labels[idx]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        n = len(item["choices"]["text"])
        return [f" {label}" for label in get_n_letters(n)]


class ARC_IDK(ARC):
    NAME = "ARC_IDK"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore,
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Answer only if you are confident, since mistakes may be penalised, while correct answers receive points. "
            "It is acceptable to answer with 'I do not know' if you are unsure, and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" I do not know."]


class _ARCChoice_Base(BaseTask[str]):
    """Shared base for choice-based ARC variants (Cloze, MC, BPB).

    Subclasses set ``NAME`` and ``TASK_STYLER``; everything else is inherited.
    """

    DATASET_PATH = "allenai/ai2_arc"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = ["ARC-Easy", "ARC-Challenge"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question"] + get_n_letters(5)
    LANGUAGE = Language.ENG

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        return item["question"]

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return item["choices"]["text"]

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return answer_key_to_index(item["answerKey"])

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        subject = item.get("subject", "")
        return _ARC_FEWSHOT_SOURCES.get(subject, [])[: self.num_fewshot]


class ARCCloze(_ARCChoice_Base):
    NAME = "ARCCloze"
    TASK_STYLER = ClozeStyle()


class ARCMC(_ARCChoice_Base):
    """ARC with OLMES-style MC prompt: options listed as ' A. ...', scored over ' A'/' B'/...."""

    NAME = "ARCMC"
    TASK_STYLER = MCStyle(space_prefixed_labels=True)


class ARCBPB(_ARCChoice_Base):
    """BPB-only variant: scores loglikelihood over the ground-truth answer text only."""

    NAME = "ARCBPB"
    TASK_STYLER = BPBStyle()
