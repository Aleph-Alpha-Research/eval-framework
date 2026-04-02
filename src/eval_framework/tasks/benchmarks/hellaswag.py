import re
from typing import Any

from eval_framework.metrics.loglikelihood.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.metrics.loglikelihood.bits_per_byte import BitsPerByteLoglikelihood
from eval_framework.metrics.loglikelihood.confidence_weighted_accuracy import ConfidenceWeightedAccuracy
from eval_framework.metrics.loglikelihood.dcs import DistributionalCorrectnessScore
from eval_framework.metrics.loglikelihood.ternary import TernaryScore
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType
from eval_framework.tasks.task_style import BPBStyle, ClozeStyle, MCStyle

# fmt: off
# OLMES fixed fewshot sources for HellaSwag.
# Source: https://github.com/allenai/olmes (FEWSHOT_SOURCES["OLMES:HellaSwag"])
_HELLASWAG_FEWSHOTS: list[dict[str, Any]] = [
    {"ind": 12, "activity_label": "Health", "ctx_a": "[header] How to cope with suicidal thoughts [title] Put off any plans. [step] Promise yourself that you'll wait 48 hours before doing anything. Remember, thoughts don't have the power to force you to act.", "ctx_b": "", "endings": ["Even when you do, there may be a small image of the future still lurking around your brain. [substeps] For instance, don't tell yourself that you can't make it.", "You're doing something, and no one can force you to act. It's completely natural to feel negative thoughts before you act.", "Do not panic if people talk to you (even if it's about quitting smoking). Have a plan for how you're going to react to a group of people who bring on suicidal thoughts.", "Sometimes extreme pain can distort our perception. Waiting before taking action will give your mind time to clear."], "label": "3"},#noqa
    {"ind": 39, "activity_label": "Education and Communications", "ctx_a": "[header] How to make a liquid into a solid [title] Place a small open container of water in the freezer compartment of a class or home refrigerator. [title] Leave the water there for several hours or overnight. [title] Remove from the freezer and note what has occurred.", "ctx_b": "", "endings": ["[step] Water changes state from liquid to solid when it reaches a temperature of 0 degrees celsius, or 32 degrees fahrenheit. This is a simple example of changing from liquid to solid, or freezing.", "[substeps] Check that the container is completely dry, but no ice has formed. You should get a sample before disposing of it.", "[step] Don't drink and continue making liquid. [title] Separate the ice water if you're not used to using water.", "[title] Set a timer to check on the reaction. [step] The liquid should be safe to use again once the water has frozen completely and the food appears firm."], "label": "0"}, #noqa
    {"ind": 9, "activity_label": "Baking cookies", "ctx_a": "A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. The pans are filled with pastries and loaded into the oven.", "ctx_b": "a knife", "endings": ["is seen moving on a board and cutting out its contents.", "hits the peeled cheesecake, followed by sliced custard and still cooked ice cream.", "etches a shape into the inside of the baked pans.", "is used to cut cylinder shaped dough into rounds."], "label": "3"},#noqa
    {"ind": 47, "activity_label": "Starting a campfire", "ctx_a": "He takes his lighter and lights the newspaper in several places to start the fire. The bonfire starts burning and continues to burn.", "ctx_b": "he", "endings": ["plays with the dog and makes two cookies.", "adds a few more twigs to keep the flames burning.", "gets up and attempts to put a flag on it, fails and makes a complete ass out of himself.", "puts on equipment and stools."], "label": "1"},#noqa
    {"ind": 38, "activity_label": "Finance and Business", "ctx_a": "[header] How to write a method statement [title] Prepare to write by conducting a risk assessment--an in-depth examination of the task or process. [substeps] Identify the work hazards (those that could potentially cause poor health or personal harm) that are inherent in the task. Analyze what has been done about these hazards and if these measures are enough to reduce the harm potential to an acceptable level.", "ctx_b": "", "endings": ["Determine if there are further steps you would like to take. For example, if you want to write about looking as though you've truly experienced the problem in practice, doing a risk assessment may help you so further in mental illness.", "Review the information presented to the project and get an understanding of the hazards. [title] Organize and plan a rest period that will help the sanitation industry and forest service team manage the task more effectively.", "Decide what additional measures need to be taken to reduce harm if an acceptable level has not been met. [title] Begin to write your method statement, starting at the header.", "[title] Write the search code (cnet) heading. [step] To write an article or report, simply write the following code (cnet: alternative sources and outcomes."], "label": "2"},#noqa
    {"ind": 38, "activity_label": "Arm wrestling", "ctx_a": "Two bodybuilder women are seated at a table. They are arm wrestling, vieing to win.", "ctx_b": "when there", "endings": ["'s another wrestler, they finish wrestling him.", "is a winner they go cheer each other on.", "is a victor, the two women shake hands.", "is not a winner, they get a huge kick in the face and continue wrestling as the crowd cheers on."], "label": "2"},#noqa
    {"ind": 51, "activity_label": "Painting", "ctx_a": "A lady named linda, creator of paint along is demonstrating how to do an acrylic painting.", "ctx_b": "she", "endings": ["extensively paints from fabric and paint horse tails on a painting screen.", "starts with a one inch flat brush and yellow and white acrylic paint.", "shows off her paint thinner and begins to tell her story about the underground bottle of magenta paints.", "demonstrates how to bring a window down from the wall."], "label": "1"},#noqa
    {"ind": 63, "activity_label": "Fixing the roof", "ctx_a": "A woman with long, black, curly hair is wearing casual wear, talking, and squatting on a roof.", "ctx_b": "the woman", "endings": ["then stands up and walks to a part of the roof where she lifts up a black shingle on the roof.", "turns on a machine attached to a hand cart with multiple metal rails and drives it underneath a large roof.", "raise her left leg to the graffiti, move it partially along, and just gets herself started climbing the tiles.", "holds her back while she works on the roof, she holds her legs behind her legs."], "label": "0"},#noqa
    {"ind": 4, "activity_label": "Removing ice from car", "ctx_a": "Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles.", "ctx_b": "then", "endings": [", the man adds wax to the windshield and cuts it.", ", a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.", ", the man puts on a christmas coat, knitted with netting.", ", the man continues removing the snow on his car."], "label": "3"},#noqa
    {"ind": 30, "activity_label": "Getting a haircut", "ctx_a": "The man in the blue shirt sits on the chair next to the sink. The other man begins washing his hair. He scrubs in the shampoo and then washes it off.", "ctx_b": "he", "endings": ["then combs it and blow dries his hair after styling it with gel.", "shows the razor that he has for shaving his hair.", "hair is now dry, he is on his way to the barber.", "moves the bucket to the other side of the sink and continues washing his hair."], "label": "0"},#noqa
    {"ind": 61, "activity_label": "Brushing teeth", "ctx_a": "A little boy walk toward the sink.", "ctx_b": "the boy", "endings": ["falling shits his pants from the bottom out.", "stands water to rinse his mouth.", "stands on front the sink and puts toothpaste on the brush, and then brush the teeth.", "rinses his cup in the pot, then put glasses on it."], "label": "2"},#noqa
]
# fmt: on


class HELLASWAG(BaseTask[str]):
    """Hellaswag dataset: https://huggingface.co/datasets/Rowan/hellaswag
    available data set sections: train, validation, test"""

    NAME = "HellaSwag"
    DATASET_PATH = "Rowan/hellaswag"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood, BitsPerByteLoglikelihood]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    @staticmethod
    def _preprocess(prompt: str) -> str:
        # remove bracketed text
        prompt = prompt.strip()
        prompt = prompt.replace(" [title]", ". ")
        prompt = re.sub("\\[.*?\\]", "", prompt)
        prompt = prompt.replace("  ", " ")
        return prompt

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        subject = self._preprocess(item["activity_label"])
        question = self._preprocess(item["ctx_a"] + " " + item["ctx_b"].capitalize()).strip()
        return f"{subject}: {question}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        ground_truth_index = int(item["label"] if item["label"] != "" else 0)
        choices = [self._preprocess(ending) for ending in item["endings"]]
        return f" {choices[ground_truth_index]}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        return [f" {self._preprocess(ending)}" for ending in item["endings"]]


class HELLASWAG_OLMES(HELLASWAG):
    NAME = "HellaSwag_OLMES"
    SAMPLE_SPLIT = "train"


class HELLASWAG_IDK(HELLASWAG):
    NAME = "HellaSwag_IDK"
    METRICS = [
        AccuracyLoglikelihood,
        AccuracyNormLoglikelihood,
        ConfidenceWeightedAccuracy,
        DistributionalCorrectnessScore,
        TernaryScore,
    ]

    def _get_initial_prompt_text(self, item: dict[str, Any]) -> str:
        return (
            "Complete the sentence only if you are confident, since mistakes may be penalised, while correct "
            "completions receive points. It is acceptable to answer with 'I do not know' if you are unsure, "
            "and you will receive 0 points."
        )

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        completions = super()._get_possible_completions(item)
        return (completions or []) + [" I do not know."]


class _HELLASWAG_Base(BaseTask[str]):
    """Shared base for HELLASWAG variants (Cloze, MC, BPB).

    Subclasses set ``NAME`` and ``TASK_STYLER``; everything else is inherited.
    """

    DATASET_PATH = "Rowan/hellaswag"
    SAMPLE_SPLIT = "validation"
    FEWSHOT_SPLIT = "train"
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = Language.ENG

    @staticmethod
    def _preprocess(prompt: str) -> str:
        # remove bracketed text
        prompt = prompt.strip()
        prompt = prompt.replace(" [title]", ". ")
        prompt = re.sub("\\[.*?\\]", "", prompt)
        prompt = prompt.replace("  ", " ")
        prompt = re.sub(r"\.\. ", ". ", prompt)
        return prompt

    def _get_choices(self, item: dict[str, Any]) -> list[str]:
        return [self._preprocess(ending) for ending in item["endings"]]

    def _get_raw_question(self, item: dict[str, Any]) -> str:
        # Include activity_label as prefix to match the OLMES prompt format:
        # "ActivityLabel: preprocessed_context"
        subject = self._preprocess(item["activity_label"])
        context = self._preprocess(item["ctx_a"] + " " + item["ctx_b"].capitalize()).strip()
        return f"{subject}: {context}"

    def _get_correct_index(self, item: dict[str, Any]) -> int:
        return int(item["label"] if item["label"] != "" else 0)

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        return _HELLASWAG_FEWSHOTS[: self.num_fewshot]


class HELLASWAGCloze(_HELLASWAG_Base):
    NAME = "HELLASWAGCloze"
    TASK_STYLER = ClozeStyle()


class HELLASWAGMC(_HELLASWAG_Base):
    NAME = "HELLASWAGMC"
    TASK_STYLER = MCStyle(space_prefixed_labels=True)


class HELLASWAGBPB(_HELLASWAG_Base):
    NAME = "HellaSwagBPB"
    TASK_STYLER = BPBStyle(question_prefix="", cue_text="", trailing_newline=False)
