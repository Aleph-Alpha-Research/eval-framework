import re
from typing import Any

from eval_framework.metrics.completion.accuracy_completion import AccuracyCompletion
from eval_framework.tasks.base import BaseTask, Language, ResponseType, Sample

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

# Predefined fewshot examples
FEWSHOT_ITEMS = [
    {
        "question": (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
            "After they are done, there will be 21 trees. "
            "How many trees did the grove workers plant today?"
        ),
        "answer": (
            "There are 15 trees originally. Then there were 21 trees after some more were planted. "
            "So there must have been 21 - 15 = 6.\n#### 6"
        ),
    },
    {
        "question": (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?"
        ),
        "answer": "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
    },
    {
        "question": (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?"
        ),
        "answer": (
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
            "After eating 35, they had 74 - 35 = 39.\n#### 39"
        ),
    },
    {
        "question": (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
            "How many lollipops did Jason give to Denny?"
        ),
        "answer": (
            "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
            "So he gave Denny 20 - 12 = 8.\n#### 8"
        ),
    },
    {
        "question": (
            "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
            "How many toys does he have now?"
        ),
        "answer": (
            "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
            "5 + 4 = 9.\n#### 9"
        ),
    },
    {
        "question": (
            "There were nine computers in the server room. Five more computers were installed each day, "
            "from monday to thursday. "
            "How many computers are now in the server room?"
        ),
        "answer": (
            "There were originally 9 computers. For each of 4 days, 5 more computers were "
            "added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n#### 29"
        ),
    },
    {
        "question": (
            "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
            "How many golf balls did he have at the end of wednesday?"
        ),
        "answer": (
            "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
            "After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33"
        ),
    },
    {
        "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "answer": (
            "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
            "So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8"
        ),
    },
]


class GSM8K(BaseTask[str]):
    """GSM8K dataset: https://huggingface.co/datasets/openai/gsm8k"""

    NAME = "GSM8K"
    DATASET_PATH = "gsm8k"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AccuracyCompletion]
    SUBJECTS = ["main"]
    PERTURBATION_UNMODIFIABLE_WORDS = ["Question", "Answer"]
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)

        # until: https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml
        self.stop_sequences: list[str] = ["Question:"]
        self.max_tokens = 1600

    def _extract_answer(self, completion: str) -> str:
        match = ANS_RE.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return "[invalid]"

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        for stop_sequence in self.stop_sequences:
            if stop_sequence in completion_text:
                completion_text = completion_text.split(stop_sequence)[0]
        return self._extract_answer(completion_text)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return f"Question: {item['question']}\nAnswer:"

    def _get_fewshot_target_text(self, item: dict[str, Any]) -> str:
        return f" {item['answer']}"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return self._extract_answer(item["answer"])


class GSM8KLlamaVersion(GSM8K):
    NAME = "GSM8K Llama Version"
    FEWSHOT_SPLIT = ""  # Changed to empty string since we're using predefined examples

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot <= len(FEWSHOT_ITEMS), f"Fewshot larger than {len(FEWSHOT_ITEMS)} is not supported for GSM8K"
        super().__init__(num_fewshot)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        # Remove the bracketed computations from the question
        question = re.sub(r"<<.*?>>", "", item["question"])
        return f"Question: {question}\nAnswer:"

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        """Override to use predefined fewshot examples instead of sampling from dataset"""
        return FEWSHOT_ITEMS[: self.num_fewshot]
