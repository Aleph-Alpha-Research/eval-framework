import random
from typing import Any

from eval_framework.metrics.loglikelihood_metrics.accuracy_loglikelihood import (
    AccuracyLoglikelihood,
    AccuracyNormLoglikelihood,
)
from eval_framework.tasks.base import NO_SUBJECT, RANDOM_SEED, BaseTask, Language, ResponseType


class CASEHOLD(BaseTask[str]):
    NAME = "CaseHold"
    DATASET_PATH = "lex_glue"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.LOGLIKELIHOODS
    METRICS = [AccuracyLoglikelihood, AccuracyNormLoglikelihood]
    SUBJECTS = ["case_hold"]
    LANGUAGE = Language.ENG

    def _load_dataset(self, subject: str) -> None:
        name = subject if subject != NO_SUBJECT else None

        hf_dataset = self.dataloader.load(path=self.DATASET_PATH, name=name)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            data_list = list(data)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = [i for i in data_list if i["context"].count("(<HOLDING>)") == 1]

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return item["context"].split("(<HOLDING>)", maxsplit=1)[0]

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        right = item["context"].split("(<HOLDING>)", maxsplit=1)[1]
        return f"{item['endings'][item['label']]}{right}"

    def _get_possible_completions(self, item: dict[str, Any]) -> list[str] | None:
        right = item["context"].split("(<HOLDING>)", maxsplit=1)[1]
        return [f"{ending}{right}" for ending in item["endings"]]
