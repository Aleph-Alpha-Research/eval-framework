import os
import random
import re
from typing import Any

from datasets import DatasetDict

from eval_framework.metrics.completion_metrics.struct_eval_metrics import RenderableStructMetric, StructMetric
from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.dataloader import Dataloader

StructEvalSubjects = [
    "CSV to YAML",
    "JSON to XML",
    "JSON to CSV",
    "XML to JSON",
    "XML to YAML",
    "Text to XML",
    "Text to YAML",
    "Text to TOML",
    "YAML to JSON",
    "TOML to JSON",
    "Text to CSV",
    "YAML to XML",
    "JSON to YAML",
    "TOML to YAML",
    "YAML to CSV",
    "CSV to JSON",
    "CSV to XML",
    "Text to JSON",
    "XML to CSV",
]


class StructEval(BaseTask[str]):
    """StructEval task: https://tiger-ai-lab.github.io/StructEval/"""

    NAME = "StructEval"
    DATASET_PATH = "TIGER-Lab/StructEval"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "train"  # Only has train split
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [StructMetric]  # Define appropriate metrics for StructEval
    SUBJECTS = StructEvalSubjects
    LANGUAGE = Language.ENG

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        if num_fewshot > 0:
            raise ValueError("StructEval only supports zero-shot evaluation.")
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)

    def _load_dataset(self, subject: str) -> None:
        hf_dataset = self.dataloader.load(path=self.DATASET_PATH, revision=self.HF_REVISION)
        assert isinstance(hf_dataset, DatasetDict), "Expected a Hugging Face Dataset object."
        hf_dataset = hf_dataset.filter(lambda item: item["task_name"] == subject, num_proc=os.cpu_count())
        self.dataset = {}
        self.rnd = random.Random(RANDOM_SEED)
        for split, data in hf_dataset.items():
            if split not in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                continue
            data_list = list(data)
            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            self.dataset[split] = data_list

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return (
            f"{item['query']}\n\nIMPORTANT: Only output the required output format. "
            "You must start the format/code with <|BEGIN_CODE|> and end the format/code with  <|END_CODE|>. "
            "No other text output (explanation, comments, etc.) are allowed.  Do not use markdown code fences.\n"
        )

    def _get_eval_kwargs(self, item: dict[str, Any]) -> dict[str, Any] | None:
        return {
            "output_type": item["output_type"],
            "paths": item["raw_output_metric"],
        }

    def _get_cue_text(self, item: dict[str, Any]) -> str:
        return "<|BEGIN_CODE|>"

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None | list[str]:
        return None

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        m = re.search(r"(?:<\|BEGIN_CODE\|>|```[\w+-]*)(.*?)(?:<\|END_CODE\|>|```*)", completion_text, re.DOTALL)
        return m.group(1).strip() if m else completion_text.strip()


# There are more subjects in the StructEval dataset, but currently only the HTML output metric is implemented.
RENDERABLE_STRUCTEVAL_SUBJECTS = [
    "Convert Markdown to HTML",
    "Convert React to HTML",
    "Convert Vue to HTML",
    "Text to HTML",
]


class RenderableStructEval(StructEval):
    """Renderable StructEval task for tasks that can be rendered visually."""

    NAME = "RenderableStructEval"
    SUBJECTS = RENDERABLE_STRUCTEVAL_SUBJECTS
    METRICS = [RenderableStructMetric]  # Define appropriate metrics for StructEval

    def _get_eval_kwargs(self, item: dict[str, Any]) -> dict[str, Any] | None:
        return {
            "output_type": item["output_type"],
            "keywords": item["raw_output_metric"],
        }
