import csv
import json
import random
import re
import tempfile
from itertools import product
from typing import Any

from eval_framework.exceptions import LogicError
from eval_framework.metrics.completion_metrics.rouge_l import ROUGE_L
from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Language, ResponseType, Sample
from eval_framework.tasks.dataloader import Dataloader
from eval_framework.tasks.utils import run_python_code
from template_formatting.formatter import Role

TABLE_BENCH_SUBJECTS = [
    "NumericalReasoning",
    "DataAnalysis",
    "FactChecking",
    # "Visualization" task is complex to re-implement, of small relevance and of small size (5.6% of dataset, Language)
    # see https://github.com/TableBench/TableBench/blob/main/eval/batch_parse_response_script.py#L56
]

TABLE_BENCH_INSTRUCTION_TYPES = [
    # "DP",  # Direct Prompting, has been deleted: https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench-Instructions/commit/534a6d859494c370f2aa6ee0e6076103d9707560 # noqa: E501
    "PoT",  # Program-of-thought
    "SCoT",  # Symbolic chain-of-thought
    "TCoT",  # Textual chain-of-thought
]


class TableBench(BaseTask[tuple[str, str]]):
    """TableBench dataset: https://huggingface.co/datasets/Multilingual-Multimodal-NLP/TableBench"""

    NAME = "TableBench"
    DATASET_PATH = "Multilingual-Multimodal-NLP/TableBench"
    HF_REVISION = "81b551c744b7f49cfa0ad69cb7a1465d865c206e"  # latest version of the dataset is corrupted
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = "test"  # (there is no dedicated split, few-shot is not expected for this dataset)
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [ROUGE_L]
    SUBJECTS = list(product(TABLE_BENCH_INSTRUCTION_TYPES, TABLE_BENCH_SUBJECTS))
    LANGUAGE = Language.ENG

    def __init__(self, dataloader: Dataloader, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for TableBench"
        super().__init__(num_fewshot=num_fewshot, dataloader=dataloader)

    def _load_dataset(self, subject: tuple[str, str]) -> None:
        instruction_type, qtype = subject
        hf_dataset = self.dataloader.load(path=self.DATASET_PATH, name=None, revision=self.HF_REVISION)
        self.dataset = {}

        self.rnd = random.Random(RANDOM_SEED)

        for split, data in hf_dataset.items():
            data = data.filter(lambda x: x["qtype"] == qtype and x["instruction_type"] == instruction_type)
            data_list = list(data)

            if split == self.SAMPLE_SPLIT:
                self.rnd.shuffle(data_list)

            if split in [self.SAMPLE_SPLIT, self.FEWSHOT_SPLIT]:
                self.dataset[split] = data_list

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return item["instruction"]

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return item["answer"]

    def post_process_generated_completion(self, completion_text: str, sample: Sample | None = None) -> str:
        assert sample is not None
        if "PoT" in sample.subject:
            # Extract the (last) generated code snippet or fail otherwise
            try:
                matches = re.findall(r"```python\n(.*?)```", completion_text, flags=re.S)
                if not matches:
                    return ""
                code = matches[-1]
            except Exception:
                return ""

            # Extract the table given in the prompt and prepare it as a file
            instruction = [m.content for m in sample.messages if m.role == Role.USER][-1]
            tables = re.findall(r"\[TABLE\] (.*?) Let's get start!", instruction, flags=re.S)
            if not tables:
                return ""

            # Check if the tables is a list or a string
            if isinstance(tables, str):
                table_dict = json.loads(tables.strip())
            elif isinstance(tables, list):
                table_dict = json.loads(tables[0].strip())
            else:
                raise LogicError(f"TableBench: {instruction} does not seem to contain one table.")

            with tempfile.TemporaryDirectory() as tmpdirname:
                filename = f"{tmpdirname}/table.csv"
                with open(filename, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(table_dict["columns"])
                    writer.writerows(table_dict["data"])

                # Run the code in a Docker image, providing the table from the prompt
                completion_text = run_python_code(
                    code, image="amancevice/pandas:slim", input_files=[(filename, "/var/lib/pandas/table.csv")]
                )

                if "Error" in completion_text:
                    return ""

        # Extract the answer, be it directly from the model or be it the result of the generated code
        try:
            match = re.search(r"Final Answer: (.+)", completion_text)
            return match.group(1).strip() if match else ""
        except Exception:
            return ""
