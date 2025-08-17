from typing import Any

from eval_framework.metrics.completion_metrics.ifeval import IFEvalMetric, IFEvalMetricContext
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType


class IFEval(BaseTask[str]):
    """IFEval: Instruction Following Eval (https://arxiv.org/pdf/2311.07911)."""

    NAME = "IFEval"
    DATASET_PATH = "google/IFEval"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [IFEvalMetric]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = {NO_SUBJECT: Language.ENG}

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        assert num_fewshot == 0, "IFEval does not support few-shot prompting."

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        return item["prompt"]

    def _get_context(self, item: dict[str, Any]) -> IFEvalMetricContext:
        assert "key" in item, "Expected 'key' in item"
        assert "instruction_id_list" in item, "Expected 'instruction_id_list' in item"
        assert "prompt" in item, "Expected 'prompt' in item"
        assert "kwargs" in item, "Expected 'kwargs' in item"

        new_kwargs = []
        for d in item["kwargs"]:
            # fixing undesired float fields in the dataset
            assert all([abs(v - float(v)) < 1e-5 for v in d.values() if isinstance(v, float)])
            new_kwargs.append({k: v if not isinstance(v, float) else int(v) for k, v in d.items()})

        # fixing changes to the HF dataset done on Apr 10 2025
        if item["key"] == 142:
            new_kwargs[2]["relation"] = None
            new_kwargs[2]["frequency"] = None
            new_kwargs[2]["keywords"] = new_kwargs[2]["keyword"]
            del new_kwargs[2]["keyword"]
        if item["key"] == 1512:
            new_kwargs[0]["relation"] = None

        item["kwargs"] = new_kwargs

        return IFEvalMetricContext(
            key=item["key"],
            instruction_id_list=item["instruction_id_list"],
            prompt=item["prompt"],
            additional_kwargs=item["kwargs"],
        )

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return None

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        return []


class IFEvalFiSv(IFEval):
    """Machine translated versions of the Instruction Following Evaluation (IFEval) benchmark."""

    NAME = "IFEval Finnish & Swedish"
    DATASET_PATH = "LumiOpen/ifeval_mt"
    SUBJECTS = ["fi", "sv"]
    LANGUAGE = {"fi": Language.FIN, "sv": Language.SWE}


class IFEvalDe(IFEval):
    """German version of the Instruction Following Evaluation (IFEval) benchmark."""

    NAME = "IFEval German"
    DATASET_PATH = "jzhang86/de_ifeval"
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = {NO_SUBJECT: Language.DEU}
