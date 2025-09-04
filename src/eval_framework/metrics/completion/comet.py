import torch
from comet import download_model, load_from_checkpoint

from eval_framework.exceptions import LogicError
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import Completion, UntemplatedPrompt
from eval_framework.utils.constants import ROOT_DIR

SAVING_DIR = ROOT_DIR / "comet_model"


class COMET(BaseMetric[Completion]):
    """COMET is a neural, multilingual framework for evaluating machine translation quality by leveraging cross-lingual
    pretrained language models to achieve state-of-the-art correlation with human judgments
    Note: this requires a Hugging Face token with access to the model: https://huggingface.co/Unbabel/XCOMET-XL
    Source: https://github.com/Unbabel/COMET
    Paper: https://arxiv.org/abs/2009.09025
    """

    NAME = "COMET"

    def __init__(self) -> None:
        checkpoint_path = download_model("Unbabel/XCOMET-XL", saving_directory=SAVING_DIR)
        self.model = load_from_checkpoint(checkpoint_path)
        assert torch.cuda.is_available(), "COMET requires a GPU"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        if (
            response.context is None
            or not isinstance(response.context, UntemplatedPrompt)
            or response.context.untemplated_prompt == ""
        ):
            raise LogicError("When calculating COMET we need an untemplated prompt.")

        scores = []
        for ground_truth in response.ground_truth_list:
            if ground_truth == "" or ground_truth is None:
                raise LogicError("When calculating COMET we need a ground truth.")

            data = [
                {
                    "src": response.context.untemplated_prompt.strip(),
                    "mt": response.completion.strip(),
                    "ref": ground_truth.strip(),
                },
            ]
            with torch.no_grad():
                model_output = self.model.predict(data, gpus=1)
            scores.append(model_output.system_score)

        return [
            MetricResult(metric_name=self.NAME, value=float(max(scores)), higher_is_better=True, error=response.error)
        ]
