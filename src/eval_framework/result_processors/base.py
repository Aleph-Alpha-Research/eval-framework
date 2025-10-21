from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

from eval_framework.shared.types import Completion, Error, Loglikelihood
from eval_framework.tasks.eval_config import EvalConfig

MAIN = "eval_framework_results"

load_dotenv()


class Result(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    subject: str
    num_fewshot: int
    llm_name: str
    task_name: str
    metric_class_name: str
    metric_name: str
    key: str | None
    value: float | None
    higher_is_better: bool
    prompt: str
    response: str
    llm_judge_prompt: str | None = None
    llm_judge_response: str | None = None
    code_execution_trace: str | None = None
    error: Error | None = None


class ResultProcessor(ABC):
    @abstractmethod
    def save_metadata(self, metadata: dict) -> None:
        """Save metadata."""
        pass

    @abstractmethod
    def load_metadata(self) -> dict:
        """Load metadata."""
        pass

    @abstractmethod
    def save_responses(self, responses: list[Completion | Loglikelihood]) -> None:
        """Save a list of response objects (overwrite a file)."""
        pass

    @abstractmethod
    def save_response(self, response: Completion | Loglikelihood) -> None:
        """Save a single response object (append into a file)."""
        pass

    @abstractmethod
    def load_responses(self) -> list[Completion | Loglikelihood]:
        """Load a list of response objects."""
        pass

    @abstractmethod
    def save_metrics_results(self, results: list[Result]) -> None:
        """Save the results of the metrics (overwrite a file)."""
        pass

    @abstractmethod
    def save_metrics_result(self, result: Result) -> None:
        """Save a single metric result (append into a file)."""
        pass

    @abstractmethod
    def save_aggregated_results(self, result: dict[str, float | None]) -> None:
        """Save the aggregated results."""
        pass

    @abstractmethod
    def load_metrics_results(self) -> list[Result]:
        """Load the aggregated results."""
        pass


class ResultsUploader(ABC):
    @abstractmethod
    def upload(self, llm_name: str, config: EvalConfig, output_dir: Path) -> bool:
        """Upload relevant parts from `output_dir` to the desired destination.
        Returns True if upload was successful, False otherwise.
        """
        pass
