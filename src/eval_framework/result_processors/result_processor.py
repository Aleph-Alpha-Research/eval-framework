import hashlib
import importlib.metadata
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import jsonlines
import wandb
from pydantic import RootModel

from eval_framework.llm.base import BaseLLM
from eval_framework.result_processors.base import Result, ResultProcessor
from eval_framework.shared.types import Completion, Loglikelihood
from eval_framework.tasks.eval_config import EvalConfig

MAIN = "eval_framework_results"
logger = logging.getLogger(__name__)


class ResultsFileProcessor(ResultProcessor):
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def save_metadata(self, metadata: dict) -> None:
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)

    def load_metadata(self) -> dict:
        metadata_file = self.output_dir / "metadata.json"
        if os.path.exists(metadata_file):
            with open(metadata_file) as f:
                return json.load(f)
        else:
            logger.info("No metadata found.")
            return {}

    def save_responses(self, responses: list[Completion | Loglikelihood]) -> None:
        responses_data = [response.model_dump(mode="json", serialize_as_any=True) for response in responses]
        with jsonlines.open(self.output_dir / "output.jsonl", "w") as f:
            f.write_all(responses_data)

    def save_response(self, response: Completion | Loglikelihood) -> None:
        with jsonlines.open(self.output_dir / "output.jsonl", "a") as f:
            f.write(response.model_dump(mode="json", serialize_as_any=True))

    def load_responses(self) -> list[Completion | Loglikelihood]:
        output_file = self.output_dir / "output.jsonl"
        broken_file = output_file.with_suffix(f".jsonl.broken.{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        try:
            Item = RootModel[Loglikelihood | Completion]
            with jsonlines.open(output_file, "r") as f:
                responses = [Item.model_validate(x).root for x in f]
        except FileNotFoundError:
            logger.info("No saved completions found.")
            responses = []
        except (json.decoder.JSONDecodeError, jsonlines.jsonlines.InvalidLineError):
            logger.info(f"Error decoding JSON, the file is corrupted. It will be renamed to {broken_file} and ignored")
            output_file.rename(broken_file)
            responses = []

        ids_list = [(resp.id, resp.subject) for resp in responses]
        if len(ids_list) != len(set(ids_list)) and "mtbench" not in str(output_file):
            logger.info(
                f"Error: {len(ids_list) - len(set(ids_list))} duplicate response IDs found, the file is corrupted. "
                f"It will be renamed to {broken_file} and ignored"
            )
            output_file.rename(broken_file)
            responses = []

        return responses

    def save_metrics_results(self, results: list[Result]) -> None:
        result_data = [x.model_dump(mode="json") for x in results]
        with jsonlines.open(self.output_dir / "results.jsonl", "w") as f:
            f.write_all(result_data)

    def save_metrics_result(self, result: Result) -> None:
        with jsonlines.open(self.output_dir / "results.jsonl", "a") as f:
            f.write(result.model_dump(mode="json"))

    def save_aggregated_results(self, results: dict[str, float | None]) -> None:
        with open(self.output_dir / "aggregated_results.json", "w") as f:
            json.dump(results, f, indent=4, sort_keys=True)

    def load_metrics_results(self) -> list[Result]:
        results_file = self.output_dir / "results.jsonl"
        try:
            with jsonlines.open(results_file, "r") as f:
                result_data = [Result.model_validate(x) for x in f]
            return result_data
        except FileNotFoundError:
            logger.info("No saved metrics found.")
            return []


def generate_output_dir(llm_name: str, config: EvalConfig) -> Path:
    # get the package version
    version_str = f"v{importlib.metadata.version('eval_framework')}"

    # Handle None values for num_fewshot and num_samples
    fewshot_str = f"fewshot_{config.num_fewshot}" if config.num_fewshot is not None else "fewshot_None"
    samples_str = f"samples_{config.num_samples}" if config.num_samples is not None else "samples_None"
    tokens_str = f"tokens_{config.max_tokens}" if config.max_tokens is not None else ""

    # Serialize key parameters for inclusion in the name
    params_str = f"{fewshot_str}__{samples_str}"
    if tokens_str:
        params_str += f"__{tokens_str}"

    # Serialize the full config for hashing
    # Convert the config to a dict and sort keys to ensure consistent hashing
    config_json = config.model_json_robust_subset_dump()
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest()[:5]  # Short hash of 5 characters

    # Include the hash in the directory name
    dir_name = f"{params_str}_{config_hash}"

    # add timestamp to dir in debug mode
    if os.environ.get("DEBUG", "FALSE").lower() == "true":
        # Generate the timestamp
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        dir_name += f"_{timestamp}"

    # Combine all components to form the full output directory path
    output_dir = config.output_dir / llm_name / f"{version_str}_{config.task_name}" / dir_name

    return output_dir


class ResultsWandbProcessor(ResultProcessor):
    def __init__(self, llm: BaseLLM, config: EvalConfig) -> None:
        self._checkpoint_step = config.wandb_checkpoint_step
        if hasattr(llm, "artifact"):
            if "global-step" in llm.artifact.metadata and self._checkpoint_step is None:
                self._checkpoint_step = int(llm.artifact.metadata["global-step"])

    def save_aggregated_results(self, results: dict[str, float | None]) -> None:
        if wandb.run is not None:
            if self._checkpoint_step is not None:
                # Step must be part of the `data`, `step` argument is not supported when sharing wandb run ids and
                #  w&b would need to write steps in consecutive order anyway (which is not given with separate evals).
                # Note also that the last run (which may not have the largest step number) appears as summary metrics.
                wandb.run.define_metric("*", step_metric="checkpoint_step")
                wandb.log(results | {"checkpoint_step": self._checkpoint_step})
            else:
                wandb.log(results)

    def save_metadata(self, metadata: dict) -> None:
        raise NotImplementedError

    def load_metadata(self) -> dict:
        raise NotImplementedError

    def save_responses(self, responses: list[Completion | Loglikelihood]) -> None:
        raise NotImplementedError

    def save_response(self, response: Completion | Loglikelihood) -> None:
        raise NotImplementedError

    def load_responses(self) -> list[Completion | Loglikelihood]:
        raise NotImplementedError

    def save_metrics_results(self, results: list[Result]) -> None:
        raise NotImplementedError

    def save_metrics_result(self, result: Result) -> None:
        raise NotImplementedError

    def load_metrics_results(self) -> list[Result]:
        raise NotImplementedError
