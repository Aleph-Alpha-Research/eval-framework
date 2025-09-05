from collections.abc import Callable, Sequence

import pytest
from _pytest.fixtures import FixtureRequest

from eval_framework.llm.base import BaseLLM, Sample
from eval_framework.llm.huggingface import Pythia410m, SmolLM135M, Smollm135MInstruct
from eval_framework.llm.vllm import Qwen3_0_6B_VLLM
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from template_formatting.formatter import Message
from tests.mock_wandb import MockArtifact, MockWandb, MockWandbApi


class MockLLM(BaseLLM):
    def __init__(self) -> None:
        self.generate_counter = 0
        self.logprob_counter = 0
        self.logprob_samples: list[Sample] = []

    def logprobs(self, samples: list[Sample]) -> list[RawLoglikelihood]:
        rawloglikelihoods = []
        for sample in samples:
            self.logprob_counter += 1
            self.logprob_samples.append(sample)
            logprobs = {}

            for choice in sample.possible_completions:  # type: ignore
                logprobs[choice] = 0.01

            rawloglikelihoods.append(
                RawLoglikelihood(
                    prompt=" ".join(message.content for message in sample.messages),
                    prompt_sequence_positions=42,
                    loglikelihoods=logprobs,
                    loglikelihoods_sequence_positions={k: 1337 for k in logprobs.keys()},
                )
            )
        return rawloglikelihoods

    def generate_from_messages(
        self,
        messages: list[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[RawCompletion]:
        self.generate_counter += 1
        return [
            RawCompletion(
                prompt="",
                prompt_sequence_positions=0,
                completion=f"This is the a very fake message number {self.generate_counter}",
                completion_sequence_positions=42,
            )
            for _ in messages
        ]


model_dict = {
    "Pythia410m": Pythia410m,
    "SmolLM135M": SmolLM135M,
    "Smollm135MInstruct": Smollm135MInstruct,
    "Qwen3_0_6B_VLLM": Qwen3_0_6B_VLLM,
    "MockLLM": MockLLM,
}


@pytest.fixture()
def test_llms(request: FixtureRequest) -> BaseLLM:
    if request.param not in model_dict:
        raise ValueError(f"Unknown LLM name: {request.param}")
    return model_dict[request.param]()


@pytest.fixture
def should_preempt_callable() -> Callable[[], bool]:
    return lambda: False


@pytest.fixture(autouse=True)
def mock_wandb(monkeypatch: pytest.MonkeyPatch) -> MockWandb:
    mock_wandb_instance = MockWandb()
    monkeypatch.setattr("wandb.init", mock_wandb_instance.init)
    monkeypatch.setattr("wandb.log", mock_wandb_instance.log)
    monkeypatch.setattr("wandb.login", mock_wandb_instance.login)
    monkeypatch.setattr("wandb.finish", mock_wandb_instance.finish)
    monkeypatch.setattr("wandb.use_artifact", mock_wandb_instance.use_artifact)
    monkeypatch.setattr("wandb.Artifact", MockArtifact)
    monkeypatch.setattr("wandb.Api", MockWandbApi)
    return mock_wandb_instance


@pytest.fixture(autouse=True)
def mock_wandb_artifact(monkeypatch: pytest.MonkeyPatch) -> MockArtifact:
    # required by test_download_and_use_artifact
    mock_artifact_instance = MockArtifact("__mock_artifact__", "model")
    monkeypatch.setattr("wandb.Artifact.files", mock_artifact_instance.files)
    monkeypatch.setattr("wandb.Artifact.download", mock_artifact_instance.download)
    monkeypatch.setattr("wandb.Artifact.add_reference", mock_artifact_instance.add_reference)
    return mock_artifact_instance


@pytest.fixture(autouse=True)
def mock_wandb_api(monkeypatch: pytest.MonkeyPatch) -> MockWandbApi:
    """Automatically mock wandb api for tests."""
    mock_api_instance = MockWandbApi()
    monkeypatch.setattr("wandb.Api", MockWandbApi)
    monkeypatch.setattr("wandb.Api.artifact", MockWandbApi.artifact)
    return mock_api_instance
