from typing import Callable, Generator, List, Sequence

import pytest
from _pytest.fixtures import FixtureRequest

from eval_framework.llm.base import BaseLLM, Sample
from eval_framework.llm.models import (
    Bert,
    Llama31_8B_Instruct_API,
    Llama31_70B_Instruct_API,
    Llama31_405B_Instruct_API,
    Llama33_70B_Instruct_API,
    Pharia1_7B_Control_API,
    Phi3Mini4kInstruct,
    Poro_34bChat_API,
    Pythia410m,
    Qwen1_5B,
    SmolLM135M,
    Smollm135MInstruct,
    SmolLM_1_7B_Instruct,
    Viking_7b_API,
)
from eval_framework.shared.types import RawCompletion, RawLoglikelihood
from template_formatting.formatter import Message
from tests.mock_wandb import MockWandb, MockWandbRun, MockArtifact, MockArtifactFile, MockWandbApi
import importlib
import inspect


class MockLLM(BaseLLM):
    def __init__(self) -> None:
        self.generate_counter = 0
        self.logprob_counter = 0
        self.logprob_samples: list[Sample] = []

    def logprobs(self, samples: List[Sample]) -> List[RawLoglikelihood]:
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
        messages: List[Sequence[Message]],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> List[RawCompletion]:
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
    "SmolLM-1.7B-Instruct": SmolLM_1_7B_Instruct,
    "Phi3Mini4kInstruct": Phi3Mini4kInstruct,
    "llama-3.1-8b-instruct": Llama31_8B_Instruct_API,
    "llama-3.1-70b-instruct": Llama31_70B_Instruct_API,
    "llama-3.3-70b-instruct": Llama33_70B_Instruct_API,
    "llama-3.1-405b-instruct": Llama31_405B_Instruct_API,
    "pharia-1-llm-7b-control": Pharia1_7B_Control_API,
    "viking-7b": Viking_7b_API,
    "poro-34b-chat": Poro_34bChat_API,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": Qwen1_5B,
    "Bert": Bert,
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


def _resolve_dotted(dotted: str) -> object:
    """Resolve a dotted path like 'wandb.Artifact' to the actual object.

    :params dotted: The dotted path to resolve.
    :returns: The resolved object.
    """
    parts = dotted.split(".")
    obj = importlib.import_module(parts[0])
    for part in parts[1:]:
        obj = getattr(obj, part)
    return obj


def _auto_monkeypatch(monkeypatch: pytest.MonkeyPatch, target_dotted: str, mock_source: object) -> None:
    """Programmatically monkeypatch public attributes from `mock_source` onto
    the resolved `target_dotted` object.
    Copies all public attributes (no leading underscore) from the mock 
    instance or class to the target object.

    Additional attributes can still be added

    :params monkeypatch: The monkeypatch fixture instance.
    :params target_dotted: The dotted path to the target object.
    :params mock_source: The mock object to copy attributes from.
    :returns: None
    """
    target = _resolve_dotted(target_dotted)

    # Collect names from the instance and its class so we pick up methods and
    # assigned class attributes.
    names: set[str] = set()
    if hasattr(mock_source, "__dict__"):
        names.update(k for k in vars(mock_source).keys() if not k.startswith("_"))
    names.update(k for k in vars(type(mock_source)).keys() if not k.startswith("_"))

    for name in names:
        try:
            value = getattr(mock_source, name)
        except AttributeError:
            continue

        # If we're patching a class (like wandb.Artifact or wandb.Api) and the
        # mock value is callable, create a wrapper that drops the `self`
        # parameter and delegates to the mock. This preserves the method call
        if inspect.isclass(target) and callable(value):
            def _make_wrapper(fn):
                def _wrapped(_self, *a, **kw):
                    return fn(*a, **kw)

                return _wrapped

            patched = _make_wrapper(value)
            monkeypatch.setattr(target, name, patched, raising=False)
        else:
            monkeypatch.setattr(target, name, value, raising=False)


@pytest.fixture(autouse=True)
def mock_wandb(monkeypatch: pytest.MonkeyPatch) -> MockWandb:
    """Automatically mock wandb for all tests."""
    mock_wandb_instance = MockWandb()
    _auto_monkeypatch(monkeypatch, "wandb", mock_wandb_instance)
    return mock_wandb_instance

@pytest.fixture(autouse=True)
def mock_wandb_artifact(monkeypatch: pytest.MonkeyPatch) -> MockWandb:
    """Automatically mock wandb Artifact class methods."""
    mock_artifact_instance = MockArtifact("__mock_artifact__")
    _auto_monkeypatch(monkeypatch, "wandb.Artifact", mock_artifact_instance)
    return mock_artifact_instance


@pytest.fixture(autouse=True)
def mock_wandb_api(monkeypatch: pytest.MonkeyPatch) -> MockWandb:
    """Automatically mock wandb api for tests."""
    mock_api_instance = MockWandbApi()
    _auto_monkeypatch(monkeypatch, "wandb.Api", mock_api_instance)
    return mock_api_instance