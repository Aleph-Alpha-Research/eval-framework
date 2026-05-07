from __future__ import annotations

import pytest

from eval_framework.llm.vllm_local_server import VLLMLocalServerOpenAIModel
from template_formatting.formatter import Message, Role


@pytest.fixture(scope="session")
def vllm_local_http() -> VLLMLocalServerOpenAIModel:
    """
    Start one local `vllm serve` instance for the whole test session.
    This keeps GPU CI fast while still exercising the HTTP boundary.
    """
    llm = VLLMLocalServerOpenAIModel(
        model_name="Qwen/Qwen3-0.6B",
        # Keep resource use low for CI stability.
        tensor_parallel_size=1,
        gpu_memory_utilization=0.3,
        dtype="float16",
        max_model_len=256,
        enforce_eager=True,
        startup_timeout_s=300.0,
        # Use chat API path (no formatter): most compatible with OpenAI-like servers.
        formatter=None,
        temperature=0.0,
    )
    yield llm
    llm._cleanup()


@pytest.mark.vllm
@pytest.mark.gpu
def test_vllm_local_server_single_prompt(vllm_local_http: VLLMLocalServerOpenAIModel) -> None:
    out = vllm_local_http.generate_from_messages(
        [[Message(role=Role.USER, content="Reply with just: ok")]],
        max_tokens=8,
    )
    assert len(out) == 1
    assert out[0].completion.strip() != ""


@pytest.mark.vllm
@pytest.mark.gpu
def test_vllm_local_server_batching(vllm_local_http: VLLMLocalServerOpenAIModel) -> None:
    out = vllm_local_http.generate_from_messages(
        [
            [Message(role=Role.USER, content="Reply with just: A")],
            [Message(role=Role.USER, content="Reply with just: B")],
        ],
        max_tokens=4,
    )
    assert len(out) == 2
    assert all(o.completion.strip() != "" for o in out)


@pytest.mark.vllm
@pytest.mark.gpu
def test_vllm_local_server_stop_sequences(vllm_local_http: VLLMLocalServerOpenAIModel) -> None:
    out = vllm_local_http.generate_from_messages(
        [[Message(role=Role.USER, content="Write 'hello STOP world' verbatim.")]],
        stop_sequences=["STOP"],
        max_tokens=32,
    )
    assert len(out) == 1
    assert out[0].completion.strip() != ""


@pytest.mark.vllm
@pytest.mark.gpu
def test_vllm_local_server_temperature_and_top_p_override(vllm_local_http: VLLMLocalServerOpenAIModel) -> None:
    out = vllm_local_http.generate_from_messages(
        [[Message(role=Role.USER, content="Reply with a single short word.")]],
        max_tokens=8,
        temperature=0.7,
        top_p=0.9,
    )
    assert len(out) == 1
    assert out[0].completion.strip() != ""


@pytest.mark.vllm
@pytest.mark.gpu
def test_vllm_local_server_cleanup_is_idempotent(vllm_local_http: VLLMLocalServerOpenAIModel) -> None:
    # Should be safe to call multiple times (best-effort cleanup).
    vllm_local_http._cleanup()
    vllm_local_http._cleanup()
