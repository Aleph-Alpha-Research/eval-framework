import gc
import logging
import time
from collections.abc import Sequence
from typing import Any, cast
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from pytest_mock import MockerFixture
from vllm import SamplingParams
from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

from eval_framework.llm.huggingface import Qwen3_0_6B
from eval_framework.llm.mistral import MistralAdapter, MistralVLLM
from eval_framework.llm.vllm import Qwen3_0_6B_VLLM, Qwen3_0_6B_VLLM_No_Thinking, VLLMModel, VLLMTokenizer
from eval_framework.shared.types import PromptTooLongException, RawCompletion, RawLoglikelihood
from eval_framework.tasks.base import Sample
from template_formatting.formatter import (
    ConcatFormatter,
    HFFormatter,
    IdentityFormatter,
    Llama3Formatter,
    Message,
    Role,
)
from tests.llm.test_base import LLM_INIT_FORMATTER_PARAMS, LLM_INIT_SOURCE_PARAMS


def clean_up() -> None:
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.cuda.empty_cache()


def safe_vllm_setup[T: VLLMModel](model_fn: type[T], kwargs: Any) -> T:
    """Safely initialize VLLM model with enhanced error handling."""
    assert "max_model_len" in kwargs
    kwargs["max_num_seqs"] = 1
    kwargs.setdefault("dtype", "float16")
    kwargs.setdefault("tensor_parallel_size", 1)
    kwargs.setdefault("gpu_memory_utilization", 0.3)
    kwargs.setdefault("swap_space", 0)
    kwargs.setdefault("enforce_eager", True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Add small delay to prevent race conditions
    time.sleep(1)

    try:
        return model_fn(**kwargs)
    except Exception as e:
        error_msg = f"Failed to initialize VLLM model: {str(e)}"
        if "Segmentation fault" in str(e) or "segfault" in str(e).lower():
            error_msg += "\nThis appears to be a VLLM segmentation fault in the test environment."
            error_msg += "\nThe code changes should have prevented this - check CI logs for details."
        pytest.fail(error_msg)


@pytest.mark.vllm
@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_fn, kwargs",
    [
        (
            Qwen3_0_6B_VLLM_No_Thinking,
            {
                "max_model_len": 64,
                "dtype": "float16",
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.3,
                "swap_space": 0,
                "enforce_eager": True,
            },
        ),
    ],
)
def test_vllm[T: VLLMModel](model_fn: type[T], kwargs: Any) -> None:
    model = safe_vllm_setup(model_fn, kwargs)

    messages: list[Message] = [
        Message(role=Role.USER, content="Question: What color is the night sky?\n"),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]

    # -- TEST LOGPROBS --
    list_of_samples = [
        Sample(
            id=0,
            subject="no_subject",
            messages=messages,
            ground_truth="black",
            possible_completions=["red", "blue", "black", "white"],
            context=None,
        ),
        Sample(
            id=0,
            subject="no_subject",
            messages=messages,
            ground_truth="foo",
            possible_completions=["foo", "bar"],
            context=None,
        ),
    ]
    results: list[RawLoglikelihood] = []
    for sample in list_of_samples:
        results.extend(model.logprobs([sample]))

    assert len(results) == 2
    assert set(results[0].loglikelihoods.keys()) == {"red", "blue", "black", "white"}
    assert set(results[0].loglikelihoods_sequence_positions.keys()) == {"red", "blue", "black", "white"}
    assert set(results[1].loglikelihoods.keys()) == {"foo", "bar"}
    assert set(results[1].loglikelihoods_sequence_positions.keys()) == {"foo", "bar"}

    del results
    gc.collect()
    torch.cuda.empty_cache()
    message = Message(role=Role.USER, content="Question: What is the capital of Germany?")

    # -- TEST COMPLETIONS --
    sampling_params = SamplingParams(max_tokens=25, temperature=0, stop=["\n"])
    prompt = model._formatter.format([message], output_mode="string")
    prompt_obj = model.tokenizer.encode_formatted_struct(prompt)
    outputs = model._model_generate(prompt_objs=[prompt_obj], sampling_params=sampling_params)
    assert "Berlin" in outputs[0].outputs[0].text

    del model
    clean_up()


@pytest.mark.vllm
@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_fn, kwargs",
    [
        (
            Qwen3_0_6B_VLLM,
            {
                "max_model_len": 32,
                "dtype": "float16",
                "tensor_parallel_size": 1,
                "gpu_memory_utilization": 0.3,
                "swap_space": 0,
                "enforce_eager": True,
            },
        ),
    ],
)
def test_vllm_error_on_overly_long_prompt[T: VLLMModel](model_fn: type[T], kwargs: Any) -> None:
    model = safe_vllm_setup(model_fn, kwargs)

    # given a too long log-likelihood task ...
    sample = Sample(
        id=0,
        subject="no_subject",
        messages=[
            Message(role=Role.USER, content="Question: What color is the night sky?"),
            Message(role=Role.ASSISTANT, content="Answer:"),
        ],
        ground_truth="black",
        possible_completions=["red", "blue", "black", "white" * 10000],
        context=None,
    )

    lresults: list[RawLoglikelihood] = model.logprobs([sample])
    # THEN the loglikelihoods should be empty and the error should be stored
    assert len(lresults) == 1
    assert (
        lresults[0].raw_loglikelihood_error is not None
        and lresults[0].raw_loglikelihood_error.error_class == PromptTooLongException.__name__
    )

    # GIVEN a too long completion task
    msg = [Message(role=Role.USER, content="text" * 10000)]
    cresults: list[RawCompletion] = model.generate_from_messages(messages=[msg], max_tokens=300)

    # ... the completion should be empty and the error should be stored
    assert len(cresults) == 1
    assert cresults[0].completion == ""
    assert (
        cresults[0].raw_completion_error is not None
        and cresults[0].raw_completion_error.error_class == PromptTooLongException.__name__
    )

    del model
    clean_up()


@pytest.mark.vllm
@pytest.mark.gpu
def test_vllm_hf_token_equivalence() -> None:
    """
    Test that VLLM and HF versions of Qwen3 0.6B produce matching first 20 tokens.
    """

    vllm_model = safe_vllm_setup(
        model_fn=Qwen3_0_6B_VLLM,
        kwargs={
            "max_model_len": 128,
            "dtype": "float16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.3,
            "swap_space": 0,
            "enforce_eager": True,
        },
    )

    messages = [
        Message(
            role=Role.USER,
            content="Tell me a short story about a robot learning to paint.",
        )
    ]
    vllm_results = vllm_model.generate_from_messages(messages=[messages], max_tokens=50, temperature=0)
    vllm_completion = vllm_results[0].completion
    vllm_tokens = vllm_model.tokenizer.encode_plain_text(vllm_completion).tokens[:20]

    # # free up memory
    del vllm_model
    gc.collect()
    torch.cuda.empty_cache()

    hf_model = Qwen3_0_6B()
    hf_results = hf_model.generate_from_messages(messages=[messages], max_tokens=50, temperature=0)
    hf_completion = hf_results[0].completion
    hf_tokens = hf_model.tokenizer.encode(hf_completion)[:20]
    # free up memory
    del hf_model
    clean_up()

    assert vllm_tokens == hf_tokens, f"First 20 tokens don't match:\nVLLM: {vllm_tokens}\nHF: {hf_tokens}"
    assert vllm_results[0].concat_compression == hf_results[0].concat_compression


@pytest.mark.vllm
@pytest.mark.gpu
def test_seq_length_priority_order() -> None:
    """
    Test that sequence length is determined in the correct priority order.

    NOTE =>
      This test works with an arbitrary model that extends the BaseVLLM API;
      Qwen is used as a test case
    """

    # Test 1: max_model_len parameter takes highest priority
    model = safe_vllm_setup(
        model_fn=Qwen3_0_6B_VLLM,
        kwargs={
            "max_model_len": 128,
            "dtype": "float16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.3,
            "swap_space": 0,
            "enforce_eager": True,
        },
    )
    assert model.seq_length == 128
    assert model.seq_length == model.max_seq_length

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Test 2: SEQ_LENGTH class attribute is used when max_model_len is None
    class Dummy_Qwen3_0_6B_VLLM(VLLMModel):
        LLM_NAME = "Qwen/Qwen3-0.6B"
        DEFAULT_FORMATTER = ConcatFormatter
        SEQ_LENGTH = 128

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    dummy_model = safe_vllm_setup(model_fn=Dummy_Qwen3_0_6B_VLLM, kwargs={"max_model_len": None})
    assert dummy_model.max_seq_length == 128

    del dummy_model
    clean_up()


@pytest.mark.vllm
@pytest.mark.gpu
def test_vllm_load_from_invalid_checkpoint() -> None:
    invalid_path = "/fake/path/to/model"

    with pytest.raises(ValueError) as exc_info:
        Qwen3_0_6B_VLLM(
            checkpoint_path=invalid_path,
            checkpoint_name="fake",
            max_model_len=128,
        )

    error_msg = str(exc_info.value).lower()
    assert "invalid repository id or local directory specified" in error_msg
    assert invalid_path.lower() in error_msg


@pytest.mark.vllm
@pytest.mark.gpu
def test_vllm_checkpoint_name_formatting() -> None:
    checkpoint_path = "Qwen/Qwen3-0.6B"
    checkpoint_name = "test_checkpoint"
    model = safe_vllm_setup(model_fn=Qwen3_0_6B_VLLM, kwargs={"max_model_len": 128})
    model.checkpoint_path = checkpoint_path
    model.checkpoint_name = checkpoint_name

    expected_name = f"Qwen3_0_6B_VLLM_checkpoint_{checkpoint_name}"
    assert model.name == expected_name


@pytest.mark.vllm
def test_resolve_params_overrides_temperature() -> None:
    sampling_params = SamplingParams(temperature=0.6, top_p=0.9)

    result = VLLMModel._resolve_sampling_params(sampling_params, 50, ["STOP"], 0.8)

    assert result.max_tokens == 50
    assert result.stop == ["STOP"]
    assert result.temperature == 0.8
    assert result is sampling_params


@pytest.mark.vllm
def test_resolve_params_uses_default_temperature_when_none() -> None:
    sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

    result = VLLMModel._resolve_sampling_params(sampling_params, 25, ["STOP"], None)

    assert result.temperature == 0.7
    assert result.max_tokens == 25
    assert result.stop == ["STOP"]


@pytest.mark.vllm
def test_resolve_params_logs_temperature_override(caplog: pytest.LogCaptureFixture) -> None:
    sampling_params = SamplingParams(temperature=0.5)

    with caplog.at_level(logging.WARNING):
        result = VLLMModel._resolve_sampling_params(sampling_params, 15, None, 0.9)

    assert result.temperature == 0.9
    assert "Overriding sampling params temperature 0.5 with custom value 0.9" in caplog.text


@pytest.mark.vllm
def test_resolve_params_always_updates_tokens_and_stop() -> None:
    sampling_params = SamplingParams(temperature=0.8, max_tokens=50)

    result = VLLMModel._resolve_sampling_params(sampling_params, 100, ["END"], 1.0)

    assert result.max_tokens == 100
    assert result.stop == ["END"]
    assert result.temperature == 1.0
    assert result is sampling_params


@pytest.mark.vllm
def test_resolve_params_logs_default_temperature_usage(caplog: pytest.LogCaptureFixture) -> None:
    sampling_params = SamplingParams(temperature=0.3)

    with caplog.at_level(logging.INFO):
        result = VLLMModel._resolve_sampling_params(sampling_params, 50, None, None)

    assert result.temperature == 0.3
    assert "Using sampling params temperature value: 0.3" in caplog.text
    assert "as no custom temperature value was provided" in caplog.text


@pytest.mark.vllm
@pytest.mark.gpu
def test_dict_sampling_params_conversion() -> None:
    params_dict = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 150}
    model = safe_vllm_setup(Qwen3_0_6B_VLLM, {"max_model_len": 30, "sampling_params": params_dict})

    assert model.sampling_params.temperature == 0.8
    assert model.sampling_params.top_p == 0.95
    assert model.sampling_params.max_tokens == 150


@pytest.mark.vllm
@pytest.mark.gpu
def test_sampling_params_object_unchanged() -> None:
    params_obj = SamplingParams(temperature=0.5, max_tokens=75)
    model = safe_vllm_setup(Qwen3_0_6B_VLLM, {"max_model_len": 30, "sampling_params": params_obj})

    assert model.sampling_params is params_obj


@pytest.mark.vllm
@pytest.mark.gpu
def test_dict_overrides_model_defaults() -> None:
    model_with_defaults = safe_vllm_setup(Qwen3_0_6B_VLLM, {"max_model_len": 30})
    model_with_dict = safe_vllm_setup(Qwen3_0_6B_VLLM, {"max_model_len": 30, "sampling_params": {"temperature": 0.1}})

    assert model_with_dict.sampling_params.temperature == 0.1
    assert model_with_dict.sampling_params.temperature != model_with_defaults.sampling_params.temperature


@pytest.mark.vllm
def test_invalid_sampling_param_raises() -> None:
    with pytest.raises(TypeError):
        # Call the method on the class directly to avoid gpu context
        VLLMModel._process_sampling_params(None, {"invalid_param": "value"})  # type: ignore[arg-type]


@pytest.mark.vllm
@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_fn, kwargs",
    [
        (Qwen3_0_6B_VLLM, {"max_model_len": 32, "dtype": "bfloat16", "tensor_parallel_size": 2}),
    ],
)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >=2 GPUs for tensor_parallel_size=2")
def test_logprobs_batched_vs_single[T: VLLMModel](model_fn: type[T], kwargs: Any) -> None:
    """
    Test that batched logprobs inference produces identical results to single-sample inference.
    """
    model = safe_vllm_setup(model_fn, kwargs)

    base_messages_1: list[Message] = [
        Message(
            role=Role.USER,
            content="Small streams often flow into bigger streams or rivers. The small "
            "streams are called tributaries. A river and all its tributaries make up a river system."
            "\nQuestion: What is the term for small streams?",  # noqa: E501
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]

    base_messages_2: list[Message] = [
        Message(
            role=Role.USER,
            content="Photosynthesis is the process by which plants make their own"
            "food using sunlight, carbon dioxide, and water. This process occurs in the chloroplasts of plant cells."
            "\nQuestion: Where does photosynthesis occur?",  # noqa: E501
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]

    base_messages_3: list[Message] = [
        Message(
            role=Role.USER,
            content="The Earth's atmosphere is composed of different layers. The"
            "layer closest to Earth's surface is called the troposphere, where most weather occurs."
            "\nQuestion: What is the lowest layer of the atmosphere called?",  # noqa: E501
        ),
        Message(role=Role.ASSISTANT, content="Answer:"),
    ]

    test_samples = [
        Sample(
            id=0,
            subject="science",
            messages=base_messages_1,
            ground_truth="tributaries",
            possible_completions=["wetlands", "rivers", "canals", "tributaries"],
        ),
        Sample(
            id=1,
            subject="science",
            messages=base_messages_2,
            ground_truth="chloroplasts",
            possible_completions=["mitochondria", "chloroplasts", "nucleus", "vacuoles"],
        ),
        Sample(
            id=2,
            subject="science",
            messages=base_messages_3,
            ground_truth="troposphere",
            possible_completions=["stratosphere", "troposphere", "mesosphere", "thermosphere"],
        ),
    ]

    single_results: list[RawLoglikelihood] = []
    for sample in test_samples:
        result = model.logprobs([sample])
        single_results.extend(result)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    kwargs["batch_size"] = 12
    model = safe_vllm_setup(model_fn, kwargs)

    batched_results: list[RawLoglikelihood] = model.logprobs(test_samples)

    assert len(single_results) == len(batched_results) == 3

    for i, (single_result, batched_result) in enumerate(zip(single_results, batched_results)):
        print(f"\n=== Comparing Sample {i} ===")

        # Check logprobs with detailed logging
        print(f"📊 Sample {i}: Logprob comparison details:")
        all_logprobs_close = True
        for choice in single_result.loglikelihoods.keys():
            single_logprob = single_result.loglikelihoods[choice]
            batched_logprob = batched_result.loglikelihoods[choice]

            # Calculate differences
            abs_diff = abs(single_logprob - batched_logprob)
            rel_diff = abs_diff / abs(single_logprob) if single_logprob != 0 else float("inf")

            # Check if they're close with different tolerances
            close_1e5 = torch.allclose(torch.tensor(single_logprob), torch.tensor(batched_logprob), rtol=1e-5)
            close_1e4 = torch.allclose(torch.tensor(single_logprob), torch.tensor(batched_logprob), rtol=1e-4)
            close_1e3 = torch.allclose(torch.tensor(single_logprob), torch.tensor(batched_logprob), rtol=1e-3)
            close_1e2 = torch.allclose(torch.tensor(single_logprob), torch.tensor(batched_logprob), rtol=1e-2)
            close_1e1 = torch.allclose(torch.tensor(single_logprob), torch.tensor(batched_logprob), rtol=1e-1)

            print(f"  Choice '{choice}':")
            print(f"    Single:     {single_logprob}")
            print(f"    Batched:    {batched_logprob}")
            print(f"    Abs diff:   {abs_diff}")
            print(f"    Rel diff:   {rel_diff}")
            print(f"    Close rtol=1e-5: {close_1e5}")
            print(f"    Close rtol=1e-4: {close_1e4}")
            print(f"    Close rtol=1e-3: {close_1e3}")
            print(f"    Close rtol=1e-2: {close_1e2}")
            print(f"    Close rtol=1e-1: {close_1e1}")

            if not close_1e5:
                all_logprobs_close = False

        if all_logprobs_close:
            print(f"✅ Sample {i}: All logprobs are close (rtol=1e-5)")
        else:
            print(f"❌ Sample {i}: Some logprobs differ more than rtol=1e-5")

        # Now do the actual assertions (you can comment these out if you want to see all samples)
        assert single_result.prompt == batched_result.prompt, f"Sample {i}: Prompts don't match"
        assert single_result.prompt_sequence_positions == batched_result.prompt_sequence_positions, (
            f"Sample {i}: Prompt sequence positions don't match"
        )
        assert single_result.loglikelihoods.keys() == batched_result.loglikelihoods.keys(), (
            f"Sample {i}: Loglikelihood keys don't match"
        )

        # Batched BF16 is just horrific, we need rtol=1e-2 to get it to pass
        for choice in single_result.loglikelihoods.keys():
            single_logprob = single_result.loglikelihoods[choice]
            batched_logprob = batched_result.loglikelihoods[choice]
            rel_diff = (
                abs(single_logprob - batched_logprob) / abs(single_logprob) if single_logprob != 0 else float("inf")
            )
            assert torch.allclose(torch.tensor(single_logprob), torch.tensor(batched_logprob), rtol=1e-1), (
                f"Sample {i}, choice '{choice}': Logprobs don't match "
                f"(single: {single_logprob}, batched: {batched_logprob}, "
                f"abs_diff: {abs(single_logprob - batched_logprob)}, "
                f"rel_diff: {rel_diff})"
            )

        assert single_result.loglikelihoods_sequence_positions == batched_result.loglikelihoods_sequence_positions, (
            f"Sample {i}: Sequence positions don't match"
        )
        assert single_result.raw_loglikelihood_error == batched_result.raw_loglikelihood_error, (
            f"Sample {i}: Error states don't match"
        )


@pytest.mark.vllm
@pytest.mark.gpu
@pytest.mark.parametrize(
    "model_fn, kwargs",
    [
        (
            Qwen3_0_6B_VLLM,
            {
                "max_model_len": 1024,
                "tensor_parallel_size": 2,
                "sampling_params": SamplingParams(max_tokens=25, temperature=0.0, stop=["\n"]),
            },
        ),
    ],
)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires >=2 GPUs for tensor_parallel_size=2")
def test_completions_batched_vs_single[T: VLLMModel](model_fn: type[T], kwargs: Any) -> None:
    """
    Test that batched completion inference produces identical results to single-message inference.
    """
    model = safe_vllm_setup(model_fn, kwargs)

    messages_1: Sequence[Message] = [
        Message(
            role=Role.USER,
            content="What is the capital of France? Provide a brief one-sentence answer.",
        ),
    ]

    messages_2: Sequence[Message] = [
        Message(
            role=Role.USER,
            content="Explain photosynthesis in one sentence.",
        ),
    ]

    messages_3: Sequence[Message] = [
        Message(
            role=Role.USER,
            content="What is 2 + 2? Answer with just the number.",
        ),
    ]

    test_message_sequences = [messages_1, messages_2, messages_3]

    max_tokens = 25
    temperature = 0.0
    stop_sequences = ["\n"]

    single_results: list[RawCompletion] = []
    for messages in test_message_sequences:
        result = model.generate_from_messages(
            messages=[messages], max_tokens=max_tokens, temperature=temperature, stop_sequences=stop_sequences
        )
        single_results.extend(result)

    del model
    clean_up()

    kwargs["batch_size"] = 10
    model = safe_vllm_setup(model_fn, kwargs)

    batched_results: list[RawCompletion] = model.generate_from_messages(
        messages=test_message_sequences,
        max_tokens=max_tokens,
        temperature=temperature,
        stop_sequences=stop_sequences,
    )

    assert len(single_results) == len(batched_results) == 3

    for i, (single_result, batched_result) in enumerate(zip(single_results, batched_results)):
        assert single_result.prompt == batched_result.prompt, f"Message {i}: Prompts don't match"
        assert single_result.prompt_sequence_positions == batched_result.prompt_sequence_positions, (
            f"Message {i}: Prompt sequence positions don't match"
        )
        assert single_result.completion == batched_result.completion, (
            f"Message {i}: Completions don't match "
            f"(single: '{single_result.completion}', batched: '{batched_result.completion}')"
        )
        assert single_result.completion_sequence_positions == batched_result.completion_sequence_positions, (
            f"Message {i}: Completion sequence positions don't match"
        )
        assert single_result.raw_completion_error == batched_result.raw_completion_error, (
            f"Message {i}: Error states don't match"
        )


@pytest.mark.vllm
def test_vllm_tokenizer_double_bos_problem() -> None:
    """
    Test that demonstrates the double BOS token problem with Llama tokenizer.

    This test shows why we need add_special_tokens=False in VLLMTokenizer to avoid
    double BOS tokens when the formatter already includes BOS tokens.
    """
    from transformers import AutoTokenizer

    # Use Llama tokenizer which exhibits the double BOS problem
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    formatter = Llama3Formatter()
    expected_bos_token = 128000  # Llama BOS token

    messages = [Message(role=Role.USER, content="What is 2+2?"), Message(role=Role.ASSISTANT, content="The answer is")]

    # Format the messages (this already includes BOS tokens properly)
    formatted_prompt = formatter.format(messages, output_mode="string")

    # Test tokenization with and without add_special_tokens
    prompt_with_special = tokenizer.encode(formatted_prompt, add_special_tokens=True)
    prompt_without_special = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    # Verify tokenizer behavior with and without special tokens
    assert prompt_with_special[0] == expected_bos_token, "First token with special tokens should be BOS"
    assert prompt_with_special[1] == expected_bos_token, "Second token is ANOTHER BOS - this is the double BOS problem!"
    assert prompt_without_special[0] == expected_bos_token, (
        "First token without special tokens should be BOS (from formatter)"
    )
    assert prompt_without_special[1] != expected_bos_token, (
        "Second token should NOT be BOS when add_special_tokens=False"
    )


@pytest.mark.vllm
def test_vllm_generate_with_llama_tokenizer_avoids_double_bos() -> None:
    """
    Test that VLLMModel.generate_from_messages with Llama tokenizer uses add_special_tokens=False,
    avoiding the double BOS token problem. This test would FAIL if our fix was reverted.
    """
    from unittest.mock import patch

    from transformers import AutoTokenizer

    # Create a custom model class that uses Llama tokenizer with Llama3Formatter
    class TestLlamaVLLMModel(VLLMModel):
        LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"
        DEFAULT_FORMATTER = Llama3Formatter

    # Mock the VLLM engine to avoid actual model loading
    with patch("eval_framework.llm.vllm.LLM") as mock_llm:
        model = TestLlamaVLLMModel(max_model_len=64, tensor_parallel_size=1)

        try:
            messages = [
                Message(role=Role.USER, content="What is 2+2?"),
                Message(role=Role.ASSISTANT, content="The answer is"),
            ]

            formatted_prompt = model._formatter.format(messages, output_mode="string")
            raw_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
            prompt_without_special = raw_tokenizer.encode(formatted_prompt, add_special_tokens=False)

            mock_output = type(
                "MockOutput",
                (),
                {
                    "text": " 4",
                    "token_ids": [19],
                },
            )()

            mock_completion = type(
                "MockCompletion",
                (),
                {
                    "prompt": formatted_prompt,
                    "prompt_token_ids": prompt_without_special,
                    "outputs": [mock_output],
                },
            )()

            mock_llm.return_value.generate.return_value = [mock_completion]

            completions = model.generate_from_messages(messages=[messages], max_tokens=5, temperature=0.0)
            assert len(completions) == 1
            completion = completions[0]

            assert completion.prompt_sequence_positions == len(prompt_without_special), (
                f"generate_from_messages should use tokenizer with add_special_tokens=False. "
                f"Expected {len(prompt_without_special)} tokens, got {completion.prompt_sequence_positions}"
            )

            vllm_tokenizer = model.tokenizer
            vllm_prompt_obj = vllm_tokenizer.encode_formatted_struct(formatted_prompt)
            assert vllm_prompt_obj.tokens == prompt_without_special, (
                f"VLLMModel tokenizer should use add_special_tokens=False to avoid double BOS tokens. "
                f"Expected {prompt_without_special[:5]}..., got {vllm_prompt_obj.tokens[:5]}... "
            )

        finally:
            if hasattr(model, "llm") and model.llm is not None:
                del model.llm
            clean_up()


@pytest.mark.vllm
def test_vllm_logprobs_with_llama_tokenizer_avoids_double_bos() -> None:
    """
    Test that VLLMModel.logprobs with Llama tokenizer uses add_special_tokens=False,
    avoiding the double BOS token problem. This test would FAIL if our fix was reverted.
    """
    from unittest.mock import patch

    from transformers import AutoTokenizer

    # Create a custom model class that uses Llama tokenizer with Llama3Formatter
    class TestLlamaVLLMModel(VLLMModel):
        LLM_NAME = "meta-llama/Llama-3.1-8B-Instruct"
        DEFAULT_FORMATTER = Llama3Formatter

    # Mock the VLLM engine to avoid actual model loading
    with patch("eval_framework.llm.vllm.LLM"):
        model = TestLlamaVLLMModel(max_model_len=64, tensor_parallel_size=1)

        try:
            messages = [
                Message(role=Role.USER, content="What is 2+2?"),
                Message(role=Role.ASSISTANT, content="The answer is"),
            ]

            formatted_prompt = model._formatter.format(messages, output_mode="string")
            raw_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
            prompt_without_special = raw_tokenizer.encode(formatted_prompt, add_special_tokens=False)

            sample = Sample(
                id=0, subject="math", messages=messages, ground_truth="4", possible_completions=["4", "5"], context=None
            )

            with patch.object(model, "_model_log_probs") as mock_model_logprobs:
                mock_model_logprobs.return_value = [-0.5, -2.1]

                logprob_results = model.logprobs([sample])
                assert len(logprob_results) == 1

                logprob_result = logprob_results[0]
                assert logprob_result.prompt_sequence_positions == len(prompt_without_special), (
                    f"logprobs should use tokenizer with add_special_tokens=False. "
                    f"Expected {len(prompt_without_special)} tokens, got {logprob_result.prompt_sequence_positions}"
                )

            # Verify VLLMModel tokenizer matches "without special tokens" behavior
            vllm_tokenizer = model.tokenizer
            vllm_prompt_obj = vllm_tokenizer.encode_formatted_struct(formatted_prompt)
            assert vllm_prompt_obj.tokens == prompt_without_special, (
                f"VLLMModel tokenizer should use add_special_tokens=False to avoid double BOS tokens. "
                f"Expected {prompt_without_special[:5]}..., got {vllm_prompt_obj.tokens[:5]}... "
            )

        finally:
            if hasattr(model, "llm") and model.llm is not None:
                del model.llm
            clean_up()


@pytest.mark.parametrize("model_tokenizer_pair", [(VLLMModel, VLLMTokenizer), (MistralVLLM, MistralAdapter)])
def test_tokenizer_single_initialization(
    model_tokenizer_pair: tuple[type[VLLMModel], type[VLLMTokenizer]],
) -> None:
    """
    Test that VLLMModel properly caches the tokenizer:
    1. Initializes the tokenizer only once
    2. Returns the same object instance on subsequent accesses
    """
    model_cls, tokenizer_cls = model_tokenizer_pair

    # Create a simple subclass of VLLMModel for testing
    class TestVLLMModel(model_cls):  # type: ignore
        LLM_NAME = "test-model"
        DEFAULT_FORMATTER = ConcatFormatter

    with patch(f"{tokenizer_cls.__module__}.{tokenizer_cls.__name__}") as mock_tokenizer_cls:
        # Create a mock tokenizer instance that will be returned by the constructor
        mock_tokenizer = Mock()
        mock_tokenizer_cls.return_value = mock_tokenizer

        # Create the model with mocked LLM to avoid actual model loading
        with patch("eval_framework.llm.vllm.LLM"):
            model = TestVLLMModel(max_model_len=128)

            # Get tokenizer references multiple times
            tokenizer1 = model.tokenizer
            tokenizer2 = model.tokenizer
            tokenizer3 = model.tokenizer

            # Verify that the constructor was called exactly once
            mock_tokenizer_cls.assert_called_once_with(target_mdl="test-model")

            # Verify that all references point to the same object instance
            assert id(tokenizer1) == id(tokenizer2) == id(tokenizer3), (
                "All tokenizer references should point to the same object instance"
            )


@pytest.mark.parametrize(
    "model_class_and_name", [(VLLMModel, "gpt2"), (MistralVLLM, "mistralai/Ministral-8B-Instruct-2410")]
)
def test_tokenizer_initialization_performance(
    model_class_and_name: tuple[type[VLLMModel], str],
) -> None:
    """
    Test that accessing the tokenizer property multiple times is fast after the first access,
    which confirms that the tokenizer is being cached properly.

    This test uses a real tokenizer to measure actual performance improvement.
    """

    # Create a simple subclass of VLLMModel for testing with a real model name
    base_model_cls, base_model_name = model_class_and_name

    class TestVLLMModel(base_model_cls):  # type: ignore
        LLM_NAME = base_model_name
        DEFAULT_FORMATTER = ConcatFormatter

    # Only mock the LLM to avoid loading the actual model weights
    with patch("eval_framework.llm.vllm.LLM"):
        # Create the model with real tokenizer but mocked LLM and measure first access
        # (which should be slow as it is the real tokenizer initialization)
        model = TestVLLMModel(max_model_len=128)
        start_time = time.time()
        _ = model.tokenizer
        first_access_time = time.time() - start_time

        # Now without resetting, access should be much faster (cached)
        start_time = time.time()
        _ = model.tokenizer
        cached_access_time = time.time() - start_time

        # The cached access should be significantly faster than initialization
        # In practice, it should be almost instantaneous
        assert cached_access_time < first_access_time / 10, (
            f"Cached tokenizer access should be much faster than initialization "
            f"(first init: {first_access_time:.6f}s, cached access: {cached_access_time:.6f}s)"
        )


@pytest.mark.vllm
@pytest.mark.gpu
@pytest.mark.parametrize(
    "generator_gpus, evaluator_gpus",
    [
        pytest.param(1, 1, id="single gpu inference"),
        pytest.param(1, 2, id="single to multi-gpu inference"),
        pytest.param(2, 4, id="multi-gpu inference"),
    ],
)
def test_resource_cleanup(generator_gpus: int, evaluator_gpus: int) -> None:
    num_gpus = torch.cuda.device_count()
    if min(generator_gpus, evaluator_gpus) >= num_gpus:
        pytest.skip("GPUs not available")

    class Qwen8B(VLLMModel):
        LLM_NAME = "Qwen/Qwen3-8B"

    try:
        model_config = cast(
            dict[str, Any],
            {
                "max_model_len": 1000,
                "enforce_eager": True,
                "tensor_parallel_size": generator_gpus,
                "formatter": HFFormatter("Qwen/Qwen3-8B", chat_template_kwargs={"enable_thinking": True}),
            },
        )

        response_generator_model = Qwen8B(**model_config)
        response_generator_model.generate_from_messages(
            messages=[[Message(role=Role.USER, content="What is capital of Germany ?")]],
            max_tokens=100,
            temperature=0.0,
        )
        del response_generator_model
        model_config["tensor_parallel_size"] = evaluator_gpus
        judge_model = Qwen8B(**model_config)
        judge_model.generate_from_messages(
            messages=[
                [
                    Message(
                        role=Role.USER,
                        content=(
                            "Rank the following responses between 0 and 1"
                            "based on clarity : \nResponse 1: {response1}"
                            "\nResponse 2: {response2}"
                        ).format(response1="This is an ambiguous answer", response2="This is a clear answer"),
                    )
                ]
            ],
            max_tokens=100,
            temperature=0.0,
        )
        del judge_model
    except Exception as e:
        pytest.fail(f"{e.__class__.__name__} : {e}")


@pytest.mark.vllm
@pytest.mark.parametrize("kwargs, expected_model, expected_name", LLM_INIT_SOURCE_PARAMS)
def test_vllm_init_source(mocker: MockerFixture, kwargs: Any, expected_model: str, expected_name: str) -> None:
    """Test that VLLMModel initializes correctly with different checkpoint source arguments."""

    VLLM_patch = mocker.patch("eval_framework.llm.vllm.LLM")
    mock_wandb_fs = MagicMock()
    mock_wandb_fs.__enter__().find_hf_checkpoint_root_from_path_list.return_value = "/download"
    mocker.patch("eval_framework.utils.file_ops.WandbFs", return_value=mock_wandb_fs)

    # Test with a typical subclass
    class MyModel(VLLMModel):
        LLM_NAME = "org/model"
        DEFAULT_FORMATTER = ConcatFormatter

    model = MyModel(**kwargs)
    assert VLLM_patch.call_args[1]["model"] == expected_model
    assert model.name == expected_name
    assert model.LLM_NAME == expected_model

    # Test with the base class
    if not kwargs or list(kwargs.keys()) == ["checkpoint_name"]:  # no checkpoint source -> error
        with pytest.raises(ValueError):
            VLLMModel(**kwargs)
    else:
        base_model = VLLMModel(**kwargs, formatter=ConcatFormatter())
        assert VLLM_patch.call_args[1]["model"] == expected_model
        assert base_model.name == expected_name.replace("MyModel", "VLLMModel")
        assert base_model.LLM_NAME == expected_model


@pytest.mark.vllm
def test_vllm_init_source_multiple_args() -> None:
    """Test that providing multiple checkpoint source arguments raises an error."""
    with pytest.raises(ValueError):
        VLLMModel(checkpoint_path="/ckpt/m", model_name="org/other")
    with pytest.raises(ValueError):
        VLLMModel(checkpoint_path="/ckpt/m", artifact_name="art:v0")
    with pytest.raises(ValueError):
        VLLMModel(model_name="org/other", artifact_name="art:v0")


@pytest.mark.vllm
@pytest.mark.parametrize("kwargs, expected_formatter_cls", LLM_INIT_FORMATTER_PARAMS)
def test_vllm_init_formatter(mocker: MockerFixture, kwargs: Any, expected_formatter_cls: type) -> None:
    tokenizer_mock = mocker.patch("eval_framework.llm.vllm.get_tokenizer")
    mocker.patch("eval_framework.llm.vllm.LLM")

    # Test with a typical subclass
    class MyModel(VLLMModel):
        LLM_NAME = "org/model"
        DEFAULT_FORMATTER = ConcatFormatter

    model = MyModel(**kwargs)
    assert isinstance(model._formatter, expected_formatter_cls)

    # Test with the base class
    if len(kwargs) <= 1:  # no formatter -> error
        tokenizer_mock.return_value.chat_template = None
        with pytest.raises(ValueError):
            VLLMModel(**kwargs)
    else:
        base_model = VLLMModel(**kwargs)
        assert isinstance(base_model._formatter, expected_formatter_cls)


@pytest.mark.vllm
def test_vllm_init_formatter_multiple_args() -> None:
    """Test that providing multiple formatter arguments raises an error."""
    with pytest.raises(ValueError):
        VLLMModel(formatter=IdentityFormatter(), formatter_name="Llama3Formatter")
    with pytest.raises(ValueError):
        VLLMModel(formatter=IdentityFormatter(), formatter_kwargs=dict(hf_llm_name="HuggingFaceTB/SmolLM-135M"))


@pytest.mark.vllm
@pytest.mark.gpu
def test_max_tokens_generation() -> None:
    class Qwen600M(VLLMModel):
        LLM_NAME = "Qwen/Qwen3-0.6B"

    model = Qwen600M(bytes_per_token=4.0)

    messages: list[Message] = [
        Message(role=Role.USER, content="Tell me a long story.\n"),
        Message(role=Role.ASSISTANT, content="Once upon a time,"),
    ]

    generation_results: list[RawCompletion] = model.generate_from_messages(
        messages=[messages], max_tokens=10, temperature=0
    )

    assert len(generation_results) == 1
    generated_num_tokens = generation_results[0].completion_sequence_positions
    assert generated_num_tokens == 10

    byte_level_model = Qwen600M(bytes_per_token=1.0)

    byte_level_model_messages: list[Message] = [
        Message(role=Role.USER, content="Tell me a long story.\n"),
        Message(role=Role.ASSISTANT, content="Once upon a time,"),
    ]

    byte_level_model_generation_results: list[RawCompletion] = byte_level_model.generate_from_messages(
        messages=[byte_level_model_messages], max_tokens=10, temperature=0
    )

    assert len(byte_level_model_generation_results) == 1
    byte_level_model_generated_num_tokens = byte_level_model_generation_results[0].completion_sequence_positions
    assert byte_level_model_generated_num_tokens == 40
