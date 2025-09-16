"""This is just a default model file with some small models for testing.

Please define your own model file externally and pass it to the eval-framework entrypoint
to use it.
"""

from eval_framework.utils.packaging import is_extra_installed

if is_extra_installed(extra="transformers"):
    pass  # noqa F401

if is_extra_installed("mistral"):
    from eval_framework.llm.mistral import MagistralVLLM  # noqa F401

if is_extra_installed("vllm"):
    from eval_framework.llm.vllm import VLLMRegistryModel, Qwen3_0_6B_VLLM, Qwen3_0_6B_VLLM_No_Thinking  # noqa F401
