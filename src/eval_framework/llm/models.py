from eval_framework.utils import is_extra_installed

if is_extra_installed(extra="transformers"):
    from eval_framework.llm.huggingface import *  # noqa F401

if is_extra_installed("mistral"):
    from eval_framework.llm.mistral import *  # noqa F401

if is_extra_installed("vllm"):
    from eval_framework.llm.vllm import *  # noqa F401

if is_extra_installed("api"):
    from eval_framework.llm.aleph_alpha import *  # noqa F401
