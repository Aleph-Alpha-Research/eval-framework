from eval_framework.utils import is_extra_installed

if is_extra_installed(extra="transformers"):
    from eval_framework.llm.huggingface_llm import *  # noqa F401

if is_extra_installed("mistral"):
    from eval_framework.llm.mistral import *  # noqa F401

if is_extra_installed("vllm"):
    from eval_framework.llm.vllm_models import *  # noqa F401

if is_extra_installed("api"):
    from eval_framework.llm.aleph_alpha_api_llm import *  # noqa F401
