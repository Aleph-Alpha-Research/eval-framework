"""This is just a default model file with API-backed models.

Please define your own model file externally and pass it to the eval-framework entrypoint
to use it.
"""

from eval_framework.llm.openai import OpenAIModel  # noqa F401
from eval_framework.utils.packaging import is_extra_installed

if is_extra_installed("api"):
    from eval_framework.llm.aleph_alpha import AlephAlphaAPIModel  # noqa F401
