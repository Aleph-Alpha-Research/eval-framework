import functools
import os
import tempfile
import traceback
import urllib.request

from llm_sandbox.const import SupportedLanguage
from llm_sandbox.exceptions import SandboxTimeoutError
from pydantic import Field

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.metrics.completion.sandbox import (
    CustomLangConfig,
    execute_in_custom_image,
    execute_in_default_image,
)
from eval_framework.shared.types import BaseMetricContext, Completion, Error, extract_context_metric

# Languages with native llm-sandbox support.
_SANDBOX_LANG_MAP: dict[str, str] = {
    "cpp": SupportedLanguage.CPP,
    "js": SupportedLanguage.JAVASCRIPT,
}

# Memory cap applied to every sandbox container
_RUNTIME_CONFIGS: dict[str, str] = {"mem_limit": "1g"}


_JAVATUPLES_URL = "https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar"


@functools.lru_cache(maxsize=1)
def _get_javatuples_jar() -> str:
    """Download javatuples-1.2.jar once per process and return its local path.

    lru_cache makes this thread-safe: concurrent callers block on the first download
    and all receive the same cached path on subsequent calls.
    """
    fd, path = tempfile.mkstemp(suffix="-javatuples-1.2.jar")
    os.close(fd)
    urllib.request.urlretrieve(_JAVATUPLES_URL, path)
    return path


# Languages not in llm-sandbox's dispatch table. We open a session with a dummy lang
# (SupportedLanguage.PYTHON) and a language-specific Docker image, then drive
# copy_to_runtime + execute_command manually instead of calling session.run().
_CUSTOM_LANG_MAP: dict[str, CustomLangConfig] = {
    # Java uses explicit javac + java -ea (matching the official MultiPL-E evaluator approach):
    #   - javatuples is copied into the container once from the host (downloaded once per process)
    #     rather than wget-ing it on every container run
    #   - javac and java are combined into one sh -c call so there is only a single round-trip
    #   - -ea enables assertions so assert(wrong_answer) actually fails instead of silently passing
    "java": CustomLangConfig(
        image="ghcr.io/vndee/sandbox-java-11-bullseye",
        file_ext="java",
        # javatuples.jar is copied into container_dir before these commands run.
        # sh -c lets us chain compile+run in one Docker exec round-trip; Docker's SDK
        # uses shlex.split on the string, so the single-quoted argument is parsed correctly.
        commands=[
            "sh -c 'javac -encoding UTF8 -cp {container_dir}/javatuples.jar"
            " -d {container_dir} {code_file}"
            " && java -ea -cp {container_dir}:{container_dir}/javatuples.jar Problem'",
        ],
    ),
    "php": CustomLangConfig(
        image="php:8.3-cli",  # https://hub.docker.com/_/php
        file_ext="php",
        commands=["php {code_file}"],
    ),
    "rs": CustomLangConfig(
        image="rust:1.82-slim",  # https://hub.docker.com/_/rust
        file_ext="rs",
        # rustc writes the binary into the same UUID dir as the source; then we run it.
        commands=["rustc {code_file} -o {container_dir}/a.out", "{container_dir}/a.out"],
    ),
    "sh": CustomLangConfig(
        image="bash:5.2",  # https://hub.docker.com/_/bash
        file_ext="sh",
        commands=["bash {code_file}"],
    ),
}


class MultiPLEMetricContext(BaseMetricContext):
    """Context for MultiPL-E code execution."""

    prompt: str = Field(description="Original function-signature prompt from the dataset")
    tests: str = Field(description="Language-native test harness code from the dataset")
    language: str = Field(description="Target programming language identifier (e.g. 'cpp', 'java', 'js')")


class MultiPLECodeAssertion(BaseMetric[Completion]):
    """Execute MultiPL-E completions in a language-specific sandbox and return pass/fail."""

    NAME = "MultiPL-E Code Assertion"

    def calculate(self, response: Completion) -> list[MetricResult]:
        if response.error is not None:
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=response.error)]

        context = extract_context_metric(response, MultiPLEMetricContext)

        try:
            success, output = self._execute(context, response.completion, timeout=60)
        except SandboxTimeoutError as exc:
            # The submitted code timed out (e.g. an infinite loop) -- a failing sample, not an infra
            # problem.
            error = Error(
                error_class=exc.__class__.__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=error)]
        except Exception as exc:
            # Any other sandbox/Docker error (e.g. an image pull rate limit) is an infra failure.
            return self._record_or_raise(exc)

        return [
            MetricResult(
                metric_name=self.NAME,
                value=1.0 if success else 0.0,
                higher_is_better=True,
                code_execution_trace=output,
            )
        ]

    @staticmethod
    def _execute(context: MultiPLEMetricContext, completion: str, timeout: int) -> tuple[bool, str]:
        """Combine prompt + completion + tests and route to the appropriate sandbox."""
        language = context.language
        full_code = context.prompt + completion + "\n" + context.tests

        if language in _SANDBOX_LANG_MAP:
            return execute_in_default_image(
                full_code, _SANDBOX_LANG_MAP[language], timeout, runtime_configs=_RUNTIME_CONFIGS
            )
        elif language in _CUSTOM_LANG_MAP:
            # For Java, supply the javatuples jar from the host (downloaded once per process)
            # so the container doesn't need to wget it on every invocation.
            extra_host_files = {"javatuples.jar": _get_javatuples_jar()} if language == "java" else {}
            return execute_in_custom_image(
                full_code,
                _CUSTOM_LANG_MAP[language],
                timeout,
                extra_host_files=extra_host_files,
                runtime_configs=_RUNTIME_CONFIGS,
            )
        else:
            raise ValueError(
                f"Unknown MultiPL-E language '{language}'. "
                f"Supported: {sorted(_SANDBOX_LANG_MAP) + sorted(_CUSTOM_LANG_MAP)}"
            )
