import os
import tempfile
import traceback
from typing import Any, NamedTuple

from llm_sandbox import SandboxSession
from llm_sandbox.const import SupportedLanguage
from pydantic import Field

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, Error, extract_context_metric

# Languages with native llm-sandbox support.
_SANDBOX_LANG_MAP: dict[str, str] = {
    "cpp": SupportedLanguage.CPP,
    "java": SupportedLanguage.JAVA,
    "js": SupportedLanguage.JAVASCRIPT,
}


class _CustomLangConfig(NamedTuple):
    image: str
    file_ext: str
    commands: list[str]


# Languages not in llm-sandbox's dispatch table. We open a session with a dummy lang
# (SupportedLanguage.PYTHON) and a language-specific Docker image, then drive
# copy_to_runtime + execute_command manually instead of calling session.run().
_CUSTOM_LANG_MAP: dict[str, _CustomLangConfig] = {
    "php": _CustomLangConfig(
        image="php:8.3-cli",  # https://hub.docker.com/_/php
        file_ext="php",
        commands=["php {code_file}"],
    ),
    "rs": _CustomLangConfig(
        image="rust:1.82-slim",  # https://hub.docker.com/_/rust
        file_ext="rs",
        # rustc writes the binary next to the source; then we run it.
        commands=["rustc {code_file} -o /tmp/a.out", "/tmp/a.out"],
    ),
    "sh": _CustomLangConfig(
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
            success, output = self._execute(context, response.completion)
        except Exception as exc:
            error = Error(
                error_class=exc.__class__.__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
            return [MetricResult(metric_name=self.NAME, value=None, higher_is_better=True, error=error)]

        return [
            MetricResult(
                metric_name=self.NAME,
                value=1.0 if success else 0.0,
                higher_is_better=True,
                code_execution_trace=output,
            )
        ]

    def _execute(self, context: MultiPLEMetricContext, completion: str) -> tuple[bool, str]:
        """Combine prompt + completion + tests and route to the appropriate sandbox."""
        full_code = context.prompt + completion + "\n" + context.tests
        language = context.language

        if language in _SANDBOX_LANG_MAP:
            return self._execute_via_sandbox_run(full_code, _SANDBOX_LANG_MAP[language])
        elif language in _CUSTOM_LANG_MAP:
            return self._execute_via_custom_image(full_code, _CUSTOM_LANG_MAP[language])
        else:
            raise ValueError(
                f"Unknown MultiPL-E language '{language}'. "
                f"Supported: {sorted(_SANDBOX_LANG_MAP) + sorted(_CUSTOM_LANG_MAP)}"
            )

    def _execute_via_sandbox_run(self, full_code: str, sandbox_lang: str) -> tuple[bool, str]:
        """Use llm-sandbox's native session.run() for cpp, java, js."""
        with SandboxSession(lang=sandbox_lang, keep_template=True, commit_container=False) as session:
            result: Any = session.run(full_code)
        output: str = getattr(result, "text", "") or ""
        exit_code: int = getattr(result, "exit_code", -1)
        return exit_code == 0, output

    def _execute_via_custom_image(self, full_code: str, cfg: _CustomLangConfig) -> tuple[bool, str]:
        """For php/rs/sh: copy code into a language-specific Docker image and run manually.

        We open a SandboxSession with a dummy lang (SupportedLanguage.PYTHON) so that
        llm-sandbox sets up the container plumbing, but we never call session.run().
        Instead we write the code to a local temp file, copy it into the container with
        copy_to_runtime, and drive each compile/run command with execute_command.
        """
        code_file = f"/tmp/code.{cfg.file_ext}"
        output = ""
        exit_code = -1

        with tempfile.NamedTemporaryFile(suffix=f".{cfg.file_ext}", mode="w", delete=False) as tmp:
            tmp.write(full_code)
            tmp_path = tmp.name

        try:
            with SandboxSession(
                lang=SupportedLanguage.PYTHON,  # dummy; we don't call run()
                image=cfg.image,
                keep_template=True,
                commit_container=False,
            ) as session:
                session.copy_to_runtime(tmp_path, code_file)
                for cmd_template in cfg.commands:
                    cmd = cmd_template.format(code_file=code_file)
                    result: Any = session.execute_command(cmd)
                    output = getattr(result, "text", "") or ""
                    exit_code = getattr(result, "exit_code", -1)
                    if exit_code != 0:
                        break
        finally:
            os.unlink(tmp_path)

        return exit_code == 0, output
