import functools
import os
import tempfile
import traceback
import urllib.request
import uuid
from typing import Any, NamedTuple

from llm_sandbox import SandboxSession
from llm_sandbox.const import DefaultImage, SupportedLanguage
from pydantic import Field

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, Error, extract_context_metric
from eval_framework.tasks.utils import get_or_create_pool

# Languages with native llm-sandbox support.
_SANDBOX_LANG_MAP: dict[str, str] = {
    "cpp": SupportedLanguage.CPP,
    "js": SupportedLanguage.JAVASCRIPT,
}


class _CustomLangConfig(NamedTuple):
    image: str
    file_ext: str
    commands: list[str]


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
_CUSTOM_LANG_MAP: dict[str, _CustomLangConfig] = {
    # Java uses explicit javac + java -ea (matching the official MultiPL-E evaluator approach):
    #   - javatuples is copied into the container once from the host (downloaded once per process)
    #     rather than wget-ing it on every container run
    #   - javac and java are combined into one sh -c call so there is only a single round-trip
    #   - -ea enables assertions so assert(wrong_answer) actually fails instead of silently passing
    "java": _CustomLangConfig(
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
    "php": _CustomLangConfig(
        image="php:8.3-cli",  # https://hub.docker.com/_/php
        file_ext="php",
        commands=["php {code_file}"],
    ),
    "rs": _CustomLangConfig(
        image="rust:1.82-slim",  # https://hub.docker.com/_/rust
        file_ext="rs",
        # rustc writes the binary into the same UUID dir as the source; then we run it.
        commands=["rustc {code_file} -o {container_dir}/a.out", "{container_dir}/a.out"],
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
            success, output = self._execute(context, response.completion, timeout=60)
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

    @staticmethod
    def _execute(context: MultiPLEMetricContext, completion: str, timeout: int) -> tuple[bool, str]:
        """Combine prompt + completion + tests and route to the appropriate sandbox."""
        language = context.language
        full_code = context.prompt + completion + "\n" + context.tests

        if language in _SANDBOX_LANG_MAP:
            return MultiPLECodeAssertion._execute_via_sandbox_run(full_code, _SANDBOX_LANG_MAP[language], timeout)
        elif language in _CUSTOM_LANG_MAP:
            # For Java, supply the javatuples jar from the host (downloaded once per process)
            # so the container doesn't need to wget it on every invocation.
            extra_host_files = {"javatuples.jar": _get_javatuples_jar()} if language == "java" else {}
            return MultiPLECodeAssertion._execute_via_custom_image(
                full_code, _CUSTOM_LANG_MAP[language], timeout, extra_host_files=extra_host_files
            )
        else:
            raise ValueError(
                f"Unknown MultiPL-E language '{language}'. "
                f"Supported: {sorted(_SANDBOX_LANG_MAP) + sorted(_CUSTOM_LANG_MAP)}"
            )

    @staticmethod
    def _execute_via_sandbox_run(full_code: str, sandbox_lang: str, timeout: int) -> tuple[bool, str]:
        """Use llm-sandbox's native session.run() for cpp, java, js."""
        image = getattr(DefaultImage, sandbox_lang.upper())
        pool = get_or_create_pool(image=image, lang=sandbox_lang)
        with SandboxSession(pool=pool, lang=sandbox_lang) as session:
            result: Any = session.run(full_code, timeout=timeout)
        return result.success(), result.stdout + result.stderr

    @staticmethod
    def _execute_via_custom_image(
        full_code: str,
        cfg: _CustomLangConfig,
        timeout: int,
        extra_host_files: dict[str, str] | None = None,
    ) -> tuple[bool, str]:
        """Copy code (and optional extra files) into a language-specific Docker image and run.

        We open a SandboxSession with a dummy lang (SupportedLanguage.PYTHON) so that
        llm-sandbox sets up the container plumbing, but we never call session.run().
        Instead we write the code to a local temp file, copy it into the container with
        copy_to_runtime, and drive each compile/run command with execute_command.

        Each invocation gets a unique UUID-named directory inside the container so that
        concurrent calls never clobber each other's source files or build artefacts.

        extra_host_files maps container filename → host path for any additional files that
        should be copied into container_dir before the commands run (e.g. dependency jars).
        """
        run_id = uuid.uuid4().hex
        # copy_to_runtime uses arcname=os.path.basename(src), so the local filename
        # must match the desired in-container filename for put_archive to place it correctly.
        code_filename = f"code.{cfg.file_ext}"
        container_dir = f"/tmp/{run_id}"
        code_file = f"{container_dir}/{code_filename}"
        output = ""

        pool = get_or_create_pool(image=cfg.image, lang=SupportedLanguage.PYTHON, min_pool_size=1, max_pool_size=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, code_filename)
            with open(tmp_path, "w") as f:
                f.write(full_code)

            with SandboxSession(pool=pool, lang=SupportedLanguage.PYTHON) as session:
                session.execute_command(f"mkdir -p {container_dir}")
                session.copy_to_runtime(tmp_path, code_file)
                for filename, host_path in (extra_host_files or {}).items():
                    session.copy_to_runtime(host_path, f"{container_dir}/{filename}")
                if timeout > 0:  # hack-add timeout from coreutils to the command executed
                    session.orig_execute_command = session.execute_command
                    session.execute_command = lambda command: session.orig_execute_command(
                        f"timeout {timeout} {command}"
                    )
                for cmd_template in cfg.commands:
                    cmd = cmd_template.format(code_file=code_file, container_dir=container_dir)
                    result: Any = session.execute_command(cmd)
                    if not result.success():
                        break
                    output = result.stdout + result.stderr

        return result.success(), output
