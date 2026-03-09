import atexit
import functools
import os
import tempfile
import threading
import traceback
import urllib.request
import uuid
from typing import Any, NamedTuple

from llm_sandbox import SandboxSession
from llm_sandbox.const import DefaultImage, SupportedLanguage
from llm_sandbox.pool import PoolConfig, create_pool_manager
from llm_sandbox.pool.base import ContainerPoolManager
from pydantic import Field

from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, Error, extract_context_metric

_pools: dict[tuple[str, str], ContainerPoolManager] = {}
_pools_lock = threading.Lock()


def _get_or_create_pool(image: str, lang: str) -> ContainerPoolManager:
    key = (image, lang)
    with _pools_lock:
        if key not in _pools:
            _pools[key] = create_pool_manager(
                config=PoolConfig(min_pool_size=2, max_pool_size=8),
                lang=lang,
                image=image,
                keep_template=True,
            )
        return _pools[key]


def _close_pools() -> None:
    for pool in _pools.values():
        try:
            pool.close()
        except Exception:
            pass


atexit.register(_close_pools)


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
    # Note: completions that use 'assert' as a variable/method name (invalid Java since 1.4) will
    # still produce a compile error — this is correct behaviour; they are invalid Java code.
    # Changing to -source 1.3 would NOT help: the test harness's assert(expr) statements would
    # then be parsed as method calls (no assert(boolean) method exists), breaking all tests.
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

        # MultiPL-E Java prompts end with an open method body; the test harness closes it with
        # the leading '    }' in the tests field.  Some LLM completions close the method inline
        # (e.g. 'return x; }') and then keep generating duplicate overloads.  None of the stop
        # tokens match '}\n    //' so all the extra code is included, the tests' '}' ends up
        # closing the *class* instead of the method, and 'main' lands at file scope, producing
        # "class, interface, or enum expected" (and a cascade of bogus "assert is keyword" errors
        # that are really just the parser choking on statements at file scope).
        # Fix: strip the completion back to the point just *before* it closes the method body so
        # the test harness can close it normally.
        # if language == "java":
        #     completion = MultiPLECodeAssertion._trim_java_completion(completion)

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
    def _trim_java_completion(completion: str) -> str:
        """Strip a Java completion back to just before it closes the open method body.

        MultiPL-E Java prompts end with an open method signature (depth 1 relative to the
        completion start).  The test harness supplies the closing '}' for that method.  When an
        LLM completes the method body *and* closes it inline (e.g. 'return x; }') and then
        continues with duplicate overloads, the class structure breaks: the test harness's '}'
        closes the class instead of the method, putting 'main' at file scope.

        This method scans the completion tracking brace depth (starting at 1 = inside the method
        body).  The moment depth would drop to 0 (the method-closing '}'), it returns everything
        *before* that brace so the test harness can close the method normally.  If no such brace
        exists the completion is returned unchanged.

        String and character literals are handled to avoid counting braces inside them.
        """
        depth = 1  # we begin inside the open method body
        in_string = False
        in_char = False
        escape_next = False
        i = 0
        while i < len(completion):
            ch = completion[i]
            if escape_next:
                escape_next = False
                i += 1
                continue
            if in_string:
                if ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
                i += 1
                continue
            if in_char:
                if ch == "\\":
                    escape_next = True
                elif ch == "'":
                    in_char = False
                i += 1
                continue
            if ch == '"':
                in_string = True
            elif ch == "'":
                in_char = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # Return everything up to (but not including) this closing brace.
                    # The test harness will supply the '}' that closes the method.
                    return completion[:i]
            i += 1
        return completion

    @staticmethod
    def _execute_via_sandbox_run(full_code: str, sandbox_lang: str, timeout: int) -> tuple[bool, str]:
        """Use llm-sandbox's native session.run() for cpp, java, js."""
        image = getattr(DefaultImage, sandbox_lang.upper())
        pool = _get_or_create_pool(image, sandbox_lang)
        with SandboxSession(pool=pool, lang=sandbox_lang) as session:
            result: Any = session.run(full_code, timeout=timeout)
        return result.success(), result.stdout

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

        Each invocation gets a unique UUID-scoped directory inside the container so that
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

        pool = _get_or_create_pool(cfg.image, SupportedLanguage.PYTHON)

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
                    output = result.stdout

        return result.success(), output
