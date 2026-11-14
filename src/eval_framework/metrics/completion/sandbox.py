"""Running code in Docker sandboxes (via ``llm_sandbox``).

- ``execute_in_default_image``: for a single file in a language llm-sandbox supports natively (``SupportedLanguage``)
- ``execute_in_custom_image``: for everything else, e.g. a language llm-sandbox doesn't support, another image,
  extra files next to the code.
"""

import os
import tempfile
import uuid
from typing import Any, NamedTuple

from llm_sandbox import SandboxSession
from llm_sandbox.const import DefaultImage, SupportedLanguage

from eval_framework.tasks.utils import get_or_create_pool


def execute_in_default_image(
    full_code: str, sandbox_lang: str, timeout: int, runtime_configs: dict[str, str] | None = None
) -> tuple[bool, str]:
    """Run code with llm-sandbox's native session.run(), in llm-sandbox's default image for ``sandbox_lang``.

    runtime_configs is passed to the container pool (e.g. ``{"mem_limit": "1g"}``).
    """
    image = getattr(DefaultImage, sandbox_lang.upper())
    pool = get_or_create_pool(image=image, lang=sandbox_lang, runtime_configs=runtime_configs)
    with SandboxSession(pool=pool, lang=sandbox_lang, encoding_errors="replace") as session:
        result: Any = session.run(full_code, timeout=timeout)
    return result.success(), result.stdout + result.stderr


class CustomLangConfig(NamedTuple):
    image: str
    file_ext: str
    commands: list[str]


def execute_in_custom_image(
    full_code: str,
    cfg: CustomLangConfig,
    timeout: int,
    extra_host_files: dict[str, str] | None = None,
    runtime_configs: dict[str, str] | None = None,
) -> tuple[bool, str]:
    """Copy code (and optional extra files) into a language-specific Docker image and run.

    We open a SandboxSession with a dummy lang (SupportedLanguage.PYTHON) so that
    llm-sandbox sets up the container plumbing, but we never call session.run().
    Instead we write the code to a local temp file, copy it into the container with
    copy_to_runtime, and drive each compile/run command with execute_command.

    Each invocation gets a unique UUID-named directory inside the container so that
    concurrent calls never clobber each other's source files or build artefacts.

    extra_host_files maps container filename -> host path for any additional files that
    should be copied into container_dir before the commands run (e.g. dependency jars).

    runtime_configs is passed to the container pool (e.g. ``{"mem_limit": "1g"}``).
    """
    run_id = uuid.uuid4().hex
    # copy_to_runtime uses arcname=os.path.basename(src), so the local filename
    # must match the desired in-container filename for put_archive to place it correctly.
    code_filename = f"code.{cfg.file_ext}"
    container_dir = f"/tmp/{run_id}"
    code_file = f"{container_dir}/{code_filename}"
    output = ""

    pool = get_or_create_pool(
        image=cfg.image,
        lang=SupportedLanguage.PYTHON,
        min_pool_size=1,
        max_pool_size=1,
        runtime_configs=runtime_configs,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, code_filename)
        with open(tmp_path, "w") as f:
            f.write(full_code)

        with SandboxSession(pool=pool, lang=SupportedLanguage.PYTHON, encoding_errors="replace") as session:
            session.execute_command(f"mkdir -p {container_dir}")
            session.copy_to_runtime(tmp_path, code_file)
            for filename, host_path in (extra_host_files or {}).items():
                session.copy_to_runtime(host_path, f"{container_dir}/{filename}")
            if timeout > 0:  # hack-add timeout from coreutils to the command executed
                session.orig_execute_command = session.execute_command
                session.execute_command = lambda command: session.orig_execute_command(f"timeout {timeout} {command}")
            for cmd_template in cfg.commands:
                cmd = cmd_template.format(code_file=code_file, container_dir=container_dir)
                result: Any = session.execute_command(cmd)
                output = result.stdout + result.stderr
                if not result.success():
                    break

    return result.success(), output
