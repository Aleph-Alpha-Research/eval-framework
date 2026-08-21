import atexit
import base64
import logging
import re
import string
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple, overload

import dill
from llm_sandbox import SandboxSession
from llm_sandbox.const import DefaultImage
from llm_sandbox.pool import PoolConfig, create_pool_manager
from llm_sandbox.pool.base import ContainerPoolManager

logger = logging.getLogger(__name__)

redis_warning_printed = False


class classproperty[T]:
    """Descriptor supporting property-like access on classes and instances."""

    def __init__(self, fget: Callable[[Any], T]) -> None:
        self.fget = fget

    @overload
    def __get__(self, obj: None, owner: type[Any]) -> T: ...

    @overload
    def __get__(self, obj: object, owner: type[Any] | None = None) -> T: ...

    def __get__(self, obj: object | None, owner: type[Any] | None = None) -> T:
        cls = owner if owner is not None else type(obj)
        return self.fget(cls)


_pools: dict[tuple[str | None, tuple[str, ...] | None], ContainerPoolManager] = {}
_pools_lock = threading.Lock()


# A ContainerPoolManager (from llm_sandbox) manages a pool of pre-warmed Docker containers
# ready to execute sandboxed code. Spinning up a new container on every code execution is slow
# and resource-intensive, so the pool keeps a configurable number of containers alive and idle
# between uses. Each pool is scoped to a specific (image/dockerfile, packages) combination,
# ensuring containers already have the right dependencies installed. The process-level singleton
# cache in _pools means a given configuration only pays the startup cost once for the lifetime
# of the process, and concurrent callers are protected by a lock.
def get_or_create_pool(
    image: str | None = None,
    dockerfile: str | None = None,
    packages: list[str] | None = None,
    lang: str = "python",
    min_pool_size: int = 1,
    max_pool_size: int = 1,
    max_container_uses: int = 100,
    runtime_configs: dict[str, str] | None = None,
) -> ContainerPoolManager:
    assert image or dockerfile, "Either image or dockerfile must be provided"
    key = (image or dockerfile, tuple(packages) if packages else None)
    with _pools_lock:
        if key not in _pools:
            pool = create_pool_manager(
                config=PoolConfig(
                    min_pool_size=min_pool_size, max_container_uses=max_container_uses, max_pool_size=max_pool_size
                ),
                lang=lang,
                image=image,
                dockerfile=dockerfile,
                keep_template=True,
                libraries=packages,
                runtime_configs=runtime_configs,
            )
            _pools[key] = pool
        return _pools[key]


def close_pools() -> None:
    with _pools_lock:
        pools = list(_pools.values())
        _pools.clear()
    for pool in pools:
        try:
            pool.close()
        except Exception:
            logger.exception("Error closing sandbox container pool")


atexit.register(close_pools)


def get_n_letters(n: int) -> list[str]:
    return list(string.ascii_uppercase)[: max(0, n)]


class DockerReturnedEmptyOutput(Exception):
    """Raised when the docker returns an empty output."""


def run_python_code(
    code: str,
    image: str | None = None,
    dockerfile: str | None = None,
    input_files: list[tuple[str, str]] | None = None,
    timeout: int = 60,
    packages: list[str] | None = None,
    runtime_configs: dict[str, str] | None = None,
) -> str:
    """
    Run code in a sandboxed environment.
    :param code: The code to run.
    :param image: Docker image to use.
    :param dockerfile: Dockerfile to use.
    :param input_files: pairs of host and docker paths, host files will be copied to the docker.
    :param timeout: Timeout in seconds, 0 if no timeout.
    :param packages: List of python packages to install with pip.
    :return: The output of the code.
    """

    # Only one of image or dockerfile should be provided.
    # we fallback to the default python image if no dockerfile is provided.
    resolved_image = image or (DefaultImage.PYTHON if not dockerfile else None)
    pool = get_or_create_pool(
        resolved_image,
        packages=packages,
        dockerfile=dockerfile,
        runtime_configs=runtime_configs,
    )
    with SandboxSession(pool=pool, lang="python") as session:
        for host_file, docker_file in input_files or []:
            session.copy_to_runtime(host_file, docker_file)

        output = session.run(code, timeout=timeout)
        out = (output.stderr + output.stdout).strip()
        if isinstance(out, bytes):
            out = out.decode("utf-8")

        if not out.strip():
            raise DockerReturnedEmptyOutput("Docker returned an empty output.")
        return out


def unittest_merge_snippets(code: str, test_code: str) -> str:
    # Add unittest.main() if not present (note that without "if" sometimes it just reports
    # "Ran 0 tests" errorneously).
    if "unittest.main(" not in test_code:
        test_code += "\n\nif __name__ == '__main__':\n  unittest.main()"

    # Combine the implementation code and test code
    combined_code = code + "\n\n" + test_code
    return combined_code


class ExecutionResult(NamedTuple):
    """
    A named tuple to store the result of code execution.

    Attributes:
        success (bool): Indicates if the execution was successful.
        output (str): Contains the output or error messages from the execution.
    """

    success: bool
    output: str


def execute_python_code_with_tests(
    code: str,
    test_code: str,
    dockerfile: str | None,
    package_mapping: dict[str, str | None],
    merge_code_fn: Callable[[str, str], str],
    image: str | None,
    timeout: int,
    parse_output_fn: Callable[[str], ExecutionResult],
) -> ExecutionResult:
    """
    Executes the given code with test cases in a sandboxed environment.

    :param code: The code to be tested.
    :param test_code: The test cases to run against the code.
    :param package_mapping: Mapping of package names to install commands.
    :param merge_code_fn: function to merge LLM and test code
    :param image: Docker image to use.
    :param timeout: Timeout for the execution in seconds.
    :param parse_otuput_fn: function to parse docker execution output
    :return: An ExecutionResult named tuple with success status and output or errors.
    """
    combined_code = merge_code_fn(code, test_code)

    packages = get_external_dependencies(combined_code, package_mapping)

    # Run the combined code in the sandbox
    output = run_python_code(
        combined_code,
        image=image,
        dockerfile=dockerfile,
        timeout=timeout,
        packages=packages,
    )

    # Parse the output to determine success
    return parse_output_fn(output)


class SerializationError(Exception):
    """Base exception for callable serialization errors."""

    pass


class EncodingError(SerializationError):
    """Raised when encoding a callable fails."""

    pass


class DecodingError(SerializationError):
    """Raised when decoding a callable fails."""

    pass


class CallableSerializer:
    @staticmethod
    def encode(fn: Callable[..., Any]) -> str:
        try:
            serialized = dill.dumps(fn)
            return base64.b64encode(serialized).decode("utf-8")
        except Exception as e:
            raise EncodingError(f"Failed to encode callable {fn}: {e}") from e

    @staticmethod
    def decode(fn_str: str) -> Callable[..., Any]:
        try:
            decoded = base64.b64decode(fn_str.encode("utf-8"))
            return dill.loads(decoded)
        except Exception as e:
            raise DecodingError(f"Failed to decode callable from string: {e}") from e


def _parse_unittest_output(output: str) -> ExecutionResult:
    """Parse the unittest output to determine success and format the result."""
    # Check for unittest success pattern
    if "OK" in output and "FAILED" not in output:
        # Extract the test summary if possible
        match = re.search(r"Ran (\d+) tests? in [\d.]+s", output)
        if match:
            test_count = match.group(1)
            test_output = f"All {test_count} tests completed successfully."
        else:
            test_output = "All tests completed successfully."

        return ExecutionResult(True, test_output)

    # Check for unittest failure pattern
    elif "FAILED" in output:
        # Try to extract failure details
        match = re.search(r"FAILED \((.+)\)", output)
        if match:
            failure_details = match.group(1)
            return ExecutionResult(False, f"Tests failed: {failure_details}\n{output}")
        else:
            return ExecutionResult(False, f"Tests failed: {output}")

    # Check for common error patterns
    elif "AssertionError" in output:
        return ExecutionResult(False, f"Test failed with assertion error: {output}")
    elif "Error:" in output or "Exception:" in output:
        return ExecutionResult(False, f"Error during execution: {output}")

    # If we can't determine success/failure, return the raw output
    return ExecutionResult(
        False,
        f"Could not determine test results, potentially due to timeout. Output: {output}",
    )


def get_external_dependencies(code: str, package_mapping: dict[str, str | None]) -> list[str]:
    """Identify external dependencies in the code."""
    _, packages = extract_imports(code)

    external_packages = []
    for pkg in packages:
        if pkg in package_mapping and package_mapping[pkg] is not None:
            external_packages.append(package_mapping[pkg])
    return external_packages  # type: ignore[return-value]


def extract_imports(code: str) -> tuple[list[str], set[str]]:
    """Extract all import statements and the imported packages from code."""
    # Pattern for 'import x' or 'import x, y, z'
    import_pattern = r"^import\s+([\w\s,.]+)"

    # Pattern for 'from x import y'
    from_pattern = r"^from\s+([\w.]+)\s+import\s+"

    imports = []
    packages = set()

    for line in code.split("\n"):
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Handle 'import x' or 'import x, y, z'
        import_match = re.match(import_pattern, line)
        if import_match:
            imports.append(line)
            # Extract all packages from the import statement
            imported_items = import_match.group(1).split(",")
            for item in imported_items:
                # Clean up and get the base package name
                pkg = item.strip().split(".")[0].split(" as ")[0]
                if pkg:
                    packages.add(pkg)
            continue

        # Handle 'from x import y'
        from_match = re.match(from_pattern, line)
        if from_match:
            imports.append(line)
            # Get the base package name
            pkg = from_match.group(1).split(".")[0]
            if pkg:
                packages.add(pkg)

    return imports, packages


def get_docker_address() -> str:
    # If it's docker-in-docker: the new docker actually started in host, so we need to use the host's IP
    # See https://stackoverflow.com/questions/48546124/what-is-the-linux-equivalent-of-host-docker-internal
    return "172.17.0.1" if Path("/.dockerenv").exists() else "localhost"


# these are all the packages that occur in the BigCodeBench dataset
BIG_CODE_BENCH_PACKAGE_MAPPING = {
    # Standard library packages (built-in)
    "array": None,
    "ast": None,
    "base64": None,
    "binascii": None,
    "bisect": None,
    "calendar": None,
    "cgi": None,
    "cmath": None,
    "codecs": None,
    "collections": None,
    "configparser": None,
    "csv": None,
    "ctypes": None,
    "datetime": None,
    "decimal": None,
    "difflib": None,
    "email": None,
    "enum": None,
    "errno": None,
    "fnmatch": None,
    "ftplib": None,
    "functools": None,
    "getpass": None,
    "glob": None,
    "gzip": None,
    "hashlib": None,
    "heapq": None,
    "hmac": None,
    "html": None,
    "http": None,
    "importlib": None,
    "inspect": None,
    "io": None,
    "ipaddress": None,
    "itertools": None,
    "json": None,
    "logging": None,
    "math": None,
    "mimetypes": None,
    "multiprocessing": None,
    "operator": None,
    "os": None,
    "pathlib": None,
    "pickle": None,
    "pkgutil": None,
    "platform": None,
    "queue": None,
    "random": None,
    "re": None,
    "select": None,
    "secrets": None,
    "shlex": None,
    "shutil": None,
    "signal": None,
    "smtplib": None,
    "socket": None,
    "sqlite3": None,
    "ssl": None,
    "statistics": None,
    "string": None,
    "struct": None,
    "subprocess": None,
    "sys": None,
    "tarfile": None,
    "textwrap": None,
    "threading": None,
    "time": None,
    "turtle": None,
    "types": None,
    "typing": None,
    "unicodedata": None,
    "urllib": None,
    "uuid": None,
    "warnings": None,
    "xml": None,
    "zipfile": None,
    "zlib": None,
    "zoneinfo": None,
    # External packages (need pip install)
    "PIL": "pillow",
    "Crypto": "pycryptodome",
    "Levenshtein": "python-Levenshtein",
    "blake3": "blake3",
    "bs4": "beautifulsoup4",
    "chardet": "chardet",
    "cryptography": "cryptography",
    "cv2": "opencv-python",
    "dateutil": "python-dateutil",
    "django": "django",
    "docx": "python-docx",
    "faker": "Faker",
    "flask": "flask",
    "flask_login": "flask-login",
    "flask_mail": "flask-mail",
    "flask_restful": "flask-restful",
    "flask_wtf": "flask-wtf",
    "folium": "folium",
    "gensim": "gensim",
    "geopandas": "geopandas",
    "geopy": "geopy",
    "holidays": "holidays",
    "keras": "keras",
    "librosa": "librosa",
    "lxml": "lxml",
    "matplotlib": "matplotlib",
    "mechanize": "mechanize",
    "mpl_toolkits": "matplotlib",
    "natsort": "natsort",
    "nltk": "nltk",
    "numpy": "numpy",
    "openpyxl": "openpyxl",
    "pandas": "pandas",
    "prettytable": "prettytable",
    "psutil": "psutil",
    "pyquery": "pyquery",
    "pytesseract": "pytesseract",
    "python_http_client": "python-http-client",
    "pytz": "pytz",
    "regex": "regex",
    "requests": "requests",
    "rsa": "rsa",
    "scipy": "scipy",
    "seaborn": "seaborn",
    "sendgrid": "sendgrid",
    "shapely": "shapely",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "soundfile": "soundfile",
    "statsmodels": "statsmodels",
    "sympy": "sympy",
    "tensorflow": "tensorflow",
    "textblob": "textblob",
    "texttable": "texttable",
    "werkzeug": "werkzeug",
    "wikipedia": "wikipedia",
    "wordcloud": "wordcloud",
    "wordninja": "wordninja",
    "wtforms": "wtforms",
    "xlwt": "xlwt",
    "xmltodict": "xmltodict",
    "yaml": "pyyaml",
}
