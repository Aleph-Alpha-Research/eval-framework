import base64
import logging
import os
import random
import re
import string
from pathlib import Path
from typing import Literal, NamedTuple, Callable, Any

import dill
import numpy as np
from llm_sandbox import SandboxSession

logger = logging.getLogger(__name__)

RANDOM_SEED = 42  # hacky way to get around circular import
redis_warning_printed = False


def raise_errors() -> bool:
    return os.environ.get("DEBUG", "FALSE").lower() == "true"


def get_n_letters(n: int) -> list[str]:
    return list(string.ascii_uppercase)[: max(0, n)]


def run_python_code(
    code: str,
    image: str | None = None,
    input_files: list[tuple[str, str]] | None = None,
    timeout: int = 60,
    packages: list[str] | None = None,
) -> str:
    """
    Run code in a sandboxed environment.
    :param code: The code to run.
    :param image: Docker image to use.
    :param input_files: pairs of host and docker paths, host files will be copied to the docker.
    :param timeout: Timeout in seconds, 0 if no timeout.
    :param packages: List of python packages to install with pip.
    :return: The output of the code.
    """
    with SandboxSession(lang="python", image=image, keep_template=True, commit_container=False) as session:
        for host_file, docker_file in input_files or []:
            session.copy_to_runtime(host_file, docker_file)

        if timeout > 0:  # hack-add timeout from coreutils to the command executed
            session.orig_execute_command = session.execute_command
            session.execute_command = lambda command: session.orig_execute_command(f"timeout {timeout} {command}")

        return session.run(code, libraries=packages).text.strip()


class ExecutionResult(NamedTuple):
    """
    A named tuple to store the result of code execution.

    Attributes:
        success (bool): Indicates if the execution was successful.
        output (str): Contains the output or error messages from the execution.
    """

    success: bool
    output: str


class CallableSerializer:
    @staticmethod
    def encode(fn: Callable[..., Any]) -> str:
        try:
            return base64.b64encode(dill.dumps(fn)).decode("utf-8")
        except:
            raise

    @staticmethod
    def decode(fn: str) -> Callable[..., Any]:
        try:
            return dill.loads(base64.b64decode(fn.encode("utf-8")))
        except:
            raise


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
    return ExecutionResult(False, f"Could not determine test results, potentially due to timeout. Output: {output}")


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


class Editor:
    def __init__(self, language: Literal["en", "de"] = "en", seed: int = RANDOM_SEED) -> None:
        self.np_rng = np.random.RandomState(seed)
        self.rng = random.Random(seed)
        if language == "en":
            self.letters = string.ascii_lowercase
        elif language == "de":
            self.letters = string.ascii_lowercase + "ßöäü"
        else:
            raise NotImplementedError

    @staticmethod
    def _split_sentence(sentence: str) -> tuple[list[str], list[str], bool]:
        words = re.findall(r"\w+", sentence)
        spaces = re.findall(r"[^\w]+", sentence)
        has_leading_space = not words or sentence[: len(words[0])] != words[0]
        return words, spaces, has_leading_space

    @staticmethod
    def _recombine(words: list[str], spaces: list[str], has_leading_space: bool) -> str:
        if has_leading_space:
            combined_lists = sum([[w, s] for w, s in zip(words, spaces[1:])], [spaces[0]])
        else:
            combined_lists = sum([[w, s] for w, s in zip(words, spaces)], [])
        if len(words) > len(spaces) - (1 if has_leading_space else 0):
            combined_lists.append(words[-1])
        return "".join(combined_lists)

    @staticmethod
    def _get_word_probs(words: list[str]) -> np.ndarray:
        # We sample words proportional to their length - 1,
        # This means we ignore one-character words such as "I" and "a",
        # because these can't be transposed or split
        lengths = np.array([len(word) - 1 for word in words])
        probs = lengths / np.sum(lengths)
        return probs

    @staticmethod
    def _transpose(word: str, idx1: int, idx2: int) -> str:
        assert abs(idx2 - idx1) == 1, "idx1 and idx2 are not next to each other"
        if idx1 > idx2:
            idx1, idx2 = idx2, idx1
        return word[:idx1] + word[idx2] + word[idx1] + word[idx2 + 1 :]

    @staticmethod
    def _delete(word: str, idx: int) -> str:
        return word[:idx] + word[idx + 1 :]

    @staticmethod
    def _insert(word: str, idx: int, letter: str) -> str:
        assert len(letter) == 1, "`letter` is not a single character"
        return word[:idx] + letter + word[idx:]

    @staticmethod
    def _change_casing(word: str, idx: int) -> str:
        character = word[idx]
        if character.islower():
            character = character.upper()
        else:
            character = character.lower()
        return word[:idx] + character + word[idx + 1 :]

    @staticmethod
    def _split_word(word: str, idx: int) -> str:
        return word[:idx] + " " + word[idx:]

    def _edit_word(self, word: str, num_edits: int) -> str:
        # NB: It could be that two edits cancel each other out
        # but the chance of this is sufficiently small that it doesn't
        # make sense to complicate the code to fix this
        if num_edits == 0:
            return word

        for _ in range(num_edits):
            # upweighted change casing
            choices = ["insert", "change_casing", "change_casing"]
            if len(word) > 1:
                choices.extend(["transpose", "split_word"])
            if len(word) > 4:
                # use delete more sparingly since it has a big impact
                choices.extend(["delete"])

            edit_function = self.rng.choice(choices)
            if edit_function == "transpose":
                idx = self.rng.randint(0, len(word) - 2)
                word = self._transpose(word, idx, idx + 1)
            elif edit_function == "delete":
                idx = self.rng.randint(1, len(word) - 2)
                word = self._delete(word, idx)
            elif edit_function == "insert":
                idx = self.rng.randint(0, len(word) - 1)
                letter = self.rng.choice(self.letters)
                word = self._insert(word, idx, letter)
            elif edit_function == "change_casing":
                idx = self.rng.randint(0, len(word) - 1)
                word = self._change_casing(word, idx)
            elif edit_function == "split_word":
                idx = self.rng.randint(1, len(word) - 1)
                word = self._split_word(word, idx)

        return word

    def __call__(self, sentence: str, character_edit_change: float, unmodifiable_words: list[str] | None = None) -> str:
        words, spaces, has_leading_space = self._split_sentence(sentence)

        num_characters = sum(map(len, words))
        num_edits = int(num_characters * character_edit_change)
        if num_edits == 0:
            return sentence

        probs = self._get_word_probs(words)
        edits_per_word = self.np_rng.multinomial(num_edits, probs)
        unmodifiable_words_set = set([w.lower() for w in unmodifiable_words or []])
        edited_words = []
        for edits, word in zip(edits_per_word, words):
            if word.lower() not in unmodifiable_words_set:
                edited_words.append(self._edit_word(word, int(edits)))
            else:
                edited_words.append(word)
        return self._recombine(edited_words, spaces, has_leading_space)


class HatPaperEditor:
    # Used for Section 4.4 in the HAT paper (https://openreview.net/pdf?id=tU074jg2vS).

    def __init__(self, seed: int = RANDOM_SEED) -> None:
        self.rng = random.Random(seed)

    def _get_indices(self, input_text: str, pct: float, unmodifiable_words: list[str] | None = None) -> list[int]:
        indices = [
            i + 1
            for i, c in enumerate(input_text[1:-1])
            if c.isalnum() and input_text[i].isalnum() and input_text[i + 2].isalnum()
        ]
        for word in unmodifiable_words or []:
            for match in re.finditer(r"\b" + word + r"\b", input_text, re.IGNORECASE):
                indices = [i for i in indices if i < match.start(0) or i >= match.end(0)]
        return self.rng.sample(indices, int(len(indices) * pct))

    def permute_chars_in_string(
        self, input_text: str, permute_pct: float, unmodifiable_words: list[str] | None = None
    ) -> str:
        """
        Randomly permute permute_pct characters in the input string.

        Only permutes within words (whitespaces and first word chars are preserved).
        """
        chars_to_permute = self._get_indices(input_text, permute_pct, unmodifiable_words)
        permuted_text = list(input_text)
        for char_index in chars_to_permute:
            permuted_text[char_index], permuted_text[char_index + 1] = (
                permuted_text[char_index + 1],
                permuted_text[char_index],
            )
        return "".join(permuted_text)

    def replace_chars_in_string(
        self, input_text: str, replace_pct: float, unmodifiable_words: list[str] | None = None
    ) -> str:
        """
        Randomly replace replace_pct characters in the input string with replace_char.

        Only replaces within words (whitespaces and first and last word chars are preserved).
        """
        chars_to_replace = self._get_indices(input_text, replace_pct, unmodifiable_words)
        replaced_text = list(input_text)
        for char_index in chars_to_replace:
            replace_char = chr(self.rng.randint(33, 126))  # ASCII printable characters
            replaced_text[char_index] = replace_char
        return "".join(replaced_text)

    def delete_chars_in_string(
        self, input_text: str, delete_pct: float, unmodifiable_words: list[str] | None = None
    ) -> str:
        """
        Randomly delete delete_pct characters in the input string.

        Only deletes within words (whitespaces and first and last word chars are preserved).
        """
        chars_to_delete = self._get_indices(input_text, delete_pct, unmodifiable_words)
        deleted_text = list(input_text)
        for char_index in chars_to_delete:
            deleted_text[char_index] = ""  # do not delete list entry since then the length of the list changes
        return "".join(deleted_text)

    def upper_case_string(self, input_text: str) -> str:
        """
        Upper case all characters in the input string.
        """
        return input_text.upper()


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
