import base64
import logging
import re
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dill

logger = logging.getLogger(__name__)

redis_warning_printed = False


def extract_python_code_from_response(llm_response: str) -> str:
    """Extract the Python source from a chat-style LLM response that wraps its answer in markdown code fences.

    The response is expected to contain a single Python solution (e.g. HumanEval, MBPP, BigCodeBench). The
    first matching fenced block is returned, tried in this order: a ```python block nested inside a
    ```markdown block, a plain ```python block, a ```markdown block, and finally any ``` block. If the
    response contains no fences at all, it is assumed to be raw code and returned unchanged.
    """
    patterns = [
        r"```markdown.*?```python\s*(.*?)\s*```",
        r"```python\s*(.*?)\s*```",
        r"```markdown\s*(.*?)\s*```",
        r"```\s*(.*?)\s*```",
    ]
    for pattern in patterns:
        if matches := re.findall(pattern, llm_response, re.DOTALL):
            return matches[0].strip()
    return llm_response


def get_n_letters(n: int) -> list[str]:
    return list(string.ascii_uppercase)[: max(0, n)]


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
