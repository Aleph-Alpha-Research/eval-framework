from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from typing import TypeVar

import lingua
from pycountry import languages


class LanguageNotSupportedError(ValueError):
    """Raised in case language in the input is not compatible with the languages supported in the task."""


Config = TypeVar("Config")

_language_detector = lingua.LanguageDetectorBuilder.from_languages(
    lingua.Language.ENGLISH,
    lingua.Language.GERMAN,
    lingua.Language.SPANISH,
    lingua.Language.ITALIAN,
    lingua.Language.FRENCH,
    lingua.Language.DUTCH,
    lingua.Language.PORTUGUESE,
    lingua.Language.FINNISH,
).build()

AVAILABLE_LANGUAGES = ["en", "de", "es", "it", "fr", "nl", "pt", "fi"]


@cache
def detect_language_of(string: str) -> lingua.Language | None:
    return _language_detector.detect_language_of(string)


@dataclass(frozen=True)
class Language:
    """A language identified by its `ISO 639-1 code <https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes>`_."""

    iso_639_1: str

    def get_name(self) -> str | None:
        language = languages.get(alpha_2=self.iso_639_1)
        return language.name if language else None

    def language_config(self, configs: Mapping["Language", Config]) -> Config:
        config = configs.get(self)
        if config is None:
            raise LanguageNotSupportedError(
                f"{self.iso_639_1} not in ({', '.join(lang.iso_639_1 for lang in configs)})"
            )
        return config

    def to_lingua_language(self) -> lingua.Language:
        iso_code = getattr(lingua.IsoCode639_1, self.iso_639_1.upper())
        language = lingua.Language.from_iso_code_639_1(iso_code)
        return language
