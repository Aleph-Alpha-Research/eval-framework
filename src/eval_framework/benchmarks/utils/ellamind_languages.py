from dataclasses import dataclass
from typing import Literal

from eval_framework.tasks.task_style import Language

LanguageCode = Literal["CS", "DA", "DE", "ES", "FI", "FR", "IT", "NL", "PL"]


@dataclass(frozen=True)
class EllamindLanguage:
    """A language of the EllaMind datasets: the framework language and the dataset config holding it."""

    language: Language
    config: str


_ELLAMIND_LANGUAGES: dict[LanguageCode, EllamindLanguage] = {
    "CS": EllamindLanguage(Language.CES, "ces"),
    "DA": EllamindLanguage(Language.DAN, "dan"),
    "DE": EllamindLanguage(Language.DEU, "deu"),
    "ES": EllamindLanguage(Language.SPA, "spa"),
    "FI": EllamindLanguage(Language.FIN, "fin"),
    "FR": EllamindLanguage(Language.FRA, "fra"),
    "IT": EllamindLanguage(Language.ITA, "ita"),
    "NL": EllamindLanguage(Language.NLD, "nld"),
    "PL": EllamindLanguage(Language.POL, "pol"),
}


def ellamind_language(code: LanguageCode) -> EllamindLanguage:
    """The language and dataset config of a language token, e.g. ``ellamind_language("DE")``."""
    return _ELLAMIND_LANGUAGES[code]
