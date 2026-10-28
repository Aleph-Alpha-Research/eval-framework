from typing import Literal, NamedTuple

from eval_framework.tasks.base import Language

LanguageCode = Literal["CS", "DA", "DE", "ES", "FI", "FR", "IT", "NL", "PL"]


class EllamindLanguage(NamedTuple):
    """A language of the EllaMind datasets: the framework language and the dataset config holding it."""

    language: Language
    config: str


ELLAMIND_LANGUAGES: dict[LanguageCode, EllamindLanguage] = {
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
    """The ``(language, config)`` of a language token, e.g. ``ellamind_language("DE")``."""
    return ELLAMIND_LANGUAGES[code]
