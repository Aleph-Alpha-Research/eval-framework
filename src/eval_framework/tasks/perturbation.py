import threading
from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from eval_framework.logger import logger
from eval_framework.tasks.base import RANDOM_SEED, BaseTask, Sample
from eval_framework.tasks.utils import Editor, HatPaperEditor


class PerturbationType(str, Enum):
    # Editor methods
    EDITOR = "editor"
    # Hat paper methods
    PERMUTE = "permute"
    REPLACE = "replace"
    DELETE = "delete"
    UPPERCASE = "uppercase"


class PerturbationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: PerturbationType = Field(default=PerturbationType.EDITOR)
    probability: float = Field(default=0.1, ge=0.0, le=1.0)
    seed: int = Field(default=RANDOM_SEED)
    verbose: bool = Field(default=False)


_DOCKER_LAUNCH_LOCK = threading.Lock()
_AUGMENTER_PORT = 0

SomeBaseTask = TypeVar("SomeBaseTask", bound=BaseTask[Any])


def create_perturbation_class[T: BaseTask](base_class: type[T], perturbation_config: PerturbationConfig) -> type[T]:
    # mypy seems to have trouble inferring the type
    class EditorPerturbation(base_class):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.perturbation_config = perturbation_config
            self.editor = Editor(
                language="de" if base_class.LANGUAGE == "German" else "en", seed=perturbation_config.seed
            )

        def _get_instruction_text(self, sample: Sample) -> str:
            text = super()._get_instruction_text(sample)
            if self.perturbation_config.verbose:
                logger.info(f"Perturbing text: {text}")
            result = self.editor(
                text, self.perturbation_config.probability, getattr(self, "PERTURBATION_UNMODIFIABLE_WORDS", [])
            )
            if self.perturbation_config.verbose:
                logger.info(f"Perturbed text: {result}")
            return result

    class HatPaperPerturbation(base_class):  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.perturbation_config = perturbation_config
            self.editor = HatPaperEditor(seed=perturbation_config.seed)

        def _get_instruction_text(self, sample: Sample) -> str:
            text = super()._get_instruction_text(sample)
            if self.perturbation_config.verbose:
                logger.info(f"Perturbing text: {text}")
            words = getattr(self, "PERTURBATION_UNMODIFIABLE_WORDS", [])
            if self.perturbation_config.type == PerturbationType.PERMUTE:
                result = self.editor.permute_chars_in_string(text, self.perturbation_config.probability, words)
            elif self.perturbation_config.type == PerturbationType.REPLACE:
                result = self.editor.replace_chars_in_string(text, self.perturbation_config.probability, words)
            elif self.perturbation_config.type == PerturbationType.DELETE:
                result = self.editor.delete_chars_in_string(text, self.perturbation_config.probability, words)
            elif self.perturbation_config.type == PerturbationType.UPPERCASE:
                result = self.editor.upper_case_string(text)
            if self.perturbation_config.verbose:
                logger.info(f"Perturbed text: {result}")
            return result

    if perturbation_config.type == PerturbationType.EDITOR:
        return EditorPerturbation
    else:
        return HatPaperPerturbation
