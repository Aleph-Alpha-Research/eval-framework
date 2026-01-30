from typing import Any

from eval_framework.metrics.completion.grid_difference import GridDifference
from eval_framework.tasks.base import BaseTask, Language, ResponseType

SUBJECTS = [
    "1_random_cell_easy",
    "5_random_cell_easy",
    "10_random_cell_easy",
    "1_random_row_easy",
    "3_random_row_easy",
    "1_random_column_easy",
    "3_random_column_easy",
    "full_easy",
    "1_random_cell_hard",
    "5_random_cell_hard",
    "10_random_cell_hard",
    "1_random_row_hard",
    "3_random_row_hard",
    "1_random_column_hard",
    "3_random_column_hard",
    "full_hard",
]

SYSTEM_PROMPT = """You are given a structural material distribution represented as a grid. Each cell can have one of the following states:
- 'L' indicates applied load.
- 'V' indicates void.
- 'S' indicates support.

The goal is to predict the correct material distribution by filling in all {FILL_INSTRUCTION}, based on the surrounding structure and implicit physical reasoning (such as load paths, supports, and forces).

Important: The completed structure should use as little material as possible while remaining stable and plausible for carrying the applied forces. Minimize material usage unless necessary for structural support."""  # noqa: E501

PROMPT_TEMPLATE = """Below is the input grid with masked regions:

{GRID}

Please output the completed grid by replacing all {FILL_INSTRUCTION}.
Maintain the same format as the input: one row per line, cells separated by spaces, and the total number of rows and columns unchanged.
Return only the completed grid without any additional explanation."""  # noqa: E501

EASY_FILL_INSTRUCTION = "'V' cells with either '1' (solid) or '0' (empty)"

HARD_FILL_INSTRUCTION = (
    "'V' cells with a floating point number between 0 and 1, with one decimal place (e.g., 0.0, 0.1, 0.2, ..., 1.0)"
)


class SPHYR(BaseTask[str]):
    """SPhyR dataset: https://huggingface.co/datasets/philippds/SPhyR"""

    NAME = "SPHYR"
    DATASET_PATH = "philippds/SPhyR"
    SAMPLE_SPLIT = "test"
    FEWSHOT_SPLIT = ""
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [GridDifference]
    SUBJECTS = SUBJECTS
    PERTURBATION_UNMODIFIABLE_WORDS = None
    LANGUAGE = Language.ENG

    def __init__(self, num_fewshot: int = 0) -> None:
        assert num_fewshot == 0, "Fewshot is not supported for SPHYR"
        super().__init__(num_fewshot)

    def _grid_to_str(self, grid: list[list[str]]) -> str:
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    def _get_system_prompt_text(self, item: dict[str, Any]) -> str | None:
        FILL_INSTRUCTION = EASY_FILL_INSTRUCTION if "easy" in item["subject"] else HARD_FILL_INSTRUCTION
        return SYSTEM_PROMPT.format(FILL_INSTRUCTION=FILL_INSTRUCTION)

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        FILL_INSTRUCTION = EASY_FILL_INSTRUCTION if "easy" in item["subject"] else HARD_FILL_INSTRUCTION
        grid = self._grid_to_str(item["input_grid"])
        return PROMPT_TEMPLATE.format(GRID=grid, FILL_INSTRUCTION=FILL_INSTRUCTION)

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return self._grid_to_str(item["ground_truth"])
