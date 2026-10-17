from typing import Any

from eval_framework.external.multi_if_impl.instructions import INSTRUCTION_DICT
from eval_framework.metrics.base import BaseMetric, MetricResult
from eval_framework.shared.types import BaseMetricContext, Completion, extract_context_metric
from template_formatting.formatter import Role


class MultiIFMetricContext(BaseMetricContext):
    key: str
    language: str
    # One entry per turn. Each turn's instructions include all earlier turns' instructions still in force.
    turn_prompts: list[str]
    turn_instruction_id_lists: list[list[str]]
    turn_kwargs: list[list[dict[str, Any]]]


def _follows(response: str, instruction_id: str, kwargs: dict[str, Any]) -> bool:
    instruction = INSTRUCTION_DICT[instruction_id](instruction_id)
    instruction.build_description(**kwargs)
    return bool(instruction.check_following(response))


def _loose_variants(response: str) -> list[str]:
    lines = response.split("\n")
    trimmed = [
        response,
        "\n".join(lines[1:]).strip(),
        "\n".join(lines[:-1]).strip(),
        "\n".join(lines[1:-1]).strip(),
    ]
    return trimmed + [variant.replace("*", "") for variant in trimmed]


class MultiIFMetric(BaseMetric[Completion]):
    """Scores every turn of a Multi-IF conversation against that turn's instructions.

    Each turn is its own metric (``Multi-IF Turn <n>``) whose keys are IFEval's four accuracies, so the framework's
    ``Average Multi-IF Turn <n>`` is the paper's per-turn score: the mean of the four, averaged over languages.
    """

    NAME = "Multi-IF"
    _PROMPT_KEYS = ("prompt_level_strict_acc", "prompt_level_loose_acc")

    def calculate(self, response: Completion) -> list[MetricResult]:
        context = extract_context_metric(response, MultiIFMetricContext)
        num_turns = len(context.turn_prompts)

        if response.error is not None:
            return [
                MetricResult(
                    metric_name=f"{self.NAME} Turn {turn}/{key}",
                    value=None,
                    higher_is_better=True,
                    error=response.error,
                )
                for turn in range(1, num_turns + 1)
                for key in self._PROMPT_KEYS
            ]

        assert response.messages is not None
        turn_responses = [message.content for message in response.messages if message.role == Role.ASSISTANT]
        assert len(turn_responses) == num_turns, f"Expected {num_turns} assistant turns, got {len(turn_responses)}"

        results = []
        for turn, (turn_response, instruction_ids, kwargs_list) in enumerate(
            zip(turn_responses, context.turn_instruction_id_lists, context.turn_kwargs), start=1
        ):
            variants = _loose_variants(turn_response)
            # Upstream skips the check for an empty response (strict) and for a blank variant (loose).
            strict = [
                bool(turn_response) and _follows(turn_response, i, kwargs)
                for i, kwargs in zip(instruction_ids, kwargs_list)
            ]
            loose = [
                any(variant.strip() and _follows(variant, i, kwargs) for variant in variants)
                for i, kwargs in zip(instruction_ids, kwargs_list)
            ]
            # One instruction-level result per instruction, so their average is over instructions, not prompts.
            values_by_key = {
                "prompt_level_strict_acc": [all(strict)],
                "prompt_level_loose_acc": [all(loose)],
                "inst_level_strict_acc": strict,
                "inst_level_loose_acc": loose,
            }
            results += [
                MetricResult(metric_name=f"{self.NAME} Turn {turn}/{key}", value=float(v), higher_is_better=True)
                for key, values in values_by_key.items()
                for v in values
            ]
        return results
