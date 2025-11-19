from typing import TYPE_CHECKING, Any

from eval_framework.llm.openai import OpenAIEmbeddingModel, OpenAIModel
from eval_framework.metrics.completion.aidanbench import AidanBenchMetric
from eval_framework.metrics.llm.graders.coherence_grader import CoherenceGrader
from eval_framework.shared.types import Completion
from eval_framework.tasks.base import NO_SUBJECT, BaseTask, Language, ResponseType, Sample
from eval_framework.utils.helpers import pairwise_cosine_similarity
from template_formatting.formatter import Message, Role

if TYPE_CHECKING:
    from eval_framework.llm.base import BaseLLM


COHERENCE_THRESHOLD = 15
NOVELTY_THRESHOLD = 0.15


class AidanBench(BaseTask[str]):
    """AidanBench (https://openreview.net/pdf?id=fz969ahcvJ)."""

    NAME = "AidanBench"
    DATASET_PATH = "Aleph-Alpha-Research/aidanbench"
    SAMPLE_SPLIT = "train"
    FEWSHOT_SPLIT = "train"
    RESPONSE_TYPE = ResponseType.COMPLETION
    METRICS = [AidanBenchMetric]
    SUBJECTS = [NO_SUBJECT]
    LANGUAGE = {NO_SUBJECT: Language.ENG}

    def __init__(self, num_fewshot: int = 0) -> None:
        super().__init__(num_fewshot)
        assert num_fewshot == 0, "AidanBench does not support few-shot prompting."
        self._coherence_grader = CoherenceGrader(grading_model=OpenAIModel())
        self._embedding_model = OpenAIEmbeddingModel()

    def _get_instruction_text(self, item: dict[str, Any]) -> str:
        item_prompt = item["prompt"]
        base_prompt = (
            "Answer the following question:.\n"
            "<question>" + item_prompt + "</question>\n"
            "Provide your answer in <answer></answer> XML tags.\n"
        )
        base_prompt += (
            "Your response should be one direct answer. "
            "Only provide one answer. DO NOT list multiple answers. Please try to be concise.\n"
        )
        return base_prompt

    def _get_ground_truth(self, item: dict[str, Any]) -> str | None:
        return None

    def _calculate_novelty_score(self, messages: list[Message]) -> float:
        assert messages[0].role == Role.USER
        assert all(msg.role != Role.USER for msg in messages[1:]), "Only the first message should be from USER"
        messages_without_instruction = messages[1:]
        messages_without_instruction = [[m] for m in messages_without_instruction]  # input format for embedding model
        if len(messages_without_instruction) == 1:
            return 1.0  # if there's only one response, it's by definition novel
        all_embeddings = self._embedding_model.generate_from_messages(messages_without_instruction)
        new_embedding = all_embeddings[-1]
        previous_embeddings = all_embeddings[:-1]
        similarities = pairwise_cosine_similarity([new_embedding], previous_embeddings)
        assert len(similarities) == 1
        similarities = similarities[0]  # "squeeze"
        assert len(similarities) == len(previous_embeddings)
        return 1 - max(similarities)

    def _sample_fewshot_examples(self, item: dict[str, Any]) -> list[dict]:
        return []

    def _fuse_messages(self, messages: list[Message]) -> list[Message]:
        """
        Takes a list of messages and fuses them into a single message:
        A USER message that also contains all previous model responses, wrapped for the next iterative generation step.
        """
        assert len(messages) >= 2, "There must be at least one USER and one ASSISTANT message"
        assert messages[0].role == Role.USER
        assert all(msg.role == Role.ASSISTANT for msg in messages[1:]), "Only the first message should be from USER"

        instruction_message = messages[0].content
        previous_answers = [msg.content for msg in messages[1:]]

        previous_answers_str = "\n\n".join(
            [
                f"<previous_answer id='{i + 1}'>\n{answer}\n</previous_answer>"
                for i, answer in enumerate(previous_answers)
            ]
        )
        instruction_message += (
            "IMPORTANT: Provide an answer you *HAVE NOT* given previously.\n"
            "Your previous answers are inside of <previous_answers></previous_answers> XML tags.\n"
            "<previous_answers>\n" + previous_answers_str + "\n</previous_answers>"
        )
        return [Message(role=Role.USER, content=instruction_message)]

    def _generation_loop(
        self, llm: "BaseLLM", stop_sequences: list[str] | None, max_tokens: int | None, initial_samples: list[Sample]
    ) -> tuple[list[list[Message]], list[Exception | None]]:
        initial_messages = [s.messages for s in initial_samples]
        samples = [(s, False) for s in initial_samples]  # (sample, is_done)
        message_history = [msg for msg in initial_messages]  # to keep track of all iterative model responses
        errors = [None for _ in message_history]
        while not all(is_done for _, is_done in samples):
            # iterative generation loop
            not_done_idx = [i for i, (_, is_done) in enumerate(samples) if not is_done]
            new_completions = super().generate_completions(
                llm,
                [samples[i][0] for i in not_done_idx],
                stop_sequences=stop_sequences,
                max_tokens=max_tokens,
            )
            new_completion_messages = [c.messages for c in new_completions]
            new_errors = [c.error for c in new_completions]

            new_samples = [s for s in samples]
            for idx, (completion_msgs, error) in zip(not_done_idx, zip(new_completion_messages, new_errors)):
                old_sample = samples[idx][0]
                message_history[idx].append(completion_msgs[-1])  # add latest model response to history
                errors[idx] = error

                assert completion_msgs[0].role == Role.USER and completion_msgs[-1].role == Role.ASSISTANT
                coherence_score = self._coherence_grader.grade(
                    instruction=old_sample.messages[0].content,  # only pass initial instruction
                    completion=completion_msgs[-1].content,
                    language=Language.ENG,
                ).coherence_score

                novelty_score = self._calculate_novelty_score(message_history[idx])

                fused_message = self._fuse_messages(message_history[idx])
                new_sample = Sample(
                    id=old_sample.id,
                    subject=old_sample.subject,
                    ground_truth=old_sample.ground_truth,
                    messages=fused_message,
                    context=old_sample.context,
                    possible_completions=old_sample.possible_completions,
                )
                if coherence_score < COHERENCE_THRESHOLD or novelty_score < NOVELTY_THRESHOLD:
                    # Fail! Stop generating
                    new_samples[idx] = (new_sample, True)
                else:
                    # Continue generating
                    new_samples[idx] = (new_sample, False)
            samples = new_samples
        return message_history, errors

    def generate_completions(
        self,
        llm: "BaseLLM",
        samples: list[Sample],
        stop_sequences: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> list[Completion]:
        assert all(len(s.messages) == 1 and s.messages[0].role == Role.USER for s in samples), (
            "Each sample must have exactly one USER message."
        )
        all_message_histories, errors = self._generation_loop(llm, stop_sequences, max_tokens, samples)

        completion_list = []
        for idx, sample in enumerate(samples):
            messages = all_message_histories[idx]
            error = errors[idx]

            completion_list.append(
                Completion(
                    id=sample.id,
                    subject=sample.subject,
                    ground_truth=sample.ground_truth,
                    prompt=sample.messages[0].content,
                    prompt_sequence_positions=None,
                    concat_compression=None,
                    messages=messages,
                    completion="".join([msg.content for msg in messages if msg.role == Role.ASSISTANT]),
                    raw_completion="".join([msg.content for msg in messages if msg.role == Role.ASSISTANT]),
                    raw_completion_sequence_positions=None,
                    context=sample.context,
                    error=error,
                )
            )

        return completion_list
