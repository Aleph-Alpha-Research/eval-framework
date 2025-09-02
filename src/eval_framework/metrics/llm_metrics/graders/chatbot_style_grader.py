from collections.abc import Mapping

from eval_framework.llm.base import BaseLLM as StructuredOutputChatModel
from eval_framework.metrics.llm_metrics.graders.language import Language
from eval_framework.metrics.llm_metrics.graders.models import GradingOutput, PromptTemplate, parse_json_output


class ChatbotStyleGradingOutput(GradingOutput):
    thought_process: str | None
    is_chatbot_style: bool | None


class ChatbotStyleGrader:
    COMPLETION_KEY = "completion"
    PROMPT_TEMPLATES = {
        Language("de"): PromptTemplate(
            system_prompt="""Deine Aufgabe ist es zu klassifizieren, ob eine von einem Textgenerator gelieferte Antwort dem Stile eines Chatbots entspricht.

Hier sind einige Schlüsselmerkmale einer Antwort im Stile eines Chatbots:
* Sie leitet den Hauptinhalt mit Phrasen wie "Natürlich, ich helfe Dir gerne", "Na klar!" oder "Selbstverständlich kann ich" ein.
* Sie endet mit Phrasen wie "Ich hoffe, ich konnte Dir weiterhelfen!"
* Sie stellt Nachfragen an den Benutzer.
* Sie neigt dazu, überaus wortreich zu sein.
* Sie enthält Gesprächs- und Unterhaltungsfloskeln.
* Sie enthält Text, der zum Verständnis des Inhalts nicht zwingend notwendig ist.
* Sie bewahrt einen überaus freundlichen Ton.

Beachte, dass die Erfüllung von nur einem dieser Merkmale ausreicht, um die Antwort als Chatbot-Stil zu klassifizieren.

Gebe deine Bewertung in folgendem JSON-Format:
{
    "thought_process": str (Achte sehr genau auf die Antwort und argumentiere in ein paar Sätzen, ob die Antwort dem Chatbot-Stil folgt oder nicht),
    "is_chatbot_style": bool
}""",  # noqa: E501
            user_prompt=f"""**Antwort des Textgenerators**
{{{COMPLETION_KEY}}}""",
        ),
        Language("en"): PromptTemplate(
            system_prompt="""Your task is to classify if a text generation model's response follows a chatbot-style format.

Here are some key characteristics of a chatbot-style response:
* It introduces the main content with phrases like "Certainly, here is", "Sure!" or "Of course."
* It ends with phrases such as "I hope this helps!"
* It asks follow-up questions.
* It tends to be verbose.
* It tends to contain fluff that is not necessary to understand the content.
* It maintains a friendly tone.

Note that even one of these characteristics is enough to classify the response as following a chatbot-style format.

You must provide your evaluation in the following JSON format:
{
    "thought_process": str (Pay very close attention to the response and argue whether the response follows a chatbot-style or not in a few sentences),
    "is_chatbot_style": bool
}""",  # noqa: E501
            user_prompt=f"""**Model Response**:
{{{COMPLETION_KEY}}}""",
        ),
    }

    def __init__(
        self,
        grading_model: StructuredOutputChatModel,
        prompt_templates: Mapping[Language, PromptTemplate] = PROMPT_TEMPLATES,
    ) -> None:
        self._grading_model = grading_model

        if not all(self.COMPLETION_KEY in prompt_template.user_prompt for prompt_template in prompt_templates.values()):
            raise ValueError(f"At least one PromptTemplate is invalid, must contain '{self.COMPLETION_KEY}'.")
        self._prompt_templates = prompt_templates

    def grade(self, completion: str, language: Language) -> ChatbotStyleGradingOutput:
        try:
            prompt_template = language.language_config(self._prompt_templates)
        except Exception as _:
            prompt_template = Language("en").language_config(self._prompt_templates)

        messages = prompt_template.to_messages(
            [],
            [
                (self.COMPLETION_KEY, completion),
            ],
        )
        raw_completion = self._grading_model.generate_from_messages([messages])[0]
        loaded_json = parse_json_output(raw_completion.completion)

        return ChatbotStyleGradingOutput(
            thought_process=loaded_json.get("thought_process", None),
            is_chatbot_style=loaded_json.get("is_chatbot_style", None),
            judge_prompt=raw_completion.prompt,
            judge_response=raw_completion.completion,
        )
