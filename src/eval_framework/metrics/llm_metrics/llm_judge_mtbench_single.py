import re

from pydantic import BaseModel

from eval_framework.logger import logger
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm_metrics.base import BaseLLMJudgeMetric
from eval_framework.shared.types import BaseMetricContext, Completion, extract_context_metric
from template_formatting.formatter import Message, Role

SINGLE_JUDGE_PROMPTS = {
    "single_assistant_single_turn": {
        "prompt_template": '[System]\nYou are a helpful assistant.\n\n[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n[Question]\n{question}\n\n[The Start of Assistant\'s Answer]\n{answer}\n[The End of Assistant\'s Answer]'  # noqa: E501
    },
    "single_assistant_multi_turn": {
        "prompt_template": 'Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You evaluation should focus on the assistant\'s answer to the second user question. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".\n\n<|The Start of Assistant A\'s Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A\'s Conversation with User|>'  # noqa: E501
    },
    "single_assistant_single_turn_w_reference": {
        "prompt_template": "[System]\nYou are a helpful assistant.\n\n[Instruction]\nPlease act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n[Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The End of Assistant's Answer]"  # noqa: E501
    },
    "single_assistant_multi_turn_w_reference": {
        "prompt_template": "Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. Your evaluation should consider correctness and helpfulness. You will be given a reference answer and the assistant's answer. You evaluation should focus on the assistant's answer to the second question. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[5]]\".\n\n<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_2}\n\n<|The End of Assistant A's Conversation with User|>"  # noqa: E501
    },
}

SINGLE_JUDGE_PROMPTS_DE = {
    "single_assistant_single_turn": {
        "prompt_template": '[System]\nDu bist ein hilfreicher Assistent.\n\n[Anweisung]\nBitte agieren Sie als unparteiischer Richter und bewerten Sie die Qualität der Antwort, die von einem KI-Assistenten auf die unten angezeigte Nutzerfrage gegeben wurde. Ihre Bewertung sollte Faktoren wie Nützlichkeit, Relevanz, Genauigkeit, Tiefe, Kreativität und Detailliertheit der Antwort berücksichtigen. Beginnen Sie Ihre Bewertung mit einer kurzen Erklärung. Seien Sie so objektiv wie möglich. Nachdem Sie Ihre Erklärung gegeben haben, müssen Sie die Antwort auf einer Skala von 1 bis 10 bewerten und dabei streng dieses Format einhalten: "[[rating]]", zum Beispiel: "Bewertung: [[5]]".\n\n[Frage]\n{question}\n\n[Der Anfang der Assistentenantwort]\n{answer}\n[Das Ende der Assistentenantwort]'  # noqa: E501
    },
    "single_assistant_multi_turn": {
        "prompt_template": 'Bitte agieren Sie als unparteiischer Richter und bewerten Sie die Qualität der Antwort, die von einem KI-Assistenten auf die unten angezeigte Nutzerfrage gegeben wurde. Ihre Bewertung sollte Faktoren wie Nützlichkeit, Relevanz, Genauigkeit, Tiefe, Kreativität und Detailliertheit der Antwort berücksichtigen. Ihre Bewertung sollte sich auf die Antwort des Assistenten auf die zweite Nutzerfrage konzentrieren. Beginnen Sie Ihre Bewertung mit einer kurzen Erklärung. Seien Sie so objektiv wie möglich. Nachdem Sie Ihre Erklärung gegeben haben, müssen Sie die Antwort auf einer Skala von 1 bis 10 bewerten, wobei Sie streng dieses Format einhalten: "[[rating]]", zum Beispiel: "Bewertung: [[5]]".\n\n<|Der Anfang von Assistent A\'s Unterhaltung mit dem Nutzer|>\n\n### Nutzer:\n{question_1}\n\n### Assistent A:\n{answer_1}\n\n### Nutzer:\n{question_2}\n\n### Assistent A:\n{answer_2}\n\n<|Das Ende von Assistent A\'s Unterhaltung mit dem Nutzer|>'  # noqa: E501
    },
    "single_assistant_single_turn_w_reference": {
        "prompt_template": '[System]\nDu bist ein hilfreicher Assistent.\n\n[Anweisung]\nBitte agieren Sie als unparteiischer Richter und bewerten Sie die Qualität der Antwort, die von einem KI-Assistenten auf die unten angezeigte Nutzerfrage gegeben wurde. Ihre Bewertung sollte Korrektheit und Nützlichkeit berücksichtigen. Ihnen wird eine Referenzantwort und die Antwort des Assistenten gegeben. Beginnen Sie Ihre Bewertung, indem Sie die Antwort des Assistenten mit der Referenzantwort vergleichen. Identifizieren Sie und korrigieren Sie etwaige Fehler. Seien Sie so objektiv wie möglich. Nachdem Sie Ihre Erklärung gegeben haben, müssen Sie die Antwort auf einer Skala von 1 bis 10 bewerten und dabei streng dieses Format einhalten: "[[rating]]", zum Beispiel: "Bewertung: [[5]]".\n\n[Frage]\n{question}\n\n[Der Anfang der Referenzantwort]\n{ref_answer_1}\n[Das Ende der Referenzantwort]\n\n[Der Anfang der Assistentenantwort]\n{answer}\n[Das Ende der Assistentenantwort]'  # noqa: E501
    },
    "single_assistant_multi_turn_w_reference": {
        "propmt_templte": 'Bitte agieren Sie als unparteiischer Richter und bewerten Sie die Qualität der Antwort, die von einem KI-Assistenten auf die Nutzerfrage gegeben wurde. Ihre Bewertung sollte Korrektheit und Nützlichkeit berücksichtigen. Ihnen wird eine Referenzantwort und die Antwort des Assistenten gegeben. Ihre Bewertung sollte sich auf die Antwort des Assistenten auf die zweite Frage konzentrieren. Beginnen Sie Ihre Bewertung, indem Sie die Antwort des Assistenten mit der Referenzantwort vergleichen. Identifizieren und korrigieren Sie etwaige Fehler. Seien Sie so objektiv wie möglich. Nachdem Sie Ihre Erklärung gegeben haben, müssen Sie die Antwort auf einer Skala von 1 bis 10 bewerten, wobei Sie streng dieses Format einhalten: "[[rating]]", zum Beispiel: "Bewertung: [[5]]".\n\n<|Der Anfang der Referenzantwort|>\n\n### Nutzer:\n{question_1}\n\n### Referenzantwort:\n{ref_answer_1}\n\n### Nutzer:\n{question_2}\n\n### Referenzantwort:\n{ref_answer_2}\n\n<|Das Ende der Referenzantwort|>\n\n\n<|Der Anfang von Assistent A\'s Unterhaltung mit dem Nutzer|>\n\n### Nutzer:\n{question_1}\n\n### Assistent A:\n{answer_1}\n\n### Nutzer:\n{question_2}\n\n### Assistent A:\n{answer_2}\n\n<|Das Ende von Assistent A\'s Unterhaltung mit dem Nutzer|>'  # noqa: E501
    },
}


SINGLE_JUDGE_PROMPTS_FI = {
    "single_assistant_single_turn": {
        "prompt_template": '[Järjestelmä]\nOlet avulias avustaja.\n\n[Ohje]\nToimi puolueettomana tuomarina ja arvioi AI-avustajan antaman vastauksen laatua käyttäjän kysymykseen, joka näkyy alla. Arviosi tulisi ottaa huomioon tekijät kuten hyödyllisyys, asiaankuuluvuus, tarkkuus, syvällisyys, luovuus ja yksityiskohtien taso. Aloita arviointisi antamalla lyhyt selitys. Ole mahdollisimman objektiivinen. Selityksen jälkeen sinun on arvioitava vastaus asteikolla 1–10 noudattamalla tarkasti tätä muotoa: "[[arvosana]]", esimerkiksi: "Arvosana: [[5]]".\n\n[Kysymys]\n{question}\n\n[Avustajan vastauksen alku]\n{answer}\n[Avustajan vastauksen loppu]'  # noqa: E501
    },
    "single_assistant_multi_turn": {
        "prompt_template": 'Toimi puolueettomana tuomarina ja arvioi AI-avustajan antaman vastauksen laatua käyttäjän kysymykseen, joka näkyy alla. Arviosi tulisi ottaa huomioon tekijät kuten hyödyllisyys, asiaankuuluvuus, tarkkuus, syvällisyys, luovuus ja yksityiskohtien taso. Arviosi tulisi keskittyä avustajan vastaukseen toiseen käyttäjän kysymykseen. Aloita arviointisi antamalla lyhyt selitys. Ole mahdollisimman objektiivinen. Selityksen jälkeen sinun on arvioitava vastaus asteikolla 1–10 noudattamalla tarkasti tätä muotoa: "[[arvosana]]", esimerkiksi: "Arvosana: [[5]]".\n\n<|Avustaja A:n keskustelun alku käyttäjän kanssa|>\n\n### Käyttäjä:\n{question_1}\n\n### Avustaja A:\n{answer_1}\n\n### Käyttäjä:\n{question_2}\n\n### Avustaja A:\n{answer_2}\n\n<|Avustaja A:n keskustelun loppu käyttäjän kanssa|>'  # noqa: E501
    },
    "single_assistant_single_turn_w_reference": {
        "prompt_template": '[Järjestelmä]\nOlet avulias avustaja.\n\n[Ohje]\nToimi puolueettomana tuomarina ja arvioi AI-avustajan antaman vastauksen laatua käyttäjän kysymykseen, joka näkyy alla. Arviosi tulisi ottaa huomioon oikeellisuus ja hyödyllisyys. Sinulle annetaan viitevastaus ja avustajan vastaus. Aloita arviointisi vertaamalla avustajan vastausta viitevastaukseen. Tunnista ja korjaa mahdolliset virheet. Ole mahdollisimman objektiivinen. Selityksen jälkeen sinun on arvioitava vastaus asteikolla 1–10 noudattamalla tarkasti tätä muotoa: "[[arvosana]]", esimerkiksi: "Arvosana: [[5]]".\n\n[Kysymys]\n{question}\n\n[Viitevastauksen alku]\n{ref_answer_1}\n[Viitevastauksen loppu]\n\n[Avustajan vastauksen alku]\n{answer}\n[Avustajan vastauksen loppu]'  # noqa: E501
    },
    "single_assistant_multi_turn_w_reference": {
        "prompt_template": 'Toimi puolueettomana tuomarina ja arvioi AI-avustajan antaman vastauksen laatua käyttäjän kysymykseen. Arviosi tulisi ottaa huomioon oikeellisuus ja hyödyllisyys. Sinulle annetaan viitevastaus ja avustajan vastaus. Arviosi tulisi keskittyä avustajan vastaukseen toiseen kysymykseen. Aloita arviointisi vertaamalla avustajan vastausta viitevastaukseen. Tunnista ja korjaa mahdolliset virheet. Ole mahdollisimman objektiivinen. Selityksen jälkeen sinun on arvioitava vastaus asteikolla 1–10 noudattamalla tarkasti tätä muotoa: "[[arvosana]]", esimerkiksi: "Arvosana: [[5]]".\n\n<|Viitevastauksen alku|>\n\n### Käyttäjä:\n{question_1}\n\n### Viitevastaus:\n{ref_answer_1}\n\n### Käyttäjä:\n{question_2}\n\n### Viitevastaus:\n{ref_answer_2}\n\n<|Viitevastauksen loppu|>\n\n\n<|Avustaja A:n keskustelun alku käyttäjän kanssa|>\n\n### Käyttäjä:\n{question_1}\n\n### Avustaja A:\n{answer_1}\n\n### Käyttäjä:\n{question_2}\n\n### Avustaja A:\n{answer_2}\n\n<|Avustaja A:n keskustelun loppu käyttäjän kanssa|>'  # noqa: E501
    },
}

NEED_REF_CATEGORIES = ["math", "reasoning", "coding", "arena-hard-200"]


class PromptToJudge(BaseModel):
    comparison_type: str
    prompt_text: str


class MTBenchJudgeSingleMetricContext(BaseMetricContext):
    category: str
    reference: list[str] | str | None


def generate_single_judge_prompts(response: Completion) -> list[PromptToJudge]:
    context = extract_context_metric(response, MTBenchJudgeSingleMetricContext)

    assert response.messages is not None

    if response.subject.startswith("de"):
        prompt_templates = SINGLE_JUDGE_PROMPTS_DE
    elif response.subject.startswith("fi"):
        prompt_templates = SINGLE_JUDGE_PROMPTS_FI
    else:
        prompt_templates = SINGLE_JUDGE_PROMPTS
    prompts_to_judge = []

    assert context.category is not None, "Category must be provided in the context for MTBenchJudgeSingleMetricContext"

    # No reference answer needed
    if context.category not in NEED_REF_CATEGORIES:
        # SINLGE TURN
        if len(response.messages) <= 2:
            # turn 1
            question = response.last_user_instruction
            answer = response.completion
            # format prompt
            single_turn_prompt = prompt_templates["single_assistant_single_turn"]["prompt_template"].format(
                question=question,
                answer=answer,
            )
            prompts_to_judge.append(PromptToJudge(comparison_type="single_judgement", prompt_text=single_turn_prompt))
            # MULTI TURN
        else:
            # turn 1
            question_1 = response.first_user_instruction
            answer_1 = response.messages[1].content
            # turn 2
            question_2 = response.last_user_instruction
            answer_2 = response.completion
            # format prompt
            multi_turn_prompt = prompt_templates["single_assistant_multi_turn"]["prompt_template"].format(
                question_1=question_1, answer_1=answer_1, question_2=question_2, answer_2=answer_2
            )
            prompts_to_judge.append(PromptToJudge(comparison_type="single_judgement", prompt_text=multi_turn_prompt))
    # Reference answer needed
    elif context.reference:
        # SINGLE TURN
        if len(response.messages) <= 2 and len(context.reference) >= 1:
            # turn 1
            question = response.last_user_instruction
            answer = response.completion
            ref_answer = context.reference[0]
            # format prompt
            single_turn_prompt = prompt_templates["single_assistant_single_turn_w_reference"]["prompt_template"].format(
                question=question,
                answer=answer,
                ref_answer_1=ref_answer,
            )
            prompts_to_judge.append(PromptToJudge(comparison_type="single_judgement", prompt_text=single_turn_prompt))
        # MULTI TURN
        elif len(context.reference) >= 2:
            # turn 1
            question_1 = response.first_user_instruction
            answer_1 = response.messages[1].content
            ref_answer_1 = context.reference[0]
            # turn 2
            question_2 = response.last_user_instruction
            answer_2 = response.completion
            ref_answer_2 = context.reference[1]
            # format prompt
            multi_turn_prompt = prompt_templates["single_assistant_multi_turn_w_reference"]["prompt_template"].format(
                question_1=question_1,
                answer_1=answer_1,
                ref_answer_1=ref_answer_1,
                question_2=question_2,
                answer_2=answer_2,
                ref_answer_2=ref_answer_2,
            )
            prompts_to_judge.append(PromptToJudge(comparison_type="single_judgement", prompt_text=multi_turn_prompt))
    else:
        logger.info(
            f"Warning: No reference answer found for this sample (category: "
            f"{context.category}), even though it is needed."
        )

    return prompts_to_judge


class MTBenchJudgeSingle(BaseLLMJudgeMetric):
    NAME = "single_judgement"

    def calculate(self, response: Completion) -> list[MetricResult]:
        try:
            prompts_to_judge: list[PromptToJudge] = generate_single_judge_prompts(response)

            all_metrics = []
            for prompt_to_judge in prompts_to_judge:
                messages = [Message(role=Role.USER, content=prompt_to_judge.prompt_text)]
                output = self._llm_judge.generate_from_messages([messages])
                parsed_output = self._output_to_rating(output[0].completion)

                all_metrics.append(
                    MetricResult(
                        metric_name=prompt_to_judge.comparison_type,
                        value=parsed_output,
                        higher_is_better=True,
                    )
                )

            return all_metrics

        except KeyError as e:
            logger.info(f"LLM judge did not produce an expected output, sample will be ignored in aggregations. {e}")
            return []

    @staticmethod
    def _output_to_rating(output: str) -> float:
        match = re.search(r"\[\[(\d+)\]\]", output)

        if match:
            return float(match.group(1))
        return 0
