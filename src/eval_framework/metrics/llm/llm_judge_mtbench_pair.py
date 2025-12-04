import re

from pydantic import BaseModel

from eval_framework.logger import logger
from eval_framework.metrics.base import MetricResult
from eval_framework.metrics.llm.base import BaseLLMJudgeMetric
from eval_framework.shared.types import BaseMetricContext, Completion, extract_context_metric
from template_formatting.formatter import Message, Role

PAIR_JUDGE_PROMPTS = {
    "pair_assistant_single_turn": {
        "prompt_template": "[System]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n[User Question]\n{question}\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"  # noqa: E501
    },
    "pair_assistant_multi_turn": {
        "prompt_template": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. You should choose the assistant that follows the user's instructions and answers the user's questions better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. You should focus on who provides a better answer to the second user question. Begin your evaluation by comparing the responses of the two assistants and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>"  # noqa: E501
    },
    "pair_assistant_single_turn_w_reference": {
        "prompt_template": "[System]\nPlease act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given a reference answer, assistant A's answer, and assistant B's answer. Your job is to evaluate which assistant's answer is better. Begin your evaluation by comparing both assistants' answers with the reference answer. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n[User Question]\n{question}\n\n[The Start of Reference Answer]\n{ref_answer_1}\n[The End of Reference Answer]\n\n[The Start of Assistant A's Answer]\n{answer_a}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{answer_b}\n[The End of Assistant B's Answer]"  # noqa: E501
    },
    "pair_assistant_multi_turn_w_reference": {
        "prompt_template": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user questions. Your evaluation should consider correctness and helpfulness. You will be given reference answers, the assistant A's answers, the assistant B's answers. Your job is to determine which assistant provides correct and helpful answers to the second user question. Begin your evaluation by comparing both assistants' answers with the reference answers. Identify and correct any mistakes. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.\n\n<|The Start of Reference Answer|>\n\n### User:\n{question_1}\n\n### Reference answer:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Reference answer:\n{ref_answer_2}\n\n<|The End of Reference Answer|>\n\n\n<|The Start of Assistant A's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistant A:\n{answer_a_2}\n\n<|The End of Assistant A's Conversation with User|>\n\n\n<|The Start of Assistant B's Conversation with User|>\n\n### User:\n{question_1}\n\n### Assistant B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistant B:\n{answer_b_2}\n\n<|The End of Assistant B's Conversation with User|>"  # noqa: E501
    },
}

PAIR_JUDGE_PROMPTS_DE = {
    "pair_assistant_single_turn": {
        "prompt_template": '[System]\nBitte agieren Sie als unparteiischer Beurteiler und bewerten Sie die Qualität der Antworten, die von zwei KI-Assistenten auf die unten angezeigte Nutzerfrage gegeben wurden. Wählen Sie den Assistenten aus, der die Anweisungen des Nutzers besser befolgt und die Nutzerfrage besser beantwortet. Ihre Bewertung sollte Faktoren wie Nützlichkeit, Relevanz, Genauigkeit, Tiefe, Kreativität und Detaillierungsgrad der Antworten berücksichtigen. Beginnen Sie Ihre Bewertung mit einem Vergleich der beiden Antworten und geben Sie eine kurze Erklärung ab. Vermeiden Sie jeglichen Bias bezüglich der Position der Antworten und stellen Sie sicher, dass die Reihenfolge, in der die Antworten präsentiert wurden, Ihre Entscheidung nicht beeinflusst. Lassen Sie nicht zu, dass die Länge der Antworten Ihre Bewertung beeinflusst. Bevorzugen Sie keine bestimmten Namen der Assistenten. Seien Sie so objektiv wie möglich. Geben Sie nach Ihrer Erklärung Ihr endgültiges Urteil streng nach folgendem Format aus: "[[A]]" wenn Assistent A besser ist, "[[B]]" wenn Assistent B besser ist und "[[C]]" bei einem Unentschieden\n[Nutzerfrage]\n{question}\n\n[Der Anfang von Assistent A\'s Antwort]\n{answer_a}\n[Das Ende Assistent A\'s Antwort]\n\n[Der Anfang von Assistent B\'s Antwort]\n{answer_b}\n[Der Anfang von Assistent B\'s Antwort]'  # noqa: E501
    },
    "pair_assistant_multi_turn": {
        "prompt_template": 'Bitte agieren Sie als unparteiischer Beurteiler und bewerten Sie die Qualität der Antworten, die von zwei KI-Assistenten auf die Nutzerfragen gegeben wurden. Wählen Sie den Assistenten aus, der die Anweisungen des Nutzers besser befolgt und die Nutzerfragen besser beantwortet. Ihre Bewertung sollte Faktoren wie Nützlichkeit, Relevanz, Genauigkeit, Tiefe, Kreativität und Detaillgrad der Antworten berücksichtigen. Konzentrieren Sie sich darauf, wer die bessere Antwort auf die zweite Nutzerfrage liefert. Beginnen Sie Ihre Bewertung mit einem Vergleich der Antworten der beiden Assistenten und geben Sie eine kurze Erklärung ab. Vermeiden Sie jegliche Positionsvoreingenommenheit und stellen Sie sicher, dass die Reihenfolge, in der die Antworten präsentiert wurden, Ihre Entscheidung nicht beeinflusst. Lassen Sie nicht zu, dass die Länge der Antworten Ihre Bewertung beeinflusst. Bevorzugen Sie keine bestimmten Namen der Assistenten. Seien Sie so objektiv wie möglich. Geben Sie nach Ihrer Erklärung Ihr endgültiges Urteil streng nach folgendem Format aus: "[[A]]" wenn Assistent A besser ist, "[[B]]" wenn Assistent B besser ist und "[[C]]" bei einem Unentschieden.\n\n<|Der Anfang von Assistent A\'s Konversation mit dem User|>\n\n### User:\n{question_1}\n\n### Assistent A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistent A:\n{answer_a_2}\n\n<|Das Ende von Assistent A\'s Konversation mit dem User|>\n\n\n<|Der Anfang von Assistent B\'s Konversation mit der User|>\n\n### User:\n{question_1}\n\n### Assistent B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistent B:\n{answer_b_2}\n\n<|Das Ende von Assistent B\'s Konversation mit dem User|>'  # noqa: E501
    },
    "pair_assistant_single_turn_w_reference": {
        "prompt_template": '[System]\nBitte agieren Sie als unparteiischer Beurteiler und bewerten Sie die Qualität der Antworten, die von zwei KI-Assistenten auf die unten angezeigte Nutzerfrage gegeben wurden. Ihre Bewertung sollte Richtigkeit und Hilfreichkeit berücksichtigen. Sie erhalten eine Referenzantwort, die Antwort von Assistent A und die Antwort von Assistent B. Ihre Aufgabe ist es zu beurteilen, welche Antwort der Assistenten besser ist. Beginnen Sie Ihre Bewertung damit, die Antworten beider Assistenten mit der Referenzantwort zu vergleichen. Identifizieren und korrigieren Sie etwaige Fehler. Vermeiden Sie jegliche Positionsvoreingenommenheit und stellen Sie sicher, dass die Reihenfolge, in der die Antworten präsentiert wurden, Ihre Entscheidung nicht beeinflusst. Lassen Sie nicht zu, dass die Länge der Antworten Ihre Bewertung beeinflusst. Bevorzugen Sie keine bestimmten Namen der Assistenten. Seien Sie so objektiv wie möglich. Geben Sie nach Ihrer Erklärung Ihr endgültiges Urteil streng nach folgendem Format aus: "[[A]]" wenn Assistent A besser ist, "[[B]]" wenn Assistent B besser ist und "[[C]]" bei einem Unentschieden\n\n[Nutzerfrage]\n{question}\n\n[Der Anfang der Referenzantwort]\n{ref_answer_1}\n[Das Ender der Referenzantwort]\n\n[Der Anfang von Assistent A\'s Antwort]\n{answer_a}\n[Das Ende von Assistent A\'s Antwort]\n\n[Der Anfag von Assistent B\'s Answer]\n{answer_b}\n[Das Ende vin Assistent B\'s Antwort]'  # noqa: E501
    },
    "pair_assistant_multi_turn_w_reference": {
        "prompt_template": 'Bitte agieren Sie als unparteiischer Beurteiler und bewerten Sie die Qualität der Antworten, die von zwei KI-Assistenten auf die Nutzerfragen gegeben wurden. Ihre Bewertung sollte Richtigkeit und Hilfreichkeit berücksichtigen. Sie erhalten Referenzantworten, die Antworten von Assistent A und die Antworten von Assistent B. Ihre Aufgabe ist es zu ermitteln, welcher Assistent richtige und hilfreiche Antworten auf die zweite Nutzerfrage liefert. Beginnen Sie Ihre Bewertung damit, die Antworten beider Assistenten mit den Referenzantworten zu vergleichen. Identifizieren und korrigieren Sie etwaige Fehler. Vermeiden Sie jegliche Positionsvoreingenommenheit und stellen Sie sicher, dass die Reihenfolge, in der die Antworten präsentiert wurden, Ihre Entscheidung nicht beeinflusst. Lassen Sie nicht zu, dass die Länge der Antworten Ihre Bewertung beeinflusst. Bevorzugen Sie keine bestimmten Namen der Assistenten. Seien Sie so objektiv wie möglich. Geben Sie nach Ihrer Erklärung Ihr endgültiges Urteil streng nach folgendem Format aus: "[[A]]" wenn Assistent A besser ist, "[[B]]" wenn Assistent B besser ist und "[[C]]" bei einem Unentschieden.\n\n<|Der Anfang der Referenzantwort|>\n\n### User:\n{question_1}\n\n### Referenzantwort:\n{ref_answer_1}\n\n### User:\n{question_2}\n\n### Referenzantwort:\n{ref_answer_2}\n\n<|Das Ende der Referenzantwort|>\n\n\n<|Der Anfang von Assistant A\'s Konversation mit dem User|>\n\n### User:\n{question_1}\n\n### Assistent A:\n{answer_a_1}\n\n### User:\n{question_2}\n\n### Assistent A:\n{answer_a_2}\n\n<|Das Ende von Assistent A\'s Konversation mit dem User|>\n\n\n<|Der Anfang von Assistent B\'s Konversation mit dem User|>\n\n### User:\n{question_1}\n\n### Assistent B:\n{answer_b_1}\n\n### User:\n{question_2}\n\n### Assistent B:\n{answer_b_2}\n\n<|Das Ende von Assistent B\'s Konversation mit dem User|>'  # noqa: E501
    },
}

PAIR_JUDGE_PROMPTS_FI = {
    "pair_assistant_single_turn": {
        "prompt_template": '[Järjestelmä]\nToimi puolueettomana tuomarina ja arvioi kahden AI-avustajan antamien vastausten laatua alla näkyvään käyttäjän kysymykseen. Sinun tulee valita se avustaja, joka noudattaa käyttäjän ohjeita ja vastaa käyttäjän kysymykseen paremmin. Arviosi tulisi ottaa huomioon tekijät kuten hyödyllisyys, asiaankuuluvuus, tarkkuus, syvällisyys, luovuus ja yksityiskohtien taso. Aloita arviointisi vertaamalla kahta vastausta ja anna lyhyt selitys. Vältä mahdollisia asemointiharhoja ja varmista, että vastausten esitysjärjestys ei vaikuta päätökseesi. Älä anna vastausten pituuden vaikuttaa arvioosi. Älä suosi tiettyjä avustajien nimiä. Ole mahdollisimman objektiivinen. Selityksen jälkeen anna lopullinen päätöksesi noudattamalla tarkasti tätä muotoa: "[[A]]", jos avustaja A on parempi, "[[B]]", jos avustaja B on parempi, ja "[[C]]" tasapelin tapauksessa.\n\n[Käyttäjän kysymys]\n{question}\n\n[Avustaja A:n vastauksen alku]\n{answer_a}\n[Avustaja A:n vastauksen loppu]\n\n[Avustaja B:n vastauksen alku]\n{answer_b}\n[Avustaja B:n vastauksen loppu]'  # noqa: E501
    },
    "pair_assistant_multi_turn": {
        "prompt_template": 'Toimi puolueettomana tuomarina ja arvioi kahden AI-avustajan antamien vastausten laatua käyttäjän kysymyksiin. Sinun tulee valita se avustaja, joka noudattaa käyttäjän ohjeita ja vastaa käyttäjän kysymyksiin paremmin. Arviosi tulisi ottaa huomioon tekijät kuten hyödyllisyys, asiaankuuluvuus, tarkkuus, syvällisyys, luovuus ja yksityiskohtien taso. Arviosi tulisi keskittyä siihen, kuka antaa paremman vastauksen toiseen käyttäjän kysymykseen. Aloita arviointisi vertaamalla kahden avustajan vastauksia ja anna lyhyt selitys. Vältä mahdollisia asemointiharhoja ja varmista, että vastausten esitysjärjestys ei vaikuta päätökseesi. Älä anna vastausten pituuden vaikuttaa arvioosi. Älä suosi tiettyjä avustajien nimiä. Ole mahdollisimman objektiivinen. Selityksen jälkeen anna lopullinen päätöksesi noudattamalla tarkasti tätä muotoa: "[[A]]", jos avustaja A on parempi, "[[B]]", jos avustaja B on parempi, ja "[[C]]" tasapelin tapauksessa.\n\n<|Avustaja A:n keskustelun alku käyttäjän kanssa|>\n\n### Käyttäjä:\n{question_1}\n\n### Avustaja A:\n{answer_a_1}\n\n### Käyttäjä:\n{question_2}\n\n### Avustaja A:\n{answer_a_2}\n\n<|Avustaja A:n keskustelun loppu käyttäjän kanssa|>\n\n\n<|Avustaja B:n keskustelun alku käyttäjän kanssa|>\n\n### Käyttäjä:\n{question_1}\n\n### Avustaja B:\n{answer_b_1}\n\n### Käyttäjä:\n{question_2}\n\n### Avustaja B:\n{answer_b_2}\n\n<|Avustaja B:n keskustelun loppu käyttäjän kanssa|>'  # noqa: E501
    },
    "pair_assistant_single_turn_w_reference": {
        "prompt_template": '[Järjestelmä]\nToimi puolueettomana tuomarina ja arvioi kahden AI-avustajan antamien vastausten laatua alla näkyvään käyttäjän kysymykseen. Arviosi tulisi ottaa huomioon oikeellisuus ja hyödyllisyys. Sinulle annetaan viitevastaus, avustajan A vastaus ja avustajan B vastaus. Tehtäväsi on arvioida, kumpi avustaja antoi paremman vastauksen. Aloita arviointisi vertaamalla molempien avustajien vastauksia viitevastaukseen. Tunnista ja korjaa mahdolliset virheet. Vältä mahdollisia asemointiharhoja ja varmista, että vastausten esitysjärjestys ei vaikuta päätökseesi. Älä anna vastausten pituuden vaikuttaa arvioosi. Älä suosi tiettyjä avustajien nimiä. Ole mahdollisimman objektiivinen. Selityksen jälkeen anna lopullinen päätöksesi noudattamalla tarkasti tätä muotoa: "[[A]]", jos avustaja A on parempi, "[[B]]", jos avustaja B on parempi, ja "[[C]]" tasapelin tapauksessa.\n\n[Käyttäjän kysymys]\n{question}\n\n[Viitevastauksen alku]\n{ref_answer_1}\n[Viitevastauksen loppu]\n\n[Avustaja A:n vastauksen alku]\n{answer_a}\n[Avustaja A:n vastauksen loppu]\n\n[Avustaja B:n vastauksen alku]\n{answer_b}\n[Avustaja B:n vastauksen loppu]'  # noqa: E501
    },
    "pair_assistant_multi_turn_w_reference": {
        "prompt_template": 'Toimi puolueettomana tuomarina ja arvioi kahden AI-avustajan antamien vastausten laatua käyttäjän kysymyksiin. Arviosi tulisi ottaa huomioon oikeellisuus ja hyödyllisyys. Sinulle annetaan viitevastaukset, avustajan A vastaukset ja avustajan B vastaukset. Tehtäväsi on määrittää, kumpi avustaja antoi oikeat ja hyödylliset vastaukset toiseen käyttäjän kysymykseen. Aloita arviointisi vertaamalla molempien avustajien vastauksia viitevastauksiin. Tunnista ja korjaa mahdolliset virheet. Vältä mahdollisia asemointiharhoja ja varmista, että vastausten esitysjärjestys ei vaikuta päätökseesi. Älä anna vastausten pituuden vaikuttaa arvioosi. Älä suosi tiettyjä avustajien nimiä. Ole mahdollisimman objektiivinen. Selityksen jälkeen anna lopullinen päätöksesi noudattamalla tarkasti tätä muotoa: "[[A]]", jos avustaja A on parempi, "[[B]]", jos avustaja B on parempi, ja "[[C]]" tasapelin tapauksessa.\n\n<|Viitevastauksen alku|>\n\n### Käyttäjä:\n{question_1}\n\n### Viitevastaus:\n{ref_answer_1}\n\n### Käyttäjä:\n{question_2}\n\n### Viitevastaus:\n{ref_answer_2}\n\n<|Viitevastauksen loppu|>\n\n\n<|Avustaja A:n keskustelun alku käyttäjän kanssa|>\n\n### Käyttäjä:\n{question_1}\n\n### Avustaja A:\n{answer_a_1}\n\n### Käyttäjä:\n{question_2}\n\n### Avustaja A:\n{answer_a_2}\n\n<|Avustaja A:n keskustelun loppu käyttäjän kanssa|>\n\n\n<|Avustaja B:n keskustelun alku käyttäjän kanssa|>\n\n### Käyttäjä:\n{question_1}\n\n### Avustaja B:\n{answer_b_1}\n\n### Käyttäjä:\n{question_2}\n\n### Avustaja B:\n{answer_b_2}\n\n<|Avustaja B:n keskustelun loppu käyttäjän kanssa|>'  # noqa: E501
    },
}


NEED_REF_CATEGORIES = ["math", "reasoning", "coding", "arena-hard-200"]
PAIR_JUDGE_PROMPTS_LIST = [
    PAIR_JUDGE_PROMPTS,
    PAIR_JUDGE_PROMPTS_DE,
    PAIR_JUDGE_PROMPTS_FI,
]


class PromptToJudge(BaseModel):
    comparison_type: str
    prompt_text: str


class MTBenchJudgePairMetricContext(BaseMetricContext):
    category: str
    answer: list[str] | str
    reference: list[str] | str | None


def generate_pair_judge_prompts(response: Completion) -> list[PromptToJudge]:
    context = extract_context_metric(response, MTBenchJudgePairMetricContext)
    assert response.messages is not None

    if response.subject.startswith("de"):
        prompt_templates = PAIR_JUDGE_PROMPTS_DE
    elif response.subject.startswith("fi"):
        prompt_templates = PAIR_JUDGE_PROMPTS_FI
    else:
        prompt_templates = PAIR_JUDGE_PROMPTS
    prompts_to_judge = []

    context = extract_context_metric(response, MTBenchJudgePairMetricContext)

    assert context.category is not None, "Category must be provided in the context for MTBenchJudgePairMetricContext"
    assert context.answer is not None, "Answer must be provided in the context for MTBenchJudgePairMetricContext"

    # No reference answer needed
    if context.category not in NEED_REF_CATEGORIES:
        # SINLGE TURN
        if len(response.messages) <= 2:
            # turn 1
            question = response.last_user_instruction
            answer_a = response.completion
            answer_b = context.answer[0]
            # format prompt
            single_turn_prompt = prompt_templates["pair_assistant_single_turn"]["prompt_template"].format(
                question=question, answer_a=answer_a, answer_b=answer_b
            )
            prompts_to_judge.append(PromptToJudge(comparison_type="pairwise_judgement", prompt_text=single_turn_prompt))

        # MULTI TURN
        else:
            # turn 1
            question_1 = response.first_user_instruction
            answer_a_1 = response.messages[1].content
            answer_b_1 = context.answer[0]
            # turn 2
            question_2 = response.last_user_instruction
            answer_a_2 = response.completion
            answer_b_2 = context.answer[1]
            # format prompt
            multi_turn_prompt = prompt_templates["pair_assistant_multi_turn"]["prompt_template"].format(
                question_1=question_1,
                answer_a_1=answer_a_1,
                answer_b_1=answer_b_1,
                question_2=question_2,
                answer_a_2=answer_a_2,
                answer_b_2=answer_b_2,
            )
            prompts_to_judge.append(PromptToJudge(comparison_type="pairwise_judgement", prompt_text=multi_turn_prompt))
    # Reference answer needed
    elif context.reference:
        # SINGLE TURN
        if len(response.messages) <= 2 and len(context.reference) >= 1:
            # turn 1
            question = response.last_user_instruction
            answer_a = response.completion
            answer_b = context.answer[0]
            ref_answer_1 = context.reference[0]
            # format prompt
            single_turn_prompt = prompt_templates["pair_assistant_single_turn_w_reference"]["prompt_template"].format(
                question=question, answer_a=answer_a, answer_b=answer_b, ref_answer_1=ref_answer_1
            )
            prompts_to_judge.append(PromptToJudge(comparison_type="pairwise_judgement", prompt_text=single_turn_prompt))
        # MULTI TURN
        elif len(context.reference) >= 2:
            # turn 1
            question_1 = response.first_user_instruction
            answer_a_1 = response.messages[1].content
            answer_b_1 = context.answer[0]
            ref_answer_1 = context.reference[0]
            # turn 2
            question_2 = response.last_user_instruction
            answer_a_2 = response.completion
            answer_b_2 = context.answer[1]
            ref_answer_2 = context.reference[1]
            # format prompt
            multi_turn_prompt = prompt_templates["pair_assistant_multi_turn_w_reference"]["prompt_template"].format(
                question_1=question_1,
                answer_a_1=answer_a_1,
                answer_b_1=answer_b_1,
                ref_answer_1=ref_answer_1,
                question_2=question_2,
                answer_a_2=answer_a_2,
                answer_b_2=answer_b_2,
                ref_answer_2=ref_answer_2,
            )
            prompts_to_judge.append(PromptToJudge(comparison_type="pairwise_judgement", prompt_text=multi_turn_prompt))
    else:
        logger.info(
            f"Warning: No reference answer found for this sample (category: "
            f"{context.category}), even though it is needed."
        )

    return prompts_to_judge


class MTBenchJudgePair(BaseLLMJudgeMetric):
    NAME = "pairwise_judgement"

    def calculate(self, response: Completion) -> list[MetricResult]:
        prompts_to_judge: list[PromptToJudge] = generate_pair_judge_prompts(response)

        all_metrics = []
        for prompt_to_judge in prompts_to_judge:
            messages = [Message(role=Role.USER, content=prompt_to_judge.prompt_text)]
            all_metrics.append(self._evaluate_prompt(prompt_to_judge, messages))

        return all_metrics

    def _evaluate_prompt(self, prompt_to_judge: PromptToJudge, messages: list[Message]) -> MetricResult:
        try:
            output = self._llm_judge.generate_from_messages([messages])
            parsed_output = self._output_to_rating(output[0].completion)
            return self._create_metric_result(
                metric_name=prompt_to_judge.comparison_type, value=parsed_output, higher_is_better=True
            )
        except Exception as e:
            logger.info(f"LLM judge failed to generate output for prompt: {prompt_to_judge.prompt_text}. Error: {e}")
            return self._create_metric_result(
                metric_name=prompt_to_judge.comparison_type, value=None, higher_is_better=True, error=e
            )

    @staticmethod
    def _output_to_rating(output: str) -> float:
        match = re.search(r"\[\[(.*?)\]\]", output)
        # A = Win, B = Lose, C = Tie
        letters = {"A": 1, "B": 0, "C": 0.5}
        if match:
            value = match.group(1)
            if value in letters:
                return letters[value]
        return 0.5  # Tie
