import pytest

from eval_framework.llm.openai import OpenAIEmbeddingModel
from eval_framework.utils.helpers import pairwise_cosine_similarity
from template_formatting.formatter import Message, Role


@pytest.mark.external_api
def test_openai_embedding_model():
    model = OpenAIEmbeddingModel()
    messages = [
        [Message(role=Role.USER, content="This is a test input for embedding generation.")],
        [Message(role=Role.USER, content="Different message to test embedding generation.")],
    ]

    embeddings = model.generate_from_messages(messages)
    assert len(embeddings) == len(messages)

    cosine_sims = pairwise_cosine_similarity(embeddings, embeddings)
    # assert self-simiarlity is 1
    for i in range(len(embeddings)):
        assert abs(cosine_sims[i][i] - 1.0) < 1e-5
    # assert different embeddings are less similar than self-similarity
    assert cosine_sims[0][1] < 1.0
    assert cosine_sims[1][0] < 1.0
