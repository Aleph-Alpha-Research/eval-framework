import numpy as np


def count_bytes(text: str, /, *, encoding: str = "utf-8") -> int:
    """Count the number of bytes in a string."""
    return len(bytes(text.encode(encoding)))


def cosine_similarity(embedding_a: list[float], embedding_b: list[float]) -> float:
    """Computes the cosine similarity between two embeddings."""
    return pairwise_cosine_similarity([embedding_a], [embedding_b])[0][0]


def pairwise_cosine_similarity(embeddings_a: list[list[float]], embeddings_b: list[list[float]]) -> list[list[float]]:
    """
    Computes the pairwise cosine similarity matrix between two lists of embeddings.
    Output[i][j] is the cosine similarity between embeddings_a[i] and embeddings_b[j].
    """
    # M, D = len(embeddings_a), len(embeddings_a[0])
    # N, D = len(embeddings_b), len(embeddings_b[0])
    A = np.array(embeddings_a)  # shape (M, D)
    B = np.array(embeddings_b)  # shape (N, D)

    norms_a = np.linalg.norm(A, axis=1, keepdims=True)
    norms_b = np.linalg.norm(B, axis=1, keepdims=True)

    epsilon = 1e-9
    A_normalized = A / (norms_a + epsilon)
    B_normalized = B / (norms_b + epsilon)

    similarity_matrix = A_normalized @ B_normalized.T  # (M, D) @ (D, N) -> (M, N)
    return similarity_matrix.tolist()
