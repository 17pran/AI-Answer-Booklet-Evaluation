import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def retrieve_relevant_chunks(
    query,
    chunks,
    embedder,
    top_k=5
):

    if not chunks:
        return []

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    chunk_embeddings = embedder.embed(
        texts
    )

    query_embedding = embedder.embed_query(
        query
    )

    similarities = cosine_similarity(
        [query_embedding],
        chunk_embeddings
    )[0]

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        result = chunks[index].copy()

        result["similarity"] = float(
            similarities[index]
        )

        results.append(result)

    return results
