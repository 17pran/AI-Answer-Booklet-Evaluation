import numpy as np

from sentence_transformers import (
    SentenceTransformer
)

from app.config import EMBEDDING_MODEL


class AnswerKeyEmbedder:

    def __init__(self):

        print(
            f"Loading embedding model: "
            f"{EMBEDDING_MODEL}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print(
            "Answer-key embedding model loaded."
        )

    def embed(self, texts):

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True
        )

        return np.array(
            embeddings
        )

    def embed_query(self, query):

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )

        return embedding[0]
