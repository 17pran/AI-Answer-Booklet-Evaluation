import numpy as np

from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL


class TextbookEmbedder:

    def __init__(self):

        print(
            f"Loading embedding model: {EMBEDDING_MODEL}"
        )

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Embedding model loaded.")

    def embed(
        self,
        texts,
        batch_size=64
    ):
        """
        Generate normalized embeddings in batches.

        Returns:
            NumPy array of shape (number_of_texts, 384)
            when using all-MiniLM-L6-v2.
        """

        if not texts:
            return np.empty(
                (0, self.model.get_sentence_embedding_dimension()),
                dtype=np.float32
            )

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100
        )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    def embed_query(self, query):
        """
        Generate an embedding for a single question.
        """

        embedding = self.model.encode(
            [query],
            batch_size=1,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return np.asarray(
            embedding[0],
            dtype=np.float32
        )
