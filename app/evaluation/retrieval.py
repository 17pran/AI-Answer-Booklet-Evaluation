from sqlalchemy import text

from app.database import engine
from app.textbook.processor import get_embedder


def retrieve_relevant_chunks(query, textbook_id, top_k=5):
    if not query or not str(query).strip():
        return []

    if textbook_id is None:
        raise ValueError("A textbook_id is required.")

    embedder = get_embedder()
    query_embedding = embedder.embed_query(str(query))

    vector_string = "[" + ",".join(
        str(float(value)) for value in query_embedding
    ) + "]"

    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT
                    id,
                    textbook_id,
                    chunk_index,
                    text,
                    chapter,
                    start_page,
                    end_page,
                    1 - (
                        embedding <=> CAST(:embedding AS vector)
                    ) AS similarity
                FROM textbook_chunks
                WHERE textbook_id = :textbook_id
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :top_k
            """),
            {
                "embedding": vector_string,
                "textbook_id": int(textbook_id),
                "top_k": max(1, int(top_k))
            }
        ).mappings().all()

    return [
        {
            "id": row["id"],
            "textbook_id": row["textbook_id"],
            "chunk_index": row["chunk_index"],
            "text": row["text"],
            "chapter": row["chapter"],
            "start_page": row["start_page"],
            "end_page": row["end_page"],
            "similarity": float(row["similarity"])
        }
        for row in rows
    ]
