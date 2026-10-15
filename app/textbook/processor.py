import hashlib
from functools import lru_cache
from pathlib import Path

from sqlalchemy import text

from app.database import engine

from app.textbook.pdf_parser import extract_text
from app.textbook.chunker import chunk_text
from app.textbook.embeddings import TextbookEmbedder


# Load the embedding model only once per backend process.
@lru_cache(maxsize=1)
def get_embedder():
    return TextbookEmbedder()


def calculate_file_hash(pdf_path: str) -> str:
    """Calculate SHA-256 without loading the entire PDF into memory."""

    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def initialize_textbook_tables():
    """
    Add required metadata columns without recreating
    or deleting existing tables.
    """

    statements = [
        "CREATE EXTENSION IF NOT EXISTS vector",

        """
        ALTER TABLE textbooks
        ADD COLUMN IF NOT EXISTS file_hash TEXT
        """,

        """
        ALTER TABLE textbook_chunks
        ADD COLUMN IF NOT EXISTS chapter TEXT
        """,

        """
        ALTER TABLE textbook_chunks
        ADD COLUMN IF NOT EXISTS start_page INTEGER
        """,

        """
        ALTER TABLE textbook_chunks
        ADD COLUMN IF NOT EXISTS end_page INTEGER
        """,

        """
        CREATE INDEX IF NOT EXISTS idx_textbook_chunks_textbook_id
        ON textbook_chunks(textbook_id)
        """,

        """
        CREATE INDEX IF NOT EXISTS idx_textbook_chunks_position
        ON textbook_chunks(textbook_id, chunk_index)
        """,

        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_textbooks_file_hash
        ON textbooks(file_hash)
        WHERE file_hash IS NOT NULL
        """
    ]

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))


def process_textbook(
    pdf_path: str,
    filename: str | None = None
) -> dict:
    """
    Process a textbook once and save its chunks and embeddings.

    An identical PDF reuses its existing database record.
    """

    pdf_path = str(Path(pdf_path).resolve())
    pdf_file = Path(pdf_path)

    if not pdf_file.is_file():
        raise FileNotFoundError(
            f"Textbook PDF not found: {pdf_path}"
        )

    filename = filename or pdf_file.name

    # --------------------------------------------------
    # 1. Calculate the PDF hash.
    # --------------------------------------------------

    file_hash = calculate_file_hash(pdf_path)

    # --------------------------------------------------
    # 2. Check whether this PDF is already processed.
    # --------------------------------------------------

    with engine.connect() as connection:
        existing = connection.execute(
            text("""
                SELECT
                    t.id,
                    t.filename,
                    COUNT(c.id) AS chunk_count
                FROM textbooks t
                LEFT JOIN textbook_chunks c
                    ON c.textbook_id = t.id
                WHERE t.file_hash = :file_hash
                GROUP BY t.id, t.filename
            """),
            {"file_hash": file_hash}
        ).mappings().first()

    if existing and existing["chunk_count"] > 0:
        print("Textbook already exists. Reusing saved embeddings.")

        return {
            "textbook_id": existing["id"],
            "filename": existing["filename"],
            "chunk_count": existing["chunk_count"],
            "reused": True
        }

    # --------------------------------------------------
    # 3. Extract PDF text.
    # --------------------------------------------------

    print(f"Extracting textbook: {filename}")

    pages = extract_text(pdf_path)

    if not pages:
        raise ValueError(
            "No extractable text found in the PDF. "
            "If this is a scanned PDF, OCR is required."
        )

    # --------------------------------------------------
    # 4. Create chapter-wise chunks.
    # --------------------------------------------------

    print("Creating chapter-wise chunks...")

    chunks = chunk_text(
        pages,
        chunk_size=1800,
        overlap=200
    )

    if not chunks:
        raise ValueError("No textbook chunks were created.")

    print(f"Created {len(chunks)} chunks.")

    # --------------------------------------------------
    # 5. Generate embeddings in batches.
    # --------------------------------------------------

    print("Generating embeddings...")

    embedder = get_embedder()

    chunk_texts = [chunk["text"] for chunk in chunks]

    embeddings = embedder.embed(
        chunk_texts,
        batch_size=64
    )

    if len(embeddings) != len(chunks):
        raise RuntimeError(
            "Embedding count does not match chunk count."
        )

    dimension = (
        embedder.model.get_sentence_embedding_dimension()
    )

    if dimension != 384:
        raise ValueError(
            f"Expected 384-dimensional embeddings, got {dimension}. "
            "Check the model and pgvector column."
        )

    # --------------------------------------------------
    # 6. Save the textbook and chunks transactionally.
    # --------------------------------------------------

    print("Saving textbook and embeddings to PostgreSQL...")

    with engine.begin() as connection:

        # Recheck before inserting to avoid ordinary duplicate
        # uploads creating multiple textbook records.
        existing = connection.execute(
            text("""
                SELECT id, filename
                FROM textbooks
                WHERE file_hash = :file_hash
                FOR UPDATE
            """),
            {"file_hash": file_hash}
        ).mappings().first()

        if existing:

            textbook_id = existing["id"]

            existing_count = connection.execute(
                text("""
                    SELECT COUNT(*)
                    FROM textbook_chunks
                    WHERE textbook_id = :textbook_id
                """),
                {"textbook_id": textbook_id}
            ).scalar_one()

            if existing_count > 0:
                return {
                    "textbook_id": textbook_id,
                    "filename": existing["filename"],
                    "chunk_count": existing_count,
                    "reused": True
                }

            # Retry processing a previously incomplete record.
            connection.execute(
                text("""
                    DELETE FROM textbook_chunks
                    WHERE textbook_id = :textbook_id
                """),
                {"textbook_id": textbook_id}
            )

            connection.execute(
                text("""
                    UPDATE textbooks
                    SET filename = :filename
                    WHERE id = :textbook_id
                """),
                {
                    "filename": filename,
                    "textbook_id": textbook_id
                }
            )

        else:

            # IMPORTANT:
            # file_hash is the existing database column.
            # Do not replace it with content_hash.
            textbook_id = connection.execute(
                text("""
                    INSERT INTO textbooks (filename, file_hash, status, created_at)
VALUES (:filename, :file_hash, :status, CURRENT_TIMESTAMP)
RETURNING id
                """),
                {
                    "filename": filename,
                    "file_hash": file_hash,
                    "status": "processing"
                }
            ).scalar_one()

        insert_sql = text("""
            INSERT INTO textbook_chunks (
                textbook_id,
                chunk_index,
                embedding,
                text,
                chapter,
                start_page,
                end_page
            )
            VALUES (
                :textbook_id,
                :chunk_index,
                CAST(:embedding AS vector),
                :chunk_text,
                :chapter,
                :start_page,
                :end_page
            )
        """)

        batch_size = 100
        rows = []

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings)
        ):

            vector_string = (
                "["
                + ",".join(str(float(value)) for value in embedding)
                + "]"
            )

            rows.append({
                "textbook_id": textbook_id,
                "chunk_index": index,
                "embedding": vector_string,
                "chunk_text": chunk["text"],
                "chapter": chunk.get("chapter", "Unknown"),
                "start_page": chunk.get("start_page"),
                "end_page": chunk.get("end_page")
            })

            if len(rows) >= batch_size:
                connection.execute(insert_sql, rows)
                rows.clear()

        if rows:
            connection.execute(insert_sql, rows)

    print(
        f"Textbook saved successfully: "
        f"id={textbook_id}, chunks={len(chunks)}"
    )

    return {
        "textbook_id": textbook_id,
        "filename": filename,
        "chunk_count": len(chunks),
        "reused": False
    }


def list_textbooks() -> list[dict]:
    """Return saved textbooks for the frontend."""

    with engine.connect() as connection:
        rows = connection.execute(
            text("""
                SELECT
                    t.id,
                    t.filename,
                    t.created_at,
                    COUNT(c.id) AS chunk_count
                FROM textbooks t
                LEFT JOIN textbook_chunks c
                    ON c.textbook_id = t.id
                GROUP BY
                    t.id,
                    t.filename,
                    t.created_at
                ORDER BY t.id DESC
            """)
        ).mappings().all()

    return [
        {
            "id": row["id"],
            "filename": row["filename"],
            "created_at": (
                row["created_at"].isoformat()
                if row["created_at"] else None
            ),
            "chunk_count": row["chunk_count"]
        }
        for row in rows
    ]
