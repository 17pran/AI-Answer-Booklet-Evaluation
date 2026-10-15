import hashlib
from pathlib import Path

from sqlalchemy import select

from app.database import SessionLocal
from app.models import Textbook, TextbookChunk

from app.textbook.pdf_parser import extract_text
from app.textbook.chunker import chunk_text
from app.textbook.embeddings import TextbookEmbedder


def calculate_file_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a textbook PDF."""

    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(block)

    return sha256.hexdigest()


def process_textbook(file_path: Path, filename: str):
    """
    Process a textbook once and store its chunks
    and embeddings in PostgreSQL.
    """

    file_hash = calculate_file_hash(file_path)

    with SessionLocal() as db:

        # Check whether this exact PDF was processed previously.
        existing = db.scalar(
            select(Textbook).where(
                Textbook.file_hash == file_hash,
                Textbook.status == "processed",
            )
        )

        if existing:
            return {
                "success": True,
                "textbook_id": existing.id,
                "filename": existing.filename,
                "chunks": len(existing.chunks),
                "already_processed": True,
                "message": "Textbook already processed. Reusing stored data.",
            }

        # Extract text using your existing PDF parser.
        pages = extract_text(str(file_path))

        if not pages:
            raise ValueError("No readable text found in the textbook.")

        # Split text using your existing chunker.
        chunks = chunk_text(pages)

        if not chunks:
            raise ValueError("No textbook chunks were generated.")

        # Generate embeddings once.
        embedder = TextbookEmbedder()
        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed(texts)

        # Save textbook and all chunks in one transaction.
        textbook = Textbook(
            filename=filename,
            file_hash=file_hash,
            status="processed",
        )

        db.add(textbook)
        db.flush()

        for index, chunk in enumerate(chunks):
            db.add(
                TextbookChunk(
                    textbook_id=textbook.id,
                    chunk_index=index,
                    text=chunk["text"],
                    embedding=embeddings[index].tolist(),
                )
            )

        db.commit()
        db.refresh(textbook)

        return {
            "success": True,
            "textbook_id": textbook.id,
            "filename": filename,
            "chunks": len(chunks),
            "already_processed": False,
            "message": "Textbook processed and stored successfully.",
        }
