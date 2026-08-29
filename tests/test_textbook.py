from pathlib import Path

from app.textbook.pdf_parser import extract_text
from app.textbook.chunker import chunk_text
from app.textbook.embeddings import TextbookEmbedder


def main():

    # ---------------------------------------------------------
    # TEXTBOOK PDF
    # ---------------------------------------------------------

    pdf_path = Path("data/textbooks/model_answer.pdf")

    if not pdf_path.exists():
        print("\nERROR: Textbook PDF not found:")
        print(pdf_path)
        return

    print("\n==============================================")
    print("TEXTBOOK RAG PIPELINE TEST")
    print("==============================================")

    # ---------------------------------------------------------
    # 1. PDF → TEXT
    # ---------------------------------------------------------

    print("\n[1] Extracting textbook text...")

    pages = extract_text(str(pdf_path))

    print(f"Pages with text: {len(pages)}")

    if not pages:
        print("ERROR: No text extracted from PDF.")
        return

    print("\nFirst page preview:")
    print("----------------------------------------------")
    print(pages[0]["text"][:1000])

    # ---------------------------------------------------------
    # 2. TEXT → CHUNKS
    # ---------------------------------------------------------

    print("\n[2] Creating chunks...")

    chunks = chunk_text(
        pages,
        chunk_size=800,
        overlap=100
    )

    print(f"Total chunks: {len(chunks)}")

    print("\nFirst chunk:")
    print("----------------------------------------------")
    print(chunks[0]["text"])

    # ---------------------------------------------------------
    # 3. CHUNKS → EMBEDDINGS
    # ---------------------------------------------------------

    print("\n[3] Creating embeddings...")

    embedder = TextbookEmbedder()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.embed(texts)

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # ---------------------------------------------------------
    # 4. TEST QUERY EMBEDDING
    # ---------------------------------------------------------

    query = "What is software engineering?"

    query_embedding = embedder.embed_query(query)

    print(
        f"Query embedding shape: "
        f"{query_embedding.shape}"
    )

    print("\n==============================================")
    print("TEXTBOOK PIPELINE TEST SUCCESSFUL")
    print("==============================================")


if __name__ == "__main__":
    main()
