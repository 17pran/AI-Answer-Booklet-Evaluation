from pathlib import Path

from app.config import TEXTBOOKS_DIR

from app.textbook.pdf_parser import extract_text
from app.textbook.chunker import chunk_text
from app.textbook.embeddings import TextbookEmbedder

from app.evaluation.retrieval import (
    retrieve_relevant_chunks
)


def main():

    print("\n==============================================")
    print("RAG RETRIEVAL TEST")
    print("==============================================")

    # ========================================================
    # 1. FIND TEXTBOOK PDF AUTOMATICALLY
    # ========================================================

    pdf_files = sorted(
        TEXTBOOKS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print(
            "\nERROR: No textbook PDF found in:"
        )

        print(
            TEXTBOOKS_DIR
        )

        print(
            "\nPlease place a textbook PDF inside "
            "the textbooks folder."
        )

        return

    # Use the first PDF found
    textbook_path = pdf_files[0]

    print(
        "\nUsing textbook:"
    )

    print(
        textbook_path
    )

    # ========================================================
    # 2. EXTRACT TEXT FROM PDF
    # ========================================================

    print(
        "\n[1] Extracting textbook text..."
    )

    pages = extract_text(
        str(textbook_path)
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    if not pages:

        print(
            "\nERROR: No text extracted from textbook."
        )

        return

    # Preview
    print(
        "\nFirst page preview:"
    )

    print(
        "----------------------------------------------"
    )

    print(
        pages[0]["text"][:1000]
    )

    # ========================================================
    # 3. CREATE TEXTBOOK CHUNKS
    # ========================================================

    print(
        "\n[2] Creating textbook chunks..."
    )

    chunks = chunk_text(
        pages
    )

    print(
        f"Total chunks: {len(chunks)}"
    )

    if not chunks:

        print(
            "\nERROR: No chunks created."
        )

        return

    # Preview first chunk
    print(
        "\nFirst chunk:"
    )

    print(
        "----------------------------------------------"
    )

    print(
        chunks[0]["text"][:1000]
    )

    # ========================================================
    # 4. LOAD EMBEDDING MODEL
    # ========================================================

    print(
        "\n[3] Loading embedding model..."
    )

    embedder = TextbookEmbedder()

    # ========================================================
    # 5. ENTER SEARCH QUERY
    # ========================================================

    question = input(
        "\nEnter a question to search "
        "in the textbook:\n> "
    )

    if not question.strip():

        print(
            "\nERROR: Empty question."
        )

        return

    # ========================================================
    # 6. RETRIEVE RELEVANT CHUNKS
    # ========================================================

    print(
        "\n[4] Retrieving relevant textbook content..."
    )

    results = retrieve_relevant_chunks(
        question,
        chunks,
        embedder,
        top_k=5
    )

    # ========================================================
    # 7. DISPLAY RESULTS
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "TOP RELEVANT TEXTBOOK CONTENT"
    )

    print(
        "=============================================="
    )

    if not results:

        print(
            "\nNo relevant textbook content found."
        )

        return

    for rank, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\n--- RESULT {rank} ---"
        )

        print(
            f"Page: "
            f"{result['page']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.3f}"
        )

        print(
            "\nText:"
        )

        print(
            result["text"]
        )

    # ========================================================
    # 8. SUCCESS MESSAGE
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "RAG RETRIEVAL TEST COMPLETE"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":

    main()
