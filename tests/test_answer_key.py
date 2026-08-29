from app.answer_key.parser import (
    extract_text,
    parse_questions
)

from app.answer_key.chunker import (
    chunk_answers
)

from app.answer_key.embeddings import (
    AnswerKeyEmbedder
)

from app.config import (
    ANSWER_KEYS_DIR
)


def main():

    print(
        "\n=============================================="
    )

    print(
        "MODEL ANSWER PIPELINE TEST"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # Find PDF automatically
    # --------------------------------------------------------

    pdf_files = sorted(
        ANSWER_KEYS_DIR.glob(
            "*.pdf"
        )
    )

    if not pdf_files:

        print(
            "\nERROR: No model-answer PDF found."
        )

        print(
            "Put the PDF inside:"
        )

        print(
            ANSWER_KEYS_DIR
        )

        return

    pdf_path = pdf_files[0]

    print(
        f"\nUsing model answer:"
        f"\n{pdf_path}"
    )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    print(
        "\n[1] Extracting PDF..."
    )

    pages = extract_text(
        str(pdf_path)
    )

    print(
        f"Pages with text: "
        f"{len(pages)}"
    )

    # --------------------------------------------------------
    # Parse questions
    # --------------------------------------------------------

    print(
        "\n[2] Parsing questions..."
    )

    questions = parse_questions(
        pages
    )

    print(
        f"Questions detected: "
        f"{len(questions)}"
    )

    for question in questions[:3]:

        print(
            "\n--------------------------------"
        )

        print(
            question["question_id"]
        )

        print(
            "Question:"
        )

        print(
            question["question"]
        )

        print(
            "\nModel Answer:"
        )

        print(
            question["model_answer"][
                :500
            ]
        )

    # --------------------------------------------------------
    # Chunk
    # --------------------------------------------------------

    print(
        "\n[3] Creating answer chunks..."
    )

    chunks = chunk_answers(
        questions
    )

    print(
        f"Total answer chunks: "
        f"{len(chunks)}"
    )

    # --------------------------------------------------------
    # Embeddings
    # --------------------------------------------------------

    print(
        "\n[4] Creating embeddings..."
    )

    embedder = AnswerKeyEmbedder()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedder.embed(
        texts
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    # --------------------------------------------------------
    # Query test
    # --------------------------------------------------------

    query = (
        "Explain the waterfall model "
        "in software engineering."
    )

    query_embedding = (
        embedder.embed_query(
            query
        )
    )

    print(
        f"Query embedding shape: "
        f"{query_embedding.shape}"
    )

    print(
        "\n=============================================="
    )

    print(
        "MODEL ANSWER PIPELINE SUCCESSFUL"
    )

    print(
        "=============================================="
    )


if __name__ == "__main__":

    main()
