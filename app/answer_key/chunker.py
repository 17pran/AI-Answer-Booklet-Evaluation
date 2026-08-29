def chunk_answers(
    questions,
    max_chars=1500
):
    """
    Create question-based chunks.

    Each question remains associated with
    its model answer.
    """

    chunks = []

    chunk_id = 0

    for question in questions:

        answer = question[
            "model_answer"
        ]

        # ----------------------------------------------------
        # Small answer → keep as one chunk
        # ----------------------------------------------------

        if len(answer) <= max_chars:

            chunks.append({

                "chunk_id":
                    chunk_id,

                "question_id":
                    question[
                        "question_id"
                    ],

                "question":
                    question[
                        "question"
                    ],

                "text":
                    answer

            })

            chunk_id += 1

        # ----------------------------------------------------
        # Large answer → split into pieces
        # ----------------------------------------------------

        else:

            start = 0

            while start < len(answer):

                end = (
                    start
                    + max_chars
                )

                chunk = answer[
                    start:end
                ].strip()

                if chunk:

                    chunks.append({

                        "chunk_id":
                            chunk_id,

                        "question_id":
                            question[
                                "question_id"
                            ],

                        "question":
                            question[
                                "question"
                            ],

                        "text":
                            chunk

                    })

                    chunk_id += 1

                start = end

    return chunks
