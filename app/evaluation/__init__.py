# app/evaluation/scoring.py
from app.config import (
    LLM_MODEL,
    OLLAMA_URL
)
import requests
import json
import re
from app.evaluation.scoring import retrieve_relevant_chunks
from app.config import TEXTBOOKS_DIR
from app.textbook.embeddings import TextbookEmbedder
from app.textbook.chunker import chunk_text
from app.textbook.pdf_parser import extract_text
from pathlib import Path
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

# i already have a test_textbook.py


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


# but u told me to for rag do


def main():

    textbook_path = (
        TEXTBOOKS_DIR /
        "textbook.pdf"
    )

    if not textbook_path.exists():

        print(
            f"Put your textbook here:\n"
            f"{textbook_path}"
        )

        return

    print("\nReading textbook...")

    pages = extract_text(
        str(textbook_path)
    )

    print(
        f"Extracted {len(pages)} pages."
    )

    chunks = chunk_text(pages)

    print(
        f"Created {len(chunks)} chunks."
    )

    embedder = TextbookEmbedder()

    question = input(
        "\nEnter a question to search: "
    )

    results = retrieve_relevant_chunks(
        question,
        chunks,
        embedder,
        top_k=5
    )

    print("\nTOP RELEVANT TEXTBOOK CONTENT")
    print("=" * 60)

    for result in results:

        print(
            f"\nPage: {result['page']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.3f}"
        )

        print(
            result["text"]
        )


if __name__ == "__main__":
    main()

# step 16 was answer extraction
# app/evaluation/answer_extractor.py


QUESTION_PATTERN = re.compile(
    r"^\s*(?:Q(?:uestion)?\s*)?(\d+)"
    r"\s*[\.\):\-]?\s*(.*)$",
    re.IGNORECASE
)


def extract_answers(ocr_pages):

    answers = {}

    current_question = None

    for page in ocr_pages:

        lines = page["text"].splitlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            match = QUESTION_PATTERN.match(
                line
            )

            if match:

                question_number = int(
                    match.group(1)
                )

                remaining_text = (
                    match.group(2).strip()
                )

                current_question = (
                    question_number
                )

                if current_question not in answers:

                    answers[current_question] = {
                        "question_number":
                            current_question,

                        "text": "",

                        "pages": []
                    }

                if remaining_text:

                    answers[
                        current_question
                    ]["text"] += (
                        remaining_text + " "
                    )

                if page["page"] not in answers[
                    current_question
                ]["pages"]:

                    answers[
                        current_question
                    ]["pages"].append(
                        page["page"]
                    )

            elif current_question is not None:

                answers[
                    current_question
                ]["text"] += (
                    line + " "
                )

                if page["page"] not in answers[
                    current_question
                ]["pages"]:

                    answers[
                        current_question
                    ]["pages"].append(
                        page["page"]
                    )

    for answer in answers.values():

        answer["text"] = (
            answer["text"].strip()
        )

    return answers


# 17 evaluation promt
# app/evaluation/evaluator.py


class AnswerEvaluator:

    def __init__(self):

        self.model = LLM_MODEL
        self.url = OLLAMA_URL

    def build_prompt(
        self,
        question,
        student_answer,
        reference_text,
        maximum_marks
    ):

        prompt = f"""
You are an AI-assisted university/school answer evaluator.

Your task is to evaluate a student's answer fairly.

IMPORTANT RULES:

1. Evaluate the meaning and logic of the answer.
2. Do NOT require the student to use the exact textbook wording.
3. Give partial marks when the student demonstrates partial understanding.
4. Do not award marks for information that is incorrect.
5. Use the reference textbook only as supporting knowledge.
6. Do not invent facts that are not supported by the reference.
7. The final marks must be between 0 and the maximum marks.
8. Give a concise explanation for the score.

QUESTION:
{question}

MAXIMUM MARKS:
{maximum_marks}

STUDENT ANSWER:
{student_answer}

RELEVANT TEXTBOOK CONTENT:
{reference_text}

Return ONLY valid JSON in this format:

{{
    "marks_awarded": 0,
    "confidence": 0.0,
    "correctness": 0.0,
    "relevance": 0.0,
    "completeness": 0.0,
    "strengths": [],
    "missing_points": [],
    "feedback": ""
}}

Scores such as confidence, correctness, relevance and completeness
must be between 0 and 1.
"""

        return prompt

    def evaluate(
        self,
        question,
        student_answer,
        reference_text,
        maximum_marks
    ):

        prompt = self.build_prompt(
            question,
            student_answer,
            reference_text,
            maximum_marks
        )

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=180
        )

        response.raise_for_status()

        data = response.json()

        raw_response = data.get(
            "response",
            ""
        )

        result = json.loads(
            raw_response
        )

        # Safety validation
        marks = float(
            result.get(
                "marks_awarded",
                0
            )
        )

        marks = max(
            0,
            min(
                marks,
                maximum_marks
            )
        )

        result["marks_awarded"] = marks

        return result
    # after this its ollama
