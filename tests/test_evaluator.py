from pathlib import Path
import re
from difflib import SequenceMatcher

from app.config import (
    TEXTBOOKS_DIR,
    ANSWER_KEYS_DIR,
    PROCESSED_DIR
)

from app.textbook.pdf_parser import extract_text
from app.textbook.chunker import chunk_text
from app.textbook.embeddings import TextbookEmbedder

from app.evaluation.retrieval import (
    retrieve_relevant_chunks
)

from app.evaluation.evaluator import (
    AnswerEvaluator
)

from app.evaluation.answer_parser import (
    parse_answer_key,
    clean_text
)


# ============================================================
# STUDENT OCR PARSER
# ============================================================

def parse_student_ocr(ocr_file):

    text = Path(
        ocr_file
    ).read_text(
        encoding="utf-8"
    )

    # ---------------------------------------------------------
    # Remove page headers / footers
    # ---------------------------------------------------------

    text = re.sub(
        r"===== PAGE \d+ =====",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"Page \d+ of \d+",
        "",
        text,
        flags=re.IGNORECASE
    )

    # ---------------------------------------------------------
    # Normalize OCR text
    # ---------------------------------------------------------

    text = text.replace(
        "\r",
        "\n"
    )

    text = text.replace(
        "ﬁ",
        "fi"
    )

    text = text.replace(
        "ﬂ",
        "fl"
    )

    # ---------------------------------------------------------
    # Question markers
    #
    # Supports:
    #
    # Q12)
    # Q12 )
    # Q12.
    # Q12:
    # Q12
    #
    # Also supports OCR forms such as:
    # Q 12 )
    # ---------------------------------------------------------

    pattern = re.compile(
        r"""
        (?<!\w)
        Q\s*(\d{1,3})
        \s*[\)\].:\-]?
        (?=\s)
        """,
        re.IGNORECASE | re.VERBOSE
    )

    matches = list(
        pattern.finditer(text)
    )

    answers = {}

    for i, match in enumerate(matches):

        question_number = int(
            match.group(1)
        )

        start = match.end()

        if i + 1 < len(matches):

            end = matches[i + 1].start()

        else:

            end = len(text)

        block = text[
            start:end
        ].strip()

        if not block:
            continue

        # -----------------------------------------------------
        # Remove "Question:" if OCR contains it
        # -----------------------------------------------------

        block = re.sub(
            r"^\s*Question\s*:\s*",
            "",
            block,
            flags=re.IGNORECASE
        )

        # -----------------------------------------------------
        # Separate Question and Answer
        #
        # Example:
        #
        # Question: What is SE?
        # Answer: SE is...
        # -----------------------------------------------------

        answer_match = re.search(
            r"\bAnswer\s*:\s*",
            block,
            re.IGNORECASE
        )

        if answer_match:

            question_text = block[
                :answer_match.start()
            ].strip()

            answer_text = block[
                answer_match.end():
            ].strip()

        else:

            # -------------------------------------------------
            # Try "Ans."
            # -------------------------------------------------

            ans_match = re.search(
                r"\bAns\s*[\.:]?\s*",
                block,
                re.IGNORECASE
            )

            if ans_match:

                question_text = block[
                    :ans_match.start()
                ].strip()

                answer_text = block[
                    ans_match.end():
                ].strip()

            else:

                # -------------------------------------------------
                # If there is no explicit Answer marker,
                # try to use the first line as the question.
                # -------------------------------------------------

                lines = [
                    line.strip()
                    for line in block.split("\n")
                    if line.strip()
                ]

                if len(lines) >= 2:

                    question_text = lines[0]

                    answer_text = " ".join(
                        lines[1:]
                    )

                else:

                    question_text = ""
                    answer_text = block

        question_text = clean_text(
            question_text
        )

        answer_text = clean_text(
            answer_text
        )

        if not answer_text:
            continue

        # -----------------------------------------------------
        # Store
        # -----------------------------------------------------

        answers[question_number] = {
            "question_number": question_number,
            "question": question_text,
            "answer": answer_text
        }

    return answers


# ============================================================
# NORMALIZE QUESTION TEXT
# ============================================================

def normalize_question(text):

    text = text.lower()

    # Remove common question words that don't affect matching
    text = re.sub(
        r"\b(what are|what is|explain|describe|define|"
        r"write short notes on|differentiate between|"
        r"differentiate)\b",
        " ",
        text
    )

    # Normalize common OCR variations
    replacements = {
        "non func": "non functional",
        "non-func": "non functional",
        "nonfunc": "non functional",
        "functional vs": "functional and",
        "sdlc": "software development life cycle",
        "se": "software engineering",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Keep only letters and numbers
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Normalize spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# QUESTION TEXT SIMILARITY
# ============================================================

def question_similarity(
    student_question,
    key_question
):

    a = normalize_question(
        student_question
    )

    b = normalize_question(
        key_question
    )

    if not a or not b:
        return 0.0

    # ---------------------------------------------------------
    # Direct sequence similarity
    # ---------------------------------------------------------

    sequence_score = SequenceMatcher(
        None,
        a,
        b
    ).ratio()

    # ---------------------------------------------------------
    # Word overlap
    # ---------------------------------------------------------

    a_words = set(
        a.split()
    )

    b_words = set(
        b.split()
    )

    if not a_words or not b_words:

        word_score = 0.0

    else:

        intersection = (
            a_words &
            b_words
        )

        union = (
            a_words |
            b_words
        )

        word_score = (
            len(intersection) /
            len(union)
        )

    # ---------------------------------------------------------
    # Combined score
    # ---------------------------------------------------------

    score = (
        0.6 * sequence_score +
        0.4 * word_score
    )

    return round(
        score,
        3
    )


# ============================================================
# MATCH QUESTIONS
# ============================================================

def match_questions(
    student_questions,
    answer_key_questions
):

    matches = []

    used_student_numbers = set()
    used_key_numbers = set()

    # ---------------------------------------------------------
    # STEP 1
    # Exact question number match
    #
    # ONLY use this when the question text is also reasonably
    # compatible.
    # ---------------------------------------------------------

    for key in answer_key_questions:

        key_number = key[
            "question_number"
        ]

        key_text = key[
            "question"
        ]

        if key_number not in student_questions:
            continue

        student = student_questions[
            key_number
        ]

        student_text = student[
            "question"
        ]

        similarity = question_similarity(
            student_text,
            key_text
        )

        # If student question text is empty, still allow
        # exact number.
        if not student_text:

            matches.append({
                "student": student,
                "key": key,
                "match_type": "EXACT NUMBER",
                "similarity": 1.0
            })

            used_student_numbers.add(
                key_number
            )

            used_key_numbers.add(
                key_number
            )

        elif similarity >= 0.35:

            matches.append({
                "student": student,
                "key": key,
                "match_type": "EXACT NUMBER",
                "similarity": similarity
            })

            used_student_numbers.add(
                key_number
            )

            used_key_numbers.add(
                key_number
            )

    # ---------------------------------------------------------
    # STEP 2
    # Question text matching
    #
    # This handles:
    #
    # Student Q1 → Answer Key Q14
    # Student Q2 → Answer Key Q16
    # ---------------------------------------------------------

    remaining_students = [
        student
        for number, student
        in student_questions.items()
        if number not in used_student_numbers
    ]

    remaining_keys = [
        key
        for key in answer_key_questions
        if key["question_number"]
        not in used_key_numbers
    ]

    candidates = []

    for student in remaining_students:

        for key in remaining_keys:

            similarity = question_similarity(
                student["question"],
                key["question"]
            )

            candidates.append({
                "student": student,
                "key": key,
                "similarity": similarity
            })

    # Highest similarity first
    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    used_students = set()
    used_keys = set()

    for candidate in candidates:

        student_number = candidate[
            "student"
        ]["question_number"]

        key_number = candidate[
            "key"
        ]["question_number"]

        similarity = candidate[
            "similarity"
        ]

        if student_number in used_students:
            continue

        if key_number in used_keys:
            continue

        # -----------------------------------------------------
        # Matching threshold
        # -----------------------------------------------------

        if similarity < 0.40:
            continue

        matches.append({
            "student": candidate["student"],
            "key": candidate["key"],
            "match_type": "QUESTION TEXT",
            "similarity": similarity
        })

        used_students.add(
            student_number
        )

        used_keys.add(
            key_number
        )

    return matches


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n==============================================")
    print("FULL ANSWER EVALUATION TEST")
    print("==============================================")

    # ========================================================
    # 1. ANSWER KEY
    # ========================================================

    answer_key_files = sorted(
        ANSWER_KEYS_DIR.glob("*.pdf")
    )

    if not answer_key_files:

        print(
            "\nERROR: No answer key PDF found."
        )

        return

    answer_key_path = answer_key_files[0]

    print(
        f"\nUsing answer key:"
        f"\n{answer_key_path}"
    )

    # ========================================================
    # 2. PARSE ANSWER KEY
    # ========================================================

    print(
        "\n[1] Reading model answer..."
    )

    answer_key_pages = extract_text(
        str(answer_key_path)
    )

    questions = parse_answer_key(
        answer_key_pages
    )

    print(
        f"Questions found: {len(questions)}"
    )

    if not questions:

        print(
            "\nERROR: No questions found."
        )

        return

    print(
        "\n----------------------------------------------"
    )

    print(
        "ANSWER KEY QUESTIONS"
    )

    print(
        "----------------------------------------------"
    )

    for question in questions:

        print(
            f"\nQ{question['question_number']}:"
        )

        print(
            question["question"]
        )

    # ========================================================
    # 3. STUDENT OCR
    # ========================================================

    print(
        "\n[2] Reading student OCR..."
    )

    ocr_file = (
        PROCESSED_DIR /
        "ocr_output.txt"
    )

    if not ocr_file.exists():

        print(
            "\nERROR: Student OCR not found:"
        )

        print(
            ocr_file
        )

        return

    student_questions = parse_student_ocr(
        ocr_file
    )

    print(
        f"Student questions detected:"
        f" {len(student_questions)}"
    )

    print(
        "\n----------------------------------------------"
    )

    print(
        "STUDENT OCR QUESTIONS"
    )

    print(
        "----------------------------------------------"
    )

    for number, student in student_questions.items():

        print(
            f"\nQ{number}:"
        )

        print(
            f"Question: "
            f"{student['question']}"
        )

        print(
            f"Answer: "
            f"{student['answer'][:300]}"
        )

    # ========================================================
    # 4. MATCH QUESTIONS
    # ========================================================

    print(
        "\n=============================================="
    )

    print(
        "QUESTION NUMBER / TEXT MATCHING"
    )

    print(
        "=============================================="
    )

    matches = match_questions(
        student_questions,
        questions
    )

    for match in matches:

        student = match["student"]
        key = match["key"]

        print(
            f"\nOCR Q{student['question_number']}"
            f"  →  "
            f"ANSWER KEY Q{key['question_number']}"
        )

        print(
            f"Match type: "
            f"{match['match_type']}"
        )

        print(
            f"Similarity: "
            f"{match['similarity']:.3f}"
        )

        print(
            f"Student question: "
            f"{student['question']}"
        )

        print(
            f"Answer-key question: "
            f"{key['question']}"
        )

    # ========================================================
    # MATCHING SUMMARY
    # ========================================================

    matched_key_numbers = {
        match["key"]["question_number"]
        for match in matches
    }

    matched_student_numbers = {
        match["student"]["question_number"]
        for match in matches
    }

    all_key_numbers = {
        q["question_number"]
        for q in questions
    }

    all_student_numbers = set(
        student_questions.keys()
    )

    missing_student_answers = (
        all_key_numbers -
        matched_key_numbers
    )

    unmatched_student_questions = (
        all_student_numbers -
        matched_student_numbers
    )

    print(
        "\n=============================================="
    )

    print(
        "MATCHING SUMMARY"
    )

    print(
        "=============================================="
    )

    print(
        f"Matched answer-key questions:"
        f" {sorted(matched_key_numbers)}"
    )

    print(
        f"Missing student answers:"
        f" {sorted(missing_student_answers)}"
    )

    print(
        f"Unmatched student questions:"
        f" {sorted(unmatched_student_questions)}"
    )

    # ========================================================
    # 5. TEXTBOOK
    # ========================================================

    print(
        "\n[3] Loading textbook..."
    )

    textbook_files = sorted(
        TEXTBOOKS_DIR.glob("*.pdf")
    )

    if not textbook_files:

        print(
            "\nERROR: No textbook PDF found."
        )

        return

    textbook_path = textbook_files[0]

    print(
        f"Using textbook:"
        f"\n{textbook_path}"
    )

    textbook_pages = extract_text(
        str(textbook_path)
    )

    print(
        f"Textbook pages:"
        f" {len(textbook_pages)}"
    )

    if not textbook_pages:

        print(
            "\nERROR: No textbook text extracted."
        )

        return

    # ========================================================
    # 6. CHUNKS
    # ========================================================

    print(
        "\n[4] Creating textbook chunks..."
    )

    chunks = chunk_text(
        textbook_pages
    )

    print(
        f"Total chunks:"
        f" {len(chunks)}"
    )

    # ========================================================
    # 7. EMBEDDINGS
    # ========================================================

    print(
        "\n[5] Loading embedding model..."
    )

    embedder = TextbookEmbedder()

    # ========================================================
    # 8. LLM
    # ========================================================

    print(
        "\n[6] Loading LLM evaluator..."
    )

    evaluator = AnswerEvaluator(
        model="qwen3:8b"
    )

    # ========================================================
    # 9. EVALUATE
    # ========================================================

    total_marks = 0
    total_max_marks = 0
    evaluated_count = 0

    print(
        "\n=============================================="
    )

    print(
        "STARTING ANSWER EVALUATION"
    )

    print(
        "=============================================="
    )

    # --------------------------------------------------------
    # Evaluate according to answer-key order
    # --------------------------------------------------------

    for match in matches:

        student = match["student"]
        key = match["key"]

        question_number = key[
            "question_number"
        ]

        question_text = key[
            "question"
        ]

        model_answer = key[
            "answer"
        ]

        student_answer = student[
            "answer"
        ]

        print(
            "\n=============================================="
        )

        print(
            f"QUESTION {question_number}"
        )

        print(
            "=============================================="
        )

        print(
            f"\nMatching method:"
            f" {match['match_type']}"
        )

        print(
            f"Match similarity:"
            f" {match['similarity']:.3f}"
        )

        print(
            f"\nQuestion:\n"
            f"{question_text}"
        )

        print(
            f"\nStudent OCR Answer:\n"
            f"{student_answer}"
        )

        print(
            f"\nModel Answer:\n"
            f"{model_answer}"
        )

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        print(
            "\nRetrieving textbook content..."
        )

        textbook_results = (
            retrieve_relevant_chunks(
                question_text,
                chunks,
                embedder,
                top_k=5
            )
        )

        print(
            f"Retrieved:"
            f" {len(textbook_results)} chunks"
        )

        # ----------------------------------------------------
        # LLM
        # ----------------------------------------------------

        print(
            "\nEvaluating with Qwen..."
        )

        result = evaluator.evaluate(
            question=question_text,
            student_answer=student_answer,
            model_answer=model_answer,
            textbook_chunks=textbook_results,
            max_marks=5
        )

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        print(
            "\n----------------------------------------------"
        )

        print(
            f"Q{question_number} RESULT"
        )

        print(
            "----------------------------------------------"
        )

        print(
            f"Marks:"
            f" {result['marks']}/"
            f"{result['max_marks']}"
        )

        print(
            f"Score:"
            f" {result['score_percentage']}%"
        )

        print(
            "\nOCR Interpretation:"
        )

        print(
            result["ocr_interpretation"]
        )

        print(
            "\nJustification:"
        )

        print(
            result["justification"]
        )

        print(
            "\nCorrect concepts:"
        )

        for concept in result[
            "correct_concepts"
        ]:

            print(
                f"- {concept}"
            )

        print(
            "\nMissing concepts:"
        )

        for concept in result[
            "missing_concepts"
        ]:

            print(
                f"- {concept}"
            )

        print(
            "\nIncorrect concepts:"
        )

        for concept in result[
            "incorrect_concepts"
        ]:

            print(
                f"- {concept}"
            )

        print(
            "\nTextbook support:"
        )

        for support in result[
            "textbook_support"
        ]:

            print(
                f"- {support}"
            )

        print(
            "\nFeedback:"
        )

        print(
            result["feedback"]
        )

        # ----------------------------------------------------
        # TOTALS
        # ----------------------------------------------------

        total_marks += float(
            result["marks"]
        )

        total_max_marks += float(
            result["max_marks"]
        )

        evaluated_count += 1

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n\n=============================================="
    )

    print(
        "FINAL EVALUATION RESULT"
    )

    print(
        "=============================================="
    )

    if evaluated_count == 0:

        print(
            "\nNo questions could be evaluated."
        )

        return

    percentage = (
        total_marks /
        total_max_marks
    ) * 100

    print(
        f"\nQuestions evaluated:"
        f" {evaluated_count}"
    )

    print(
        f"Total Marks:"
        f" {total_marks}/"
        f"{total_max_marks}"
    )

    print(
        f"Percentage:"
        f" {percentage:.2f}%"
    )

    print(
        "\n----------------------------------------------"
    )

    print(
        "MATCHING SUMMARY"
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"Matched questions:"
        f" {len(matches)}"
    )

    print(
        f"Missing student answers:"
        f" {sorted(missing_student_answers)}"
    )

    print(
        f"Unmatched student questions:"
        f" {sorted(unmatched_student_questions)}"
    )

    print(
        "\n=============================================="
    )

    print(
        "FULL EVALUATION COMPLETE"
    )

    print(
        "==============================================")


if __name__ == "__main__":
    main()
