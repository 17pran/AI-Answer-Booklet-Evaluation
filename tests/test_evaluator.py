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
    text = Path(ocr_file).read_text(encoding="utf-8")

    # Remove page headers and normalize text
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

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("ﬁ", "fi").replace("ﬂ", "fl")

    # Find question markers such as Q12, Q12), Q 12., etc.
    pattern = re.compile(
        r"(?<!\w)Q\s*(\d{1,3})\s*[\)\].:\-]?(?=\s|$)",
        re.IGNORECASE
    )

    matches = list(pattern.finditer(text))
    answers = {}

    for i, match in enumerate(matches):
        question_number = int(match.group(1))

        start = match.end()
        end = (
            matches[i + 1].start()
            if i + 1 < len(matches)
            else len(text)
        )

        block = text[start:end].strip()

        if not block:
            continue

        # Remove a leading "Question:" label
        block = re.sub(
            r"^\s*Question\s*:\s*",
            "",
            block,
            flags=re.IGNORECASE
        )

        # Prefer an explicit Answer: or Ans: separator
        answer_match = re.search(
            r"\b(?:Answer|Ans)\s*[\.:]?\s*",
            block,
            re.IGNORECASE
        )

        if answer_match:
            question_text = block[:answer_match.start()].strip()
            answer_text = block[answer_match.end():].strip()

        else:
            # If OCR has no answer label, treat the block as
            # the student's answer rather than inventing a question.
            question_text = ""
            answer_text = block

        question_text = clean_text(question_text)
        answer_text = clean_text(answer_text)

        if not answer_text:
            print(
                f"Warning: No answer extracted for Q{question_number}"
            )
            continue

        # Avoid silently overwriting an existing answer
        if question_number in answers:
            answers[question_number]["answer"] += " " + answer_text
        else:
            answers[question_number] = {
                "question_number": question_number,
                "question": question_text,
                "answer": answer_text
            }

    print("\n--- OCR PARSING SUMMARY ---")

    for number, item in sorted(answers.items()):
        print(f"\nQ{number}")
        print(f"Question: {item['question'] or '(not detected)'}")
        print(f"Answer: {item['answer'][:250]}")

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

def match_questions(student_questions, answer_key_questions):
    matches = []

    # Normalize student question keys and numbers to int
    student_by_number = {}
    for k, v in student_questions.items():
        try:
            num = int(k)
            student_by_number[num] = dict(v)
            student_by_number[num]["question_number"] = num
        except (ValueError, TypeError):
            student_by_number[k] = v

    key_by_number = {
        int(key["question_number"]): key
        for key in answer_key_questions
    }

    matched_keys = set()
    matched_students = set()

    # 1. Match questions with the exact same number (int-safe)
    for number in sorted(student_by_number.keys(), key=lambda x: int(x) if str(x).isdigit() else 999):
        if number not in key_by_number:
            continue

        student = student_by_number[number]
        key = key_by_number[number]

        if not student.get("answer", "").strip():
            continue

        similarity = question_similarity(
            student.get("question", ""),
            key.get("question", "")
        )

        matches.append({
            "student": student,
            "key": key,
            "match_type": "EXACT NUMBER",
            "similarity": similarity
        })
        matched_keys.add(number)
        matched_students.add(number)

    # 2. Fallback: match by question text similarity for any remaining questions
    for s_num, student in student_by_number.items():
        if s_num in matched_students:
            continue
        s_text = student.get("question", "").strip()
        if not s_text or not student.get("answer", "").strip():
            continue

        best_key = None
        best_sim = 0.0
        for k_num, key in key_by_number.items():
            if k_num in matched_keys:
                continue
            sim = question_similarity(s_text, key.get("question", ""))
            if sim >= 0.35 and sim > best_sim:
                best_sim = sim
                best_key = key

        if best_key is not None:
            matches.append({
                "student": student,
                "key": best_key,
                "match_type": "TEXT SIMILARITY",
                "similarity": best_sim
            })
            matched_keys.add(best_key["question_number"])
            matched_students.add(s_num)

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
