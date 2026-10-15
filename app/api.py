import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
import traceback
from pathlib import Path
from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.database import engine
from app import models
from app.config import BASE_DIR
from app.utils.pdf_utils import pdf_to_images
from app.ocr.ocr_engine import HandwritingOCR

from app.textbook.pdf_parser import extract_text
from app.textbook.chunker import chunk_text
from app.textbook.embeddings import TextbookEmbedder
from app.textbook.processor import (
    initialize_textbook_tables,
    process_textbook,
    list_textbooks
)

from app.evaluation.retrieval import retrieve_relevant_chunks
from app.evaluation.evaluator import AnswerEvaluator
from app.evaluation.answer_parser import parse_answer_key

from tests.test_evaluator import parse_student_ocr, match_questions


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AI Answer Booklet Evaluation API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# DATABASE CONFIGURATION & OCR CACHE
# =========================================================

DATABASE_PATH = BASE_DIR / "data" / "evaluation_history.db"


def initialize_database():
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                student_usn TEXT NOT NULL,
                obtained_marks REAL NOT NULL,
                total_marks REAL NOT NULL,
                percentage REAL NOT NULL,
                details TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS answer_scripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                student_name TEXT,
                student_usn TEXT,
                extracted_text TEXT NOT NULL,
                parsed_questions TEXT NOT NULL,
                avg_confidence REAL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)


def compute_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def get_cached_answer_script(file_hash: str) -> Optional[dict]:
    # 1. Try PostgreSQL first
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT file_hash, filename, student_name, student_usn,
                           extracted_text, parsed_questions, avg_confidence
                    FROM answer_scripts
                    WHERE file_hash = :hash
                """),
                {"hash": file_hash}
            ).mappings().first()
            if row:
                return {
                    "file_hash": row["file_hash"],
                    "filename": row["filename"],
                    "student_name": row["student_name"] or "",
                    "student_usn": row["student_usn"] or "",
                    "extracted_text": row["extracted_text"],
                    "parsed_questions": json.loads(row["parsed_questions"]),
                    "avg_confidence": float(row["avg_confidence"] or 0.0),
                }
    except Exception as exc:
        print(f"PostgreSQL cache lookup warning: {exc}")

    # 2. Try SQLite cache
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT file_hash, filename, student_name, student_usn,
                       extracted_text, parsed_questions, avg_confidence
                FROM answer_scripts
                WHERE file_hash = ?
                """,
                (file_hash,)
            ).fetchone()
            if row:
                return {
                    "file_hash": row["file_hash"],
                    "filename": row["filename"],
                    "student_name": row["student_name"] or "",
                    "student_usn": row["student_usn"] or "",
                    "extracted_text": row["extracted_text"],
                    "parsed_questions": json.loads(row["parsed_questions"]),
                    "avg_confidence": float(row["avg_confidence"] or 0.0),
                }
    except Exception as exc:
        print(f"SQLite cache lookup warning: {exc}")

    return None


def save_cached_answer_script(
    file_hash: str,
    filename: str,
    student_name: Optional[str],
    student_usn: Optional[str],
    extracted_text: str,
    parsed_questions: dict,
    avg_confidence: float
) -> None:
    questions_json = json.dumps(parsed_questions)

    # 1. Save to PostgreSQL
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO answer_scripts (
                        file_hash, filename, student_name, student_usn,
                        extracted_text, parsed_questions, avg_confidence, created_at
                    ) VALUES (
                        :hash, :filename, :name, :usn,
                        :text, :questions, :conf, NOW()
                    )
                    ON CONFLICT (file_hash) DO UPDATE SET
                        filename = EXCLUDED.filename,
                        student_name = COALESCE(EXCLUDED.student_name, answer_scripts.student_name),
                        student_usn = COALESCE(EXCLUDED.student_usn, answer_scripts.student_usn),
                        extracted_text = EXCLUDED.extracted_text,
                        parsed_questions = EXCLUDED.parsed_questions,
                        avg_confidence = EXCLUDED.avg_confidence
                """),
                {
                    "hash": file_hash,
                    "filename": filename,
                    "name": student_name,
                    "usn": student_usn,
                    "text": extracted_text,
                    "questions": questions_json,
                    "conf": avg_confidence,
                }
            )
    except Exception as exc:
        print(f"PostgreSQL cache save warning: {exc}")

    # 2. Save to SQLite
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            conn.execute(
                """
                INSERT INTO answer_scripts (
                    file_hash, filename, student_name, student_usn,
                    extracted_text, parsed_questions, avg_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    filename = excluded.filename,
                    student_name = COALESCE(excluded.student_name, answer_scripts.student_name),
                    student_usn = COALESCE(excluded.student_usn, answer_scripts.student_usn),
                    extracted_text = excluded.extracted_text,
                    parsed_questions = excluded.parsed_questions,
                    avg_confidence = excluded.avg_confidence
                """,
                (
                    file_hash,
                    filename,
                    student_name,
                    student_usn,
                    extracted_text,
                    questions_json,
                    avg_confidence
                )
            )
    except Exception as exc:
        print(f"SQLite cache save warning: {exc}")



# =========================================================
# MODEL CACHING
# =========================================================

ocr_engine = None
embedder = None
evaluator = None


def get_models():
    global ocr_engine, embedder, evaluator

    if ocr_engine is None:
        ocr_engine = HandwritingOCR()

    if embedder is None:
        embedder = TextbookEmbedder()

    if evaluator is None:
        evaluator = AnswerEvaluator(
            model="qwen3:8b"
        )

    return ocr_engine, embedder, evaluator


# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
def startup_event():
    initialize_database()
    initialize_textbook_tables()


# =========================================================
# COMMON HELPERS
# =========================================================

def validate_pdf(upload: UploadFile):
    if (
        not upload.filename
        or Path(upload.filename).suffix.lower() != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"{upload.filename or 'Uploaded file'} "
                "must be a PDF."
            )
        )


async def save_upload(
    upload: UploadFile,
    destination: Path
):
    content = await upload.read()

    if not content:
        raise HTTPException(
            status_code=400,
            detail=f"{upload.filename} is empty."
        )

    destination.write_bytes(content)

    return destination


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
def health():
    return {
        "status": "running",
        "service": "AI Answer Booklet Evaluation API"
    }


# =========================================================
# PROCESS AND SAVE A TEXTBOOK
# =========================================================

@app.post("/api/textbooks/process")
def upload_and_process_textbook(
    file: UploadFile = File(...)
):
    """
    Upload a textbook, process it, and save its
    chunks and embeddings in PostgreSQL.
    """

    if (
        not file.filename
        or Path(file.filename).suffix.lower() != ".pdf"
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF textbook."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False
        ) as temp_file:

            temp_path = Path(temp_file.name)

            shutil.copyfileobj(
                file.file,
                temp_file
            )

        result = process_textbook(
            pdf_path=str(temp_path),
            filename=Path(file.filename).name
        )

        return {
            "success": True,
            **result
        }

    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Textbook processing failed: {exc}"
        )

    finally:
        if temp_path and temp_path.exists():
            temp_path.unlink(missing_ok=True)

        file.file.close()


# =========================================================
# LIST SAVED TEXTBOOKS
# =========================================================

@app.get("/api/textbooks")
def get_saved_textbooks():
    """
    Return textbooks already saved in PostgreSQL.
    """

    try:
        return {
            "success": True,
            "textbooks": list_textbooks()
        }

    except Exception as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Could not load saved textbooks: {exc}"
        )

# =========================================================
# EXTRACT HANDWRITING (OCR) WITH DATABASE CACHING
# =========================================================

@app.post("/api/ocr/extract")
async def extract_handwriting(
    answer_script: UploadFile = File(...),
    student_name: Optional[str] = Form(None),
    student_usn: Optional[str] = Form(None)
):
    """
    Extract handwriting from uploaded student answer script.
    Checks PostgreSQL and SQLite cache first. If already processed,
    instantly returns cached extraction. Otherwise runs TrOCR,
    saves to database, and returns results.
    """
    validate_pdf(answer_script)
    content = await answer_script.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded answer script is empty.")

    file_hash = compute_sha256(content)
    filename = Path(answer_script.filename).name

    # 1. Check Database Cache
    cached = get_cached_answer_script(file_hash)
    if cached:
        print(f"[OCR] Cache hit for hash {file_hash[:10]}... ({cached['filename']})")
        return {
            "success": True,
            "cached": True,
            "scriptHash": file_hash,
            "filename": cached["filename"],
            "studentName": cached["student_name"] or (student_name or "").strip(),
            "studentUSN": cached["student_usn"] or (student_usn or "").strip(),
            "extractedText": cached["extracted_text"],
            "questions": cached["parsed_questions"],
            "confidence": cached["avg_confidence"],
            "totalQuestions": len(cached["parsed_questions"]),
            "message": "Extracted handwriting loaded from database (Instant Cache Hit)"
        }

    # 2. Run OCR Pipeline on Fresh Upload
    print(f"[OCR] Processing fresh script: {filename} ({file_hash[:10]}...)")
    ocr, _, _ = get_models()

    with tempfile.TemporaryDirectory(prefix="ocr_extract_") as temp:
        work_dir = Path(temp)
        pdf_path = work_dir / filename
        pdf_path.write_bytes(content)

        image_dir = work_dir / "ocr_images"
        image_paths = pdf_to_images(str(pdf_path), str(image_dir), dpi=300)

        if not image_paths:
            raise HTTPException(
                status_code=422,
                detail="No pages could be rendered from answer script PDF."
            )

        ocr_results = ocr.process_document(image_paths)
        ocr_text_path = work_dir / "ocr_output.txt"
        ocr.save_text(ocr_results, str(ocr_text_path))

        full_text = ocr_text_path.read_text(encoding="utf-8")
        student_questions = parse_student_ocr(ocr_text_path)

        all_lines = [line for page in ocr_results for line in page["lines"]]
        avg_confidence = (
            sum(line.get("confidence", 0.0) for line in all_lines) / len(all_lines)
            if all_lines else 0.0
        )

        detected_name = (student_name or "").strip()
        detected_usn = (student_usn or "").strip()

        if not detected_name:
            m = re.search(r"Name\s*[:]\s*([A-Za-z\s]+)", full_text)
            if m:
                detected_name = m.group(1).split("\n")[0].strip()

        if not detected_usn:
            m = re.search(r"(?:Roll\s*no\.?|USN)\s*[:.]?\s*([A-Za-z0-9]+)", full_text, re.IGNORECASE)
            if m:
                detected_usn = m.group(1).strip()

        save_cached_answer_script(
            file_hash=file_hash,
            filename=filename,
            student_name=detected_name,
            student_usn=detected_usn,
            extracted_text=full_text,
            parsed_questions=student_questions,
            avg_confidence=round(avg_confidence, 4)
        )

    return {
        "success": True,
        "cached": False,
        "scriptHash": file_hash,
        "filename": filename,
        "studentName": detected_name,
        "studentUSN": detected_usn,
        "extractedText": full_text,
        "questions": student_questions,
        "confidence": round(avg_confidence, 4),
        "totalQuestions": len(student_questions),
        "message": "Handwriting successfully extracted and saved to database."
    }


# =========================================================
# EVALUATE A STUDENT'S ANSWER BOOKLET
# =========================================================

@app.post("/api/evaluate")
async def evaluate_student(
    student_name: str = Form(...),
    student_usn: str = Form(...),
    textbook_id: int = Form(...),
    answer_key: UploadFile = File(...),
    answer_script: Optional[UploadFile] = File(None),
    script_hash: Optional[str] = Form(None),
    extracted_answers: Optional[str] = Form(None)
):
    student_name = student_name.strip()
    student_usn = student_usn.strip()

    if not student_name or not student_usn:
        raise HTTPException(
            status_code=400,
            detail="Student name and USN are required."
        )

    if textbook_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="A valid textbook must be selected."
        )

    validate_pdf(answer_key)
    if not extracted_answers and not script_hash and not answer_script:
        raise HTTPException(
            status_code=400,
            detail="Either an answer script PDF, a cached script hash, or extracted answers must be provided."
        )

    if answer_script:
        validate_pdf(answer_script)

    try:
        ocr, _, answer_evaluator = get_models()

        # 1. Verify that the selected textbook exists.
        with engine.connect() as connection:
            textbook = connection.execute(
                text("""
                    SELECT
                        t.id,
                        t.filename,
                        COUNT(c.id) AS chunk_count
                    FROM textbooks t
                    LEFT JOIN textbook_chunks c
                        ON c.textbook_id = t.id
                    WHERE t.id = :textbook_id
                    GROUP BY t.id, t.filename
                """),
                {"textbook_id": textbook_id}
            ).mappings().first()

        if textbook is None:
            raise HTTPException(
                status_code=404,
                detail="The selected textbook was not found."
            )

        if textbook["chunk_count"] == 0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "The selected textbook has no saved chunks. "
                    "Please process the textbook first."
                )
            )

        print(
            f"Using textbook: {textbook['filename']} "
            f"(ID: {textbook_id}, "
            f"chunks: {textbook['chunk_count']})"
        )

        with tempfile.TemporaryDirectory(
            prefix="answer_eval_"
        ) as temp:

            work_dir = Path(temp)

            # 2. Save and parse the answer key.
            key_path = await save_upload(
                answer_key,
                work_dir / "answer_key.pdf"
            )

            print("Extracting answer-key text...")
            answer_key_pages = extract_text(str(key_path))
            questions = parse_answer_key(answer_key_pages)

            if not questions:
                raise HTTPException(
                    status_code=422,
                    detail="No questions were detected in the answer key."
                )

            print(f"Answer-key questions found: {len(questions)}")

            # 3. Obtain student answers (from passed JSON, cache, or fresh OCR)
            student_questions = None

            if extracted_answers:
                try:
                    raw_json = json.loads(extracted_answers)
                    student_questions = {
                        int(k) if str(k).isdigit() else k: v
                        for k, v in raw_json.items()
                    }
                    print(f"Using pre-extracted answers passed from frontend ({len(student_questions)} questions).")
                except Exception as exc:
                    print(f"Failed to parse extracted_answers JSON: {exc}")
                    student_questions = None

            if not student_questions and script_hash:
                cached = get_cached_answer_script(script_hash)
                if cached:
                    print(f"Reusing cached OCR answers for hash: {script_hash[:10]}...")
                    student_questions = cached["parsed_questions"]

            if not student_questions and answer_script:
                script_content = await answer_script.read()
                file_hash = compute_sha256(script_content)

                cached = get_cached_answer_script(file_hash)
                if cached:
                    print(f"Reusing cached OCR for uploaded script: {file_hash[:10]}...")
                    student_questions = cached["parsed_questions"]
                else:
                    print("Running OCR on answer script...")
                    script_path = work_dir / Path(answer_script.filename).name
                    script_path.write_bytes(script_content)

                    image_dir = work_dir / "ocr_images"
                    image_paths = pdf_to_images(str(script_path), str(image_dir), dpi=300)

                    if not image_paths:
                        raise HTTPException(
                            status_code=422,
                            detail="No pages could be read from the answer script."
                        )

                    ocr_results = ocr.process_document(image_paths)
                    ocr_text_path = work_dir / "ocr_output.txt"
                    ocr.save_text(ocr_results, str(ocr_text_path))
                    student_questions = parse_student_ocr(ocr_text_path)

                    all_lines = [line for page in ocr_results for line in page["lines"]]
                    avg_conf = (
                        sum(line.get("confidence", 0.0) for line in all_lines) / len(all_lines)
                        if all_lines else 0.0
                    )
                    save_cached_answer_script(
                        file_hash=file_hash,
                        filename=Path(answer_script.filename).name,
                        student_name=student_name,
                        student_usn=student_usn,
                        extracted_text=ocr_text_path.read_text(encoding="utf-8"),
                        parsed_questions=student_questions,
                        avg_confidence=round(avg_conf, 4)
                    )

            if not student_questions:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "No student answers were detected. "
                        "Check the OCR output and question numbering."
                    )
                )

            print(
                f"Student questions detected: "
                f"{len(student_questions)}"
            )

            # 7. Match student answers with the answer key.
            matches = match_questions(
                student_questions,
                questions
            )

            if not matches:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "No student answers matched the answer key. "
                        "Check question numbers and question text."
                    )
                )

            print(f"Questions matched: {len(matches)}")

            # 8. Evaluate each answer using saved textbook chunks.
            question_results = []
            obtained_marks = 0.0
            total_marks = 0.0

            for index, match in enumerate(matches, start=1):
                student = match["student"]
                key = match["key"]

                question_number = key["question_number"]
                question_text = key["question"]
                model_answer = key["answer"]

                student_answer = (
                    student.get("answer", "").strip()
                )

                print(
                    f"\nEvaluating question {index}/{len(matches)} "
                    f"(Q{question_number})..."
                )

                if not student_answer:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"No student answer was extracted "
                            f"for Q{question_number}. "
                            "Check the OCR output."
                        )
                    )

                # Retrieve relevant chunks from PostgreSQL.
                textbook_results = retrieve_relevant_chunks(
                    query=question_text,
                    textbook_id=textbook_id,
                    top_k=5
                )

                if not textbook_results:
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            f"No textbook chunks were retrieved "
                            f"for Q{question_number}."
                        )
                    )

                # Evaluate using the existing AI evaluator.
                result = answer_evaluator.evaluate(
                    question=question_text,
                    student_answer=student_answer,
                    model_answer=model_answer,
                    textbook_chunks=textbook_results,
                    max_marks=5
                )

                marks = float(result["marks"])
                maximum = float(result["max_marks"])

                obtained_marks += marks
                total_marks += maximum

                question_results.append({
                    "questionNumber": question_number,
                    "question": question_text,
                    "studentAnswer": student_answer,
                    "modelAnswer": model_answer,
                    "marks": marks,
                    "maxMarks": maximum,
                    "scorePercentage": result["score_percentage"],
                    "ocrInterpretation": result["ocr_interpretation"],
                    "justification": result["justification"],
                    "correctConcepts": result["correct_concepts"],
                    "missingConcepts": result["missing_concepts"],
                    "incorrectConcepts": result["incorrect_concepts"],
                    "textbookSupport": result["textbook_support"],
                    "feedback": result["feedback"],
                    "matchType": match["match_type"],
                    "matchSimilarity": match["similarity"]
                })

                print(
                    f"Q{question_number}: {marks}/{maximum} marks"
                )

            # 9. Calculate the final score.
            percentage = (
                round((obtained_marks / total_marks) * 100, 2)
                if total_marks
                else 0
            )

            matched_key_numbers = {
                match["key"]["question_number"]
                for match in matches
            }

            matched_student_numbers = {
                match["student"]["question_number"]
                for match in matches
            }

            details = {
                "studentName": student_name,
                "studentUSN": student_usn,
                "textbookId": textbook_id,
                "textbookFilename": textbook["filename"],
                "obtainedMarks": obtained_marks,
                "totalMarks": total_marks,
                "percentage": percentage,
                "questions": question_results,
                "matchedQuestions": len(matches),
                "missingAnswerKeyQuestions": sorted(
                    {
                        q["question_number"]
                        for q in questions
                    } - matched_key_numbers
                ),
                "unmatchedStudentQuestions": sorted(
                    set(student_questions.keys())
                    - matched_student_numbers
                )
            }

            # 10. Save evaluation history.
            with sqlite3.connect(DATABASE_PATH) as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO evaluations (
                        student_name,
                        student_usn,
                        obtained_marks,
                        total_marks,
                        percentage,
                        details
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        student_name,
                        student_usn,
                        obtained_marks,
                        total_marks,
                        percentage,
                        json.dumps(details)
                    )
                )

                evaluation_id = cursor.lastrowid

            print(
                f"\nEvaluation complete: "
                f"{obtained_marks}/{total_marks} ({percentage}%)"
            )

            return {
                "success": True,
                "evaluationId": evaluation_id,
                **details
            }

    except HTTPException:
        raise

    except Exception as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(exc)}"
        )


# =========================================================
# EVALUATION HISTORY
# =========================================================
# =========================================================
# EVALUATION HISTORY
# =========================================================


@app.get("/api/evaluations")
def get_evaluation_history():
    """Return saved evaluations, newest first."""

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT
                id,
                student_name,
                student_usn,
                obtained_marks,
                total_marks,
                percentage,
                created_at
            FROM evaluations
            ORDER BY id DESC
        """).fetchall()

    return {
        "evaluations": [
            dict(row)
            for row in rows
        ]
    }


@app.get("/api/evaluations/{evaluation_id}")
def get_evaluation(evaluation_id: int):
    """Return the full result for one saved evaluation."""

    with sqlite3.connect(
        DATABASE_PATH
    ) as connection:

        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT *
            FROM evaluations
            WHERE id = ?
            """,
            (evaluation_id,)
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail="Evaluation not found."
        )

    return {
        "id": row["id"],
        "createdAt": row["created_at"],
        **json.loads(row["details"])
    }
