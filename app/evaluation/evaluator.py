import json
import re
from typing import Any, Dict, List, Optional
import requests


class AnswerEvaluator:
    """
    Optimized AI Answer Evaluator using Qwen / Ollama.
    Features:
    - Persistent HTTP connection pooling for fast inference.
    - Strips <think> reasoning tokens from thinking-enabled models (e.g. Qwen 3).
    - Concise, high-signal prompt with strict OCR error tolerance.
    - Robust JSON extractor with regex fallback to prevent unfair 0-mark drops.
    """

    def __init__(
        self,
        model: str = "qwen3:8b",
        ollama_url: str = "http://localhost:11434/api/generate",
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.session = requests.Session()

    # ============================================================
    # CALL LLM WITH SESSION POOLING
    # ============================================================

    def _call_llm(self, prompt: str) -> str:
        try:
            response = self.session.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 1024,
                    },
                },
                timeout=180,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("response", "")
        except requests.RequestException as exc:
            print(f"Error calling LLM at {self.ollama_url}: {exc}")
            raise

    # ============================================================
    # CLEAN LLM JSON & STRIP THINKING TOKENS
    # ============================================================

    def _clean_json_response(self, raw_response: str) -> str:
        if not raw_response:
            return ""

        text = raw_response.strip()

        # 1. Strip reasoning/thinking tokens (<think>...</think>) from Qwen 3
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # 2. Strip markdown code blocks
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        # 3. Extract outermost JSON object if surrounded by extra commentary
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            text = match.group(0)

        # 4. Clean trailing commas before closing braces/brackets (common LLM bug)
        text = re.sub(r",\s*([\]}])", r"\1", text)

        return text.strip()

    # ============================================================
    # EVALUATE ANSWER
    # ============================================================

    def evaluate(
        self,
        question: str,
        student_answer: str,
        model_answer: str,
        textbook_chunks: Optional[List[Dict[str, Any]]] = None,
        max_marks: float = 5.0,
    ) -> Dict[str, Any]:
        textbook_chunks = textbook_chunks or []
        max_marks = float(max_marks)

        # --------------------------------------------------------
        # CHECK FOR MISSING STUDENT ANSWER
        # --------------------------------------------------------
        if not student_answer or not str(student_answer).strip():
            return {
                "marks": 0.0,
                "max_marks": max_marks,
                "score_percentage": 0.0,
                "ocr_interpretation": "",
                "justification": "The student answer is blank or was not detected.",
                "correct_concepts": [],
                "missing_concepts": ["Student answer not provided"],
                "incorrect_concepts": [],
                "textbook_support": [],
                "feedback": "No answer detected for this question.",
            }

        # --------------------------------------------------------
        # PREPARE TEXTBOOK CONTEXT (Top 3 most relevant)
        # --------------------------------------------------------
        textbook_context = ""
        for i, chunk in enumerate(textbook_chunks[:3], start=1):
            chunk_text = str(chunk.get("text", "")).strip()[:600]
            sim = chunk.get("similarity", 0)
            textbook_context += f"Evidence {i} (Similarity: {sim:.2f}):\n{chunk_text}\n\n"

        # --------------------------------------------------------
        # CONCISE, HIGH-SIGNAL EXAMINER PROMPT
        # --------------------------------------------------------
        prompt = f"""You are an expert university examiner evaluating a student's handwritten examination answer for Software Engineering.

EVALUATION PRINCIPLES:
1. FAIRNESS & OCR TOLERANCE: The student answer comes from handwriting OCR. Mentally correct obvious OCR spelling/word errors (e.g. "bcoz"->"because", "reqmt"->"requirement", merged words, punctuation). Grade their technical understanding, not their handwriting or OCR imperfections.
2. SUBSTANCE OVER WORDING: The Model Answer is a reference guide, NOT a verbatim checklist. Do not penalize students for using different wording, equivalent synonyms, or concise phrasing.
3. CONCISE ACCURACY: If a student provides a concise, accurate definition or correctly identifies the required steps/phases, award HIGH/FULL marks (4.5 or 5.0).
4. MARKS SCALE (out of {max_marks}):
   - {max_marks}/{max_marks}: Excellent. Directly addresses the question with core required concepts.
   - 4.0/5: Good. Correctly addresses question with minor omissions.
   - 3.0/5: Partially correct. Relevant knowledge but noticeable missing parts.
   - 1.5-2.0/5: Limited. Small amount of relevant knowledge.
   - 0/5: Completely incorrect or irrelevant.

QUESTION:
{question}

MODEL REFERENCE ANSWER:
{model_answer}

STUDENT'S HANDWRITTEN ANSWER:
{student_answer}

TEXTBOOK REFERENCE:
{textbook_context or 'N/A'}

INSTRUCTIONS:
Return a single valid JSON object with EXACTLY this structure:
{{
    "marks": <float between 0 and {max_marks}, increments of 0.5>,
    "max_marks": {max_marks},
    "score_percentage": <percentage 0 to 100>,
    "ocr_interpretation": "<clean reading of student meaning without OCR typos>",
    "justification": "<brief, fair justification of marks awarded>",
    "correct_concepts": ["<concept 1>", "<concept 2>"],
    "missing_concepts": ["<genuinely missing core concepts, if any>"],
    "incorrect_concepts": ["<any stated technical inaccuracies, if any>"],
    "textbook_support": ["<brief reference to supporting textbook points>"],
    "feedback": "<constructive feedback for the student>"
}}
"""

        # --------------------------------------------------------
        # CALL LLM & PARSE JSON
        # --------------------------------------------------------
        raw_response = self._call_llm(prompt)
        cleaned_json = self._clean_json_response(raw_response)

        result = {}
        try:
            result = json.loads(cleaned_json)
        except Exception as err:
            print(f"JSON parse error on LLM response: {err}")
            # Regex recovery fallback
            marks_match = re.search(r'"marks"\s*:\s*([0-9.]+)', raw_response)
            recovered_marks = float(marks_match.group(1)) if marks_match else 0.0
            just_match = re.search(r'"justification"\s*:\s*"([^"]+)"', raw_response)
            recovered_just = just_match.group(1) if just_match else "Evaluation completed."

            result = {
                "marks": recovered_marks,
                "max_marks": max_marks,
                "score_percentage": round((recovered_marks / max_marks) * 100, 1),
                "ocr_interpretation": student_answer,
                "justification": recovered_just,
                "correct_concepts": [],
                "missing_concepts": [],
                "incorrect_concepts": [],
                "textbook_support": [],
                "feedback": "Answer evaluated.",
            }

        # --------------------------------------------------------
        # SANITIZE & NORMALIZE RESULT
        # --------------------------------------------------------
        try:
            marks = float(result.get("marks", 0.0))
            marks = max(0.0, min(max_marks, marks))
            # Round to nearest 0.5
            marks = round(marks * 2) / 2
        except (ValueError, TypeError):
            marks = 0.0

        return {
            "marks": marks,
            "max_marks": max_marks,
            "score_percentage": round((marks / max_marks) * 100, 1) if max_marks else 0.0,
            "ocr_interpretation": result.get("ocr_interpretation") or student_answer,
            "justification": result.get("justification") or "Evaluated based on technical accuracy.",
            "correct_concepts": result.get("correct_concepts") or [],
            "missing_concepts": result.get("missing_concepts") or [],
            "incorrect_concepts": result.get("incorrect_concepts") or [],
            "textbook_support": result.get("textbook_support") or [],
            "feedback": result.get("feedback") or "Good attempt.",
        }

