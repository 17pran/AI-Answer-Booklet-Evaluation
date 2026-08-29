import json
import requests
import re


class AnswerEvaluator:

    def __init__(
        self,
        model="qwen3:8b",
        ollama_url="http://localhost:11434/api/generate"
    ):

        self.model = model
        self.ollama_url = ollama_url

    # ============================================================
    # CALL LLM
    # ============================================================

    def _call_llm(self, prompt):

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json"
            },
            timeout=300
        )

        response.raise_for_status()

        data = response.json()

        return data["response"]

    # ============================================================
    # CLEAN LLM JSON
    # ============================================================

    def _clean_json_response(self, raw_response):

        raw_response = raw_response.strip()

        # Remove markdown code fences if Qwen adds them
        raw_response = re.sub(
            r"^```json\s*",
            "",
            raw_response,
            flags=re.IGNORECASE
        )

        raw_response = re.sub(
            r"^```\s*",
            "",
            raw_response
        )

        raw_response = re.sub(
            r"\s*```$",
            "",
            raw_response
        )

        return raw_response.strip()

    # ============================================================
    # EVALUATE ANSWER
    # ============================================================

    def evaluate(
        self,
        question,
        student_answer,
        model_answer,
        textbook_chunks=None,
        max_marks=5
    ):

        textbook_chunks = textbook_chunks or []

        # --------------------------------------------------------
        # Prepare textbook evidence
        # --------------------------------------------------------

        textbook_context = ""

        for i, chunk in enumerate(
            textbook_chunks,
            start=1
        ):

            textbook_context += (
                f"\n--- TEXTBOOK CHUNK {i} ---\n"
                f"Page: {chunk.get('page', 'Unknown')}\n"
                f"Similarity: "
                f"{chunk.get('similarity', 0):.3f}\n"
                f"{chunk.get('text', '')}\n"
            )

        # --------------------------------------------------------
        # MAIN PROMPT
        # --------------------------------------------------------

        prompt = f"""
You are an experienced university examiner evaluating a student's
Software Engineering examination answer.

You must grade FAIRLY and REASONABLY.

The student's answer was obtained using handwriting OCR.
Therefore, OCR may contain:
- spelling mistakes
- missing letters
- merged words
- broken sentences
- incorrect punctuation
- incorrectly recognized technical terms
- words that sound similar to the intended word

You MUST mentally correct obvious OCR errors when the intended
meaning is clear.

Do NOT penalize the student for grammar, spelling, handwriting,
punctuation, or OCR errors when the intended technical meaning
is understandable.

============================================================
QUESTION
============================================================

{question}

============================================================
MODEL ANSWER
============================================================

{model_answer}

============================================================
STUDENT ANSWER
============================================================

{student_answer}

============================================================
TEXTBOOK EVIDENCE
============================================================

{textbook_context}

============================================================
VERY IMPORTANT MARKING PRINCIPLES
============================================================

1. MARK THE STUDENT'S ANSWER, NOT ITS SIMILARITY TO THE MODEL ANSWER.

2. The model answer is a REFERENCE, not a checklist that the
   student must reproduce word-for-word.

3. A concise answer can receive high marks if it contains the
   important concepts required by the question.

4. Do NOT reduce marks simply because the student did not include
   every sentence, example, explanation, or detail from the model
   answer.

5. Only treat something as a "missing concept" if it is an
   IMPORTANT concept necessary to properly answer the question.

6. Minor details, optional examples, additional textbook points,
   and elaborations should NOT significantly reduce marks.

7. If the question asks for a definition, a correct definition
   should receive strong credit even if it is short.

8. If the question asks for differences, check whether the main
   difference is correctly stated. Do not require many examples
   when the student has already provided valid examples.

9. If the question asks for phases/steps, award credit for correctly
   identifying the required phases. Detailed descriptions of every
   phase are NOT mandatory unless the question explicitly asks
   for explanations.

10. If the question asks "why is it important", the student should
    receive credit for valid reasons even if they do not mention
    every possible benefit from the model answer.

11. Do NOT penalize the student for using different wording from
    the model answer.

12. Equivalent terminology should receive credit.

13. A concept should only be marked incorrect if the student
    actually states something technically wrong.

14. Do NOT mark unrelated concepts as incorrect unless they
    contradict the answer or substantially demonstrate
    misunderstanding.

15. Textbook evidence is supporting evidence. It is NOT a checklist.

16. Do not invent information that the student did not communicate.

17. If the student's answer is clearly correct but shorter than
    the model answer, prefer a HIGHER mark rather than a lower one.

18. Be especially tolerant of short but technically correct
    examination answers.

============================================================
MARKING GUIDELINE
============================================================

Use the following general scale for a {max_marks}-mark question:

5/5:
Excellent answer. Correct, directly addresses the question, and
contains nearly all important required concepts.

4/5:
Good answer. Correctly addresses the question and contains most
important concepts. Minor omissions or limited explanation are okay.

3/5:
Partially correct answer. Contains some important concepts but
has noticeable missing information or limited explanation.

2/5:
Limited answer. Contains a small amount of relevant knowledge but
misses major parts of the question.

1/5:
Very little relevant information.

0/5:
Completely incorrect, irrelevant, blank, or does not answer the
question.

IMPORTANT:
Do NOT automatically give 3/5 simply because the answer is shorter
than the model answer.

If the student's short answer correctly answers the question,
4/5 or 5/5 may be appropriate.

============================================================
QUESTION-SPECIFIC MARKING
============================================================

First determine what the question actually requires.

For example:

If the question is:

"What is Software Engineering?"

A concise correct definition can receive 4/5 or 5/5.

If the question is:

"What are the main phases of SDLC?"

Correctly listing all six phases should receive very high marks,
even if detailed explanations are brief.

If the question is:

"Explain the Waterfall Model."

The answer should receive strong credit if it correctly explains:
- sequential development
- major phases
- suitability for stable requirements
- difficulty when requirements change

It does NOT need to reproduce the full model answer.

If the question is:

"What is software testing? Why is it important?"

A student who:
- defines testing correctly
- explains that it finds defects/errors
- explains that it ensures requirements are met
- mentions quality/reliability
- lists valid testing types

has given a strong answer even if every testing type is not
explained individually.

If the question is:

"Differentiate between functional and non-functional requirements
with examples."

A student who correctly states:
- functional = what the system does
- non-functional = how the system performs / quality constraints
- gives valid examples

should receive strong marks.

============================================================
OCR INTERPRETATION
============================================================

First determine what the student's answer most likely means.

Correct obvious OCR mistakes internally.

For example:

"bcoz" -> "because"
"reqmt" -> "requirement"
"booz" -> "because"
"regards" -> "requirements" when the surrounding context clearly
indicates this meaning.

Do not penalize these OCR errors.

However, if a sentence is genuinely impossible to interpret,
do not invent a meaning.

============================================================
SCORING
============================================================

Before assigning marks, internally identify:

A. What did the student answer correctly?

B. What important parts of the question did the student answer?

C. What IMPORTANT concepts are genuinely missing?

D. Did the student state anything technically incorrect?

Then assign marks based on the overall quality.

Do not subtract marks mechanically for every missing detail.

A student does NOT need to reproduce the model answer to get 5/5.

============================================================
OUTPUT
============================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
    "marks": 0,
    "max_marks": {max_marks},
    "score_percentage": 0,
    "ocr_interpretation": "",
    "justification": "",
    "correct_concepts": [],
    "missing_concepts": [],
    "incorrect_concepts": [],
    "textbook_support": [],
    "feedback": ""
}}

Field meanings:

marks:
The marks awarded. Use integers or .5 increments only.

max_marks:
Maximum possible marks.

score_percentage:
marks / max_marks * 100.

ocr_interpretation:
A cleaned interpretation of what the student actually wrote.
Do not add concepts that were not present.

justification:
Explain the mark briefly and fairly.

correct_concepts:
Only list important concepts that the student correctly
communicated.

missing_concepts:
ONLY list important concepts that are genuinely necessary and
missing.

Do NOT list minor details or optional elaborations.

incorrect_concepts:
Only list concepts the student actually stated incorrectly.

textbook_support:
Mention relevant textbook evidence that supports the evaluation.
Do not use textbook content to invent missing student concepts.

feedback:
Give short, constructive feedback.
Do not make the student feel that they must reproduce the entire
model answer.

============================================================
FINAL FAIRNESS CHECK
============================================================

Before returning the score, ask yourself:

"Would a reasonable university examiner give this answer a good
mark based on what the student actually wrote?"

If YES, do not be unnecessarily harsh.

If the answer is substantially correct but concise, award 4 or 5
marks as appropriate.

If the answer is completely correct and directly answers the
question, 5/5 is allowed even if it is much shorter than the model
answer.

Return ONLY JSON.
"""

        # --------------------------------------------------------
        # CALL QWEN
        # --------------------------------------------------------

        raw_response = self._call_llm(prompt)

        print("\nRaw LLM response:")
        print(raw_response)

        # --------------------------------------------------------
        # PARSE JSON
        # --------------------------------------------------------

        try:

            cleaned_response = self._clean_json_response(
                raw_response
            )

            result = json.loads(
                cleaned_response
            )

        except (json.JSONDecodeError, TypeError):

            result = {
                "marks": 0,
                "max_marks": max_marks,
                "score_percentage": 0,
                "ocr_interpretation": "",
                "justification":
                    "LLM returned an invalid evaluation format.",
                "correct_concepts": [],
                "missing_concepts": [],
                "incorrect_concepts": [],
                "textbook_support": [],
                "feedback":
                    "Evaluation could not be parsed."
            }

        # --------------------------------------------------------
        # ENSURE REQUIRED FIELDS
        # --------------------------------------------------------

        result.setdefault(
            "marks",
            0
        )

        result.setdefault(
            "max_marks",
            max_marks
        )

        result.setdefault(
            "score_percentage",
            0
        )

        result.setdefault(
            "ocr_interpretation",
            ""
        )

        result.setdefault(
            "justification",
            ""
        )

        result.setdefault(
            "correct_concepts",
            []
        )

        result.setdefault(
            "missing_concepts",
            []
        )

        result.setdefault(
            "incorrect_concepts",
            []
        )

        result.setdefault(
            "textbook_support",
            []
        )

        result.setdefault(
            "feedback",
            ""
        )

        # --------------------------------------------------------
        # FORCE VALID MARK RANGE
        # --------------------------------------------------------

        try:

            marks = float(
                result["marks"]
            )

            # Keep marks within valid range
            marks = max(
                0,
                min(
                    max_marks,
                    marks
                )
            )

            # Round to nearest 0.5
            marks = round(
                marks * 2
            ) / 2

            result["marks"] = marks

            result["max_marks"] = max_marks

            result["score_percentage"] = round(
                (marks / max_marks) * 100,
                1
            )

        except (ValueError, TypeError):

            result["marks"] = 0

            result["max_marks"] = max_marks

            result["score_percentage"] = 0

        return result
