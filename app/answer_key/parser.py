import re
import pymupdf


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_text(pdf_path: str):

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text("text").strip()

        if text:

            pages.append({
                "page": page_number,
                "text": text
            })

    document.close()

    return pages


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace(
        "\x00",
        ""
    )

    # Fix common PDF ligature issue
    text = text.replace(
        "ﬃ",
        "ffi"
    )

    text = text.replace(
        "ﬀ",
        "ff"
    )

    text = text.replace(
        "ﬁ",
        "fi"
    )

    text = text.replace(
        "ﬂ",
        "fl"
    )

    # Remove excessive spaces
    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    # Normalize excessive blank lines
    text = re.sub(
        r"\n\s*\n\s*\n+",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# PARSE MODEL ANSWERS
# ============================================================

def parse_questions(pages):
    """
    Parse a typed model-answer PDF.

    Expected format:

    Q1) What is Software Engineering?
    Answer:
    Software Engineering is ...

    Q2) What are the main phases of SDLC?
    Answer:
    The main phases are:
    1. Requirement Analysis
    2. System Design
    ...

    Only Q1), Q2), Q3)... are treated as questions.
    Numbered points inside answers are NOT treated as questions.
    """

    # --------------------------------------------------------
    # Combine all pages
    # --------------------------------------------------------

    full_text = ""

    for page in pages:

        full_text += (
            "\n"
            + page["text"]
        )

    full_text = clean_text(
        full_text
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Only detect Q followed by a number.
    #
    # Example:
    # Q1)
    # Q2)
    # Q10)
    #
    # This prevents:
    #
    # 1. Requirement Analysis
    # 2. System Design
    #
    # from becoming questions.
    # --------------------------------------------------------

    question_pattern = re.compile(
        r"(?im)"
        r"^\s*Q\s*(\d+)\s*[\)\.\:\-]\s*(.+?)"
        r"(?=\n|$)"
    )

    matches = list(
        question_pattern.finditer(
            full_text
        )
    )

    questions = []

    for index, match in enumerate(matches):

        question_number = int(
            match.group(1)
        )

        question_text = match.group(2).strip()

        # ----------------------------------------------------
        # Determine where answer content starts
        # ----------------------------------------------------

        content_start = match.end()

        if index + 1 < len(matches):

            content_end = matches[
                index + 1
            ].start()

        else:

            content_end = len(
                full_text
            )

        content = full_text[
            content_start:content_end
        ].strip()

        # ----------------------------------------------------
        # Find Answer:
        # ----------------------------------------------------

        answer_match = re.search(
            r"(?im)"
            r"^\s*Answer\s*:\s*",
            content
        )

        if answer_match:

            answer_text = content[
                answer_match.end():
            ].strip()

        else:

            # If Answer: is missing, keep
            # the remaining content as answer.
            answer_text = content

        # ----------------------------------------------------
        # Remove accidental page markers
        # ----------------------------------------------------

        answer_text = re.sub(
            r"(?im)"
            r"^\s*Page\s+\d+\s*(?:of\s+\d+)?\s*$",
            "",
            answer_text
        ).strip()

        # ----------------------------------------------------
        # Add structured question
        # ----------------------------------------------------

        questions.append({

            "question_id":
                f"Q{question_number}",

            "question_number":
                question_number,

            "question":
                question_text,

            "model_answer":
                answer_text

        })

    return questions
