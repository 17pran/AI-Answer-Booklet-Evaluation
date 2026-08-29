import re


def parse_answer_key(pages):
    """
    Parse answer-key PDF.

    Returns:
    [
        {
            "question_number": 12,
            "question": "...",
            "answer": "..."
        }
    ]
    """

    full_text = ""

    for page in pages:
        text = page.get("text", "")

        if text.strip():
            full_text += "\n" + text

    full_text = normalize_text(full_text)

    # ---------------------------------------------------------
    # Detect Q12), Q13), Q14), etc.
    # ---------------------------------------------------------

    question_pattern = re.compile(
        r"""
        (?:^|\n)
        \s*
        Q\s*(?P<number>\d+)
        \s*[\)\].:\-]?\s*
        (?P<block>.*?)
        (?=
            \n\s*Q\s*\d+\s*[\)\].:\-]?
            |
            \Z
        )
        """,
        re.IGNORECASE | re.DOTALL | re.VERBOSE
    )

    questions = []

    for match in question_pattern.finditer(full_text):

        number = int(match.group("number"))

        block = match.group("block").strip()

        # -----------------------------------------------------
        # Find Ans / Answer
        # -----------------------------------------------------

        answer_match = re.search(
            r"""
            \b
            (?:Ans(?:wer)?)
            \s*
            [\.\:\-]?
            \s*
            """,
            block,
            re.IGNORECASE | re.VERBOSE
        )

        if not answer_match:
            continue

        question_text = block[:answer_match.start()].strip()

        answer_text = block[answer_match.end():].strip()

        question_text = clean_text(question_text)
        answer_text = clean_text(answer_text)

        if not question_text or not answer_text:
            continue

        questions.append({
            "question_number": number,
            "question": question_text,
            "answer": answer_text
        })

    return questions


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    text = text.replace("\r", "\n")

    # PDF ligatures
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")

    # Fix common OCR/PDF artifacts
    text = text.replace("\u00a0", " ")

    return text


def clean_text(text):

    text = normalize_text(text)

    # Join words broken by newline
    text = re.sub(
        r"(?<=\w)\n(?=\w)",
        " ",
        text
    )

    # Convert remaining newlines to spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove spaces before punctuation
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    # Normalize spaces around brackets
    text = re.sub(
        r"\(\s+",
        "(",
        text
    )

    text = re.sub(
        r"\s+\)",
        ")",
        text
    )

    return text.strip()
