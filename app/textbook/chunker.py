import re


def detect_heading(text):
    """
    Try to detect chapter/section headings.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return None

    for line in lines[:10]:

        # Chapter 1, Chapter 2, etc.
        if re.match(
            r"^(chapter|CHAPTER)\s+\d+",
            line
        ):
            return line

        # Numbered sections such as:
        # 1.1 Introduction
        # 2.3 Software Processes
        if re.match(
            r"^\d+(\.\d+)*\s+[A-Za-z]",
            line
        ):
            return line

    return None


def chunk_text(
    pages,
    chunk_size=1200,
    overlap=200
):
    """
    Create context-aware textbook chunks.

    Each chunk keeps:
        - chunk ID
        - page number
        - chapter/section context
        - text

    chunk_size:
        Approximate number of characters.

    overlap:
        Characters shared between chunks.
    """

    chunks = []

    chunk_id = 0

    current_chapter = "Unknown"
    current_section = "Unknown"

    for page in pages:

        page_number = page["page"]
        text = page["text"].strip()

        if not text:
            continue

        # ----------------------------------------------------
        # Detect headings on this page
        # ----------------------------------------------------

        heading = detect_heading(text)

        if heading:

            if heading.lower().startswith("chapter"):

                current_chapter = heading
                current_section = "General"

            else:

                current_section = heading

        # ----------------------------------------------------
        # Split text into paragraphs
        # ----------------------------------------------------

        paragraphs = re.split(
            r"\n\s*\n",
            text
        )

        current_text = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # ------------------------------------------------
            # If adding paragraph exceeds chunk size
            # create current chunk first
            # ------------------------------------------------

            if (
                len(current_text) > 0
                and
                len(current_text) + len(paragraph)
                > chunk_size
            ):

                chunks.append({
                    "chunk_id": chunk_id,
                    "page": page_number,
                    "chapter": current_chapter,
                    "section": current_section,
                    "text": current_text.strip()
                })

                chunk_id += 1

                # Keep overlap from previous chunk
                current_text = (
                    current_text[-overlap:]
                    + "\n"
                    + paragraph
                )

            else:

                current_text += (
                    "\n"
                    + paragraph
                )

        # ----------------------------------------------------
        # Save remaining text from page
        # ----------------------------------------------------

        if current_text.strip():

            chunks.append({
                "chunk_id": chunk_id,
                "page": page_number,
                "chapter": current_chapter,
                "section": current_section,
                "text": current_text.strip()
            })

            chunk_id += 1

            current_text = ""

    return chunks
