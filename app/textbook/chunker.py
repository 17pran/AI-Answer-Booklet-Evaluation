import re


# Recognizes headings such as:
# Chapter 1 Introduction
# CHAPTER 2
# Chapter 3: Software Processes
# Chapter 4 - Requirements Engineering

CHAPTER_PATTERN = re.compile(
    r"^\s*CHAPTER\s+(\d+|[IVXLCDM]+)\b"
    r"(?:\s*[:.\-–—]?\s*(.*))?$",
    re.IGNORECASE
)

# Also recognizes headings such as:
# 1 Introduction
# 2 Software Processes
#
# Used cautiously near the beginning of a page.

NUMBERED_CHAPTER_PATTERN = re.compile(
    r"^\s*(\d{1,2})\s+"
    r"([A-Z][A-Za-z0-9 ,&()'/-]{2,90})\s*$"
)

SECTION_PATTERN = re.compile(
    r"^\s*\d+(?:\.\d+)+\s+.+$"
)


def detect_heading(text):
    """
    Detect an explicit chapter heading in a block of text.

    Returns:
        Heading string if found, otherwise None.
    """

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Explicit CHAPTER headings can be checked throughout
    # the supplied block.
    for line in lines:

        match = CHAPTER_PATTERN.match(line)

        if match:

            number = match.group(1)
            title = (match.group(2) or "").strip()

            if title:
                return f"Chapter {number}: {title}"

            return f"Chapter {number}"

    # Look for a numbered chapter heading near the top.
    # This avoids treating every numbered section as a chapter.
    for line in lines[:12]:

        match = NUMBERED_CHAPTER_PATTERN.match(line)

        if match:
            return line.strip()

    return None


def _is_chapter_heading(line):
    """
    Return True for an explicit chapter heading.
    """

    return bool(CHAPTER_PATTERN.match(line.strip()))


def _split_long_text(text, max_chars):
    """
    Split unusually long paragraphs into smaller pieces.

    Prefer sentence boundaries when possible.
    """

    text = text.strip()

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    pieces = []
    current = ""

    for sentence in sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        if len(sentence) > max_chars:

            if current:
                pieces.append(current)
                current = ""

            # Fall back to character splitting for a single
            # exceptionally long sentence.
            for start in range(0, len(sentence), max_chars):
                pieces.append(
                    sentence[start:start + max_chars]
                )

            continue

        if current and len(current) + len(sentence) + 1 > max_chars:

            pieces.append(current)

            current = sentence

        else:

            current = (
                f"{current} {sentence}".strip()
            )

    if current:
        pieces.append(current)

    return pieces


def chunk_text(
    pages,
    chunk_size=1800,
    overlap=200
):
    """
    Create chapter-wise textbook chunks.

    Args:
        pages:
            Output from extract_text().

        chunk_size:
            Maximum approximate chunk length in characters.

        overlap:
            Shared context between consecutive chunks
            within the same chapter.

    Returns:
        List of dictionaries containing:

        chunk_id
        chapter
        section
        start_page
        end_page
        text
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller "
            "than chunk_size"
        )

    chapters = []

    current_chapter = "Introduction / Unclassified"
    current_section = "General"

    current_lines = []
    current_pages = []

    def save_chapter():

        if not current_lines:
            return

        chapters.append({
            "chapter": current_chapter,
            "lines": list(current_lines),
            "pages": list(current_pages)
        })

    for page_data in pages:

        page_number = page_data["page"]

        lines = page_data["text"].splitlines()

        # Detect a chapter heading near the beginning of
        # the page. Explicit chapter headings can also
        # appear later on the page.
        heading_index = None
        heading_text = None

        for index, line in enumerate(lines):

            stripped = line.strip()

            if not stripped:
                continue

            explicit_match = CHAPTER_PATTERN.match(stripped)

            # An explicit CHAPTER heading can occur anywhere.
            # A plain numbered heading is only trusted near
            # the beginning of a page.
            numbered_match = (
                NUMBERED_CHAPTER_PATTERN.match(stripped)
                if index < 12
                else None
            )

            if explicit_match or numbered_match:

                heading_index = index

                if explicit_match:

                    number = explicit_match.group(1)
                    title = (
                        explicit_match.group(2) or ""
                    ).strip()

                    heading_text = (
                        f"Chapter {number}: {title}"
                        if title
                        else f"Chapter {number}"
                    )

                else:

                    heading_text = stripped

                break

        if heading_text:

            # Finish the previous chapter before starting
            # the next one.
            save_chapter()

            current_lines = []
            current_pages = []

            current_chapter = heading_text
            current_section = "General"

            # Exclude the chapter heading itself.
            lines = lines[heading_index + 1:]

        for line in lines:

            stripped = line.strip()

            if not stripped:
                current_lines.append("")
                continue

            # Track section headings for additional context.
            if SECTION_PATTERN.match(stripped):
                current_section = stripped

            current_lines.append(stripped)
            current_pages.append(page_number)

    # Save the final chapter.
    save_chapter()

    chunks = []
    chunk_id = 0

    for chapter_data in chapters:

        chapter_name = chapter_data["chapter"]

        # Build paragraphs from the accumulated chapter text.
        raw_text = "\n".join(chapter_data["lines"])

        paragraphs = re.split(
            r"\n\s*\n+",
            raw_text
        )

        # Recover page positions in document order.
        chapter_pages = chapter_data["pages"]

        page_start = min(chapter_pages) if chapter_pages else 1
        page_end = max(chapter_pages) if chapter_pages else page_start

        pieces = []

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            pieces.extend(
                _split_long_text(
                    paragraph,
                    chunk_size
                )
            )

        current_text = ""

        for piece in pieces:

            if (
                current_text
                and len(current_text) + len(piece) + 1
                > chunk_size
            ):

                chunks.append({
                    "chunk_id": chunk_id,
                    "chapter": chapter_name,
                    "section": "General",
                    "start_page": page_start,
                    "end_page": page_end,
                    "text": current_text.strip()
                })

                chunk_id += 1

                # Overlap stays within the current chapter.
                previous_tail = current_text[-overlap:]

                current_text = (
                    f"{previous_tail}\n{piece}".strip()
                )

            else:

                current_text = (
                    f"{current_text}\n{piece}".strip()
                )

        if current_text:

            chunks.append({
                "chunk_id": chunk_id,
                "chapter": chapter_name,
                "section": "General",
                "start_page": page_start,
                "end_page": page_end,
                "text": current_text.strip()
            })

            chunk_id += 1

    return chunks
