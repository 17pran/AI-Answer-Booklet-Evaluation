import pymupdf


def extract_text(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF page by page.

    Returns:
        [
            {
                "page": 1,
                "text": "..."
            },
            ...
        ]
    """

    pages = []

    with pymupdf.open(pdf_path) as document:

        for page_number, page in enumerate(document, start=1):

            text = page.get_text("text")

            if text:
                text = text.strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text
                })

    return pages
