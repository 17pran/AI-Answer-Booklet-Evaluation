from pathlib import Path
import pymupdf


def pdf_to_images(pdf_path: str, output_dir: str, dpi: int = 200):
    """
    Convert every PDF page into a PNG image.

    Returns:
        List of generated image paths.
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    document = pymupdf.open(pdf_path)

    image_paths = []

    zoom = dpi / 72
    matrix = pymupdf.Matrix(zoom, zoom)

    for page_number, page in enumerate(document, start=1):

        pixmap = page.get_pixmap(
            matrix=matrix,
            alpha=False
        )

        output_path = (
            output_dir /
            f"{pdf_path.stem}_page_{page_number}.png"
        )

        pixmap.save(output_path)

        image_paths.append(str(output_path))

    document.close()

    return image_paths


def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from a digital PDF.

    Returns:
        List of dictionaries containing page and text.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        pages.append({
            "page": page_number,
            "text": text.strip()
        })

    document.close()

    return pages
