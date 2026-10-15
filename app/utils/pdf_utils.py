from pathlib import Path
from typing import List

import pymupdf
from PIL import Image


def pdf_to_images(
    pdf_path: str,
    output_dir: str,
    dpi: int = 300,
) -> List[str]:
    """
    Convert PDF pages to high-resolution PNG images for handwriting OCR.

    Args:
        pdf_path: Path to the input PDF.
        output_dir: Directory for generated images.
        dpi: Rendering resolution (default: 300 for optimal stroke clarity).

    Returns:
        Ordered list of generated image paths.
    """
    source = Path(pdf_path).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()

    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")

    if source.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF.")

    if not isinstance(dpi, int) or not 150 <= dpi <= 400:
        raise ValueError("DPI must be an integer between 150 and 400.")

    destination.mkdir(parents=True, exist_ok=True)
    image_paths = []

    try:
        with pymupdf.open(str(source)) as document:
            if document.page_count == 0:
                raise ValueError("The PDF contains no pages.")

            for page_index, page in enumerate(document):
                # 300 DPI ensures fine pen strokes are not broken up
                pixmap = page.get_pixmap(
                    dpi=dpi,
                    colorspace=pymupdf.csRGB,
                    alpha=False,
                    annots=True,
                )

                image_path = destination / f"page_{page_index + 1:04d}.png"
                pixmap.save(str(image_path))

                # Verify image integrity
                with Image.open(image_path) as img:
                    img.verify()

                image_paths.append(str(image_path))

    except Exception:
        # Clean up partial files if conversion encounters an error
        for image_path in image_paths:
            Path(image_path).unlink(missing_ok=True)
        raise

    return image_paths
