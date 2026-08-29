"""
Legacy preprocessing module.

The handwriting OCR engine now performs its own
multi-stage preprocessing inside ocr_engine.py.
"""

from pathlib import Path


def preprocess_image(image_path: str):

    path = Path(image_path)

    if not path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    return str(path)
