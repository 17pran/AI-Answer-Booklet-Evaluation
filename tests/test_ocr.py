import os

from app.config import (
    ANSWER_SCRIPTS_DIR,
    PROCESSED_DIR
)

from app.utils.pdf_utils import pdf_to_images
from app.ocr.ocr_engine import HandwritingOCR


def main():

    # ============================================================
    # FIND PDF AUTOMATICALLY
    # ============================================================

    pdf_files = sorted(
        ANSWER_SCRIPTS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print("\nERROR:")
        print("No PDF found in:")
        print(ANSWER_SCRIPTS_DIR)

        print(
            "\nPlease place a handwritten answer PDF "
            "inside the answer_scripts folder."
        )

        return

    # Use the first PDF found
    pdf_path = pdf_files[0]

    print("\nStudent answer script detected:")
    print(pdf_path)

    # ============================================================
    # CLEAR OLD OCR IMAGES
    # ============================================================

    image_dir = (
        PROCESSED_DIR /
        "ocr_images"
    )

    image_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for file in image_dir.iterdir():

        if file.suffix.lower() in (
            ".png",
            ".jpg",
            ".jpeg"
        ):
            file.unlink()

    print("\nOld OCR images cleared.")

    # ============================================================
    # PDF → IMAGES
    # ============================================================

    print("\nConverting PDF to images...")

    image_paths = pdf_to_images(
        str(pdf_path),
        str(image_dir)
    )

    print(
        f"Created {len(image_paths)} images."
    )

    # ============================================================
    # OCR
    # ============================================================

    print("\nStarting handwriting OCR...")

    ocr = HandwritingOCR()

    results = ocr.process_document(
        image_paths
    )

    # ============================================================
    # SAVE
    # ============================================================

    output_file = (
        PROCESSED_DIR /
        "ocr_output.txt"
    )

    ocr.save_text(
        results,
        str(output_file)
    )

    # ============================================================
    # DISPLAY
    # ============================================================

    print(
        "\n=============================================="
    )
    print("OCR COMPLETE")
    print(
        "=============================================="
    )

    print(f"\nOutput:\n{output_file}")

    print("\nExtracted text:")
    print("----------------------------------------------")

    for page in results:

        print(
            f"\n===== PAGE {page['page']} =====\n"
        )

        for line in page["lines"]:
            print(line["text"])

    print(
        "\n----------------------------------------------"
    )

    print("OCR TEST FINISHED.")


if __name__ == "__main__":
    main()
