import time
from pathlib import Path

from app.config import ANSWER_SCRIPTS_DIR, PROCESSED_DIR
from app.ocr.ocr_engine import HandwritingOCR
from app.utils.pdf_utils import pdf_to_images


def main():
    pdf_files = sorted(ANSWER_SCRIPTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF found in: {ANSWER_SCRIPTS_DIR}")
        return

    pdf_path = pdf_files[0]
    print(f"\n==========================================")
    print(f"Testing OCR Pipeline on: {pdf_path.name}")
    print(f"==========================================")

    image_dir = PROCESSED_DIR / "ocr_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    print("\n[Step 1/3] Converting PDF to 300 DPI images...")
    conv_start = time.perf_counter()
    image_paths = pdf_to_images(
        str(pdf_path),
        str(image_dir),
        dpi=300,
    )
    conv_time = time.perf_counter() - conv_start

    if not image_paths:
        print("ERROR: No page images were generated.")
        return

    print(f"Pages converted: {len(image_paths)} in {conv_time:.2f}s")

    print("\n[Step 2/3] Initializing HandwritingOCR Engine...")
    ocr = HandwritingOCR(batch_size=8, use_easyocr_fallback=True)

    print("\n[Step 3/3] Running Recognition Pipeline...")
    results = ocr.process_document(image_paths)

    output_file = PROCESSED_DIR / "ocr_output.txt"
    ocr.save_text(results, str(output_file))

    # Calculate metrics
    all_lines = [line for page in results for line in page["lines"]]
    total_lines = len(all_lines)
    review_lines = sum(
        1 for line in all_lines if line.get("needs_review", False))
    avg_confidence = (
        sum(line.get("confidence", 0.0) for line in all_lines) / total_lines
        if total_lines > 0
        else 0.0
    )

    print("\n================ OCR TEST SUMMARY ================")
    print(f"File Tested:          {pdf_path.name}")
    print(f"Pages:                {len(results)}")
    print(f"Total Lines Extracted:{total_lines}")
    print(
        f"Lines Flagged Review: {review_lines} ({(review_lines / max(total_lines, 1)) * 100:.1f}%)")
    print(f"Average Confidence:   {avg_confidence * 100:.2f}%")
    print(f"Results Saved To:     {output_file}")
    print("==================================================")

    print("\nSample Extracted Lines (First 15 lines with confidence):")
    print("-" * 75)
    for line in all_lines[:15]:
        flag = "[!REVIEW]" if line.get("needs_review") else "[OK]     "
        src = f"({line.get('source', 'TrOCR')})"
        conf = f"{line.get('confidence', 0.0) * 100:.1f}%"
        print(f"{flag} {conf:>6} {src:<12} | {line.get('text', '')}")
    print("-" * 75)


if __name__ == "__main__":
    main()
