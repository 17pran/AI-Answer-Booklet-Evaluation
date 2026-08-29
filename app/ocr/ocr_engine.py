import os
import re
import cv2
import torch
import easyocr
import numpy as np

from pathlib import Path

from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)


# ============================================================
# HYBRID HANDWRITING OCR - LEVEL 3
# ============================================================

class HandwritingOCR:

    def __init__(self):

        print("==============================================")
        print("INITIALIZING HYBRID HANDWRITING OCR")
        print("==============================================")

        # ====================================================
        # TrOCR - PRIMARY HANDWRITING OCR
        # ====================================================

        print("Loading TrOCR handwriting model...")

        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"TrOCR device: {self.device}")

        # ====================================================
        # EasyOCR - used for BOTH line detection AND fallback
        # recognition. The CRAFT-based detector inside EasyOCR
        # is much more reliable at finding word/line boxes than
        # manual contour morphology, so we use it to drive line
        # ordering, then hand line crops to TrOCR as before.
        # ====================================================

        print("Loading EasyOCR (detector + fallback recognizer)...")

        self.easyocr = easyocr.Reader(
            ["en"],
            gpu=torch.cuda.is_available()
        )

        print("EasyOCR loaded.")
        print("==============================================")

    # ========================================================
    # IMAGE ENHANCEMENT
    # ========================================================
    #
    # NOTE: this is now used only to produce a *cleaned crop*
    # for recognition (TrOCR / EasyOCR fallback). Detection is
    # done separately on a lightly-processed image, because
    # aggressive adaptive thresholding thins strokes and hurts
    # EasyOCR's CRAFT detector.
    # ========================================================

    def enhance_image(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Reduce noise
        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        # Improve contrast
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        # Adaptive threshold
        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            15
        )

        return binary

    # ========================================================
    # DESKEW
    # ========================================================
    #
    # Real photographed/scanned sheets are rarely perfectly
    # aligned. A rotated page breaks the y-band line clustering
    # (which assumes roughly horizontal lines) and hurts TrOCR
    # accuracy. This estimates the skew angle from the ink mask
    # and rotates it back straight, but bails out safely if the
    # page is too sparse to estimate reliably or the detected
    # angle is implausibly large (usually noise, not real skew).
    # ========================================================

    def deskew_image(self, image_bgr):

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        coords = np.column_stack(
            np.where(thresh > 0)
        )

        # Not enough ink on the page to estimate an angle safely
        # (e.g. a near-blank page) — skip deskewing.
        if coords.shape[0] < 200:
            return image_bgr

        angle = cv2.minAreaRect(coords)[-1]

        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Real handwritten-page skew is rarely more than a few
        # degrees. A larger "angle" almost always means minAreaRect
        # latched onto noise rather than true page rotation, so we
        # leave the image untouched rather than risk making it worse.
        if abs(angle) > 12:
            return image_bgr

        (h, w) = image_bgr.shape[:2]
        center = (w // 2, h // 2)

        rotation_matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image_bgr,
            rotation_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    # ========================================================
    # REMOVE RULED / MARGIN LINES
    # ========================================================
    #
    # Real (and synthetic) answer sheets have horizontal ruling
    # and a vertical red margin line. These get picked up as
    # spurious boxes by any detector, so we inpaint them out
    # before detection.
    # ========================================================

    def remove_ruled_lines(self, image_bgr):

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        thresh = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        horiz_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (40, 1)
        )

        horiz_lines = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            horiz_kernel,
            iterations=2
        )

        vert_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, 40)
        )

        vert_lines = cv2.morphologyEx(
            thresh,
            cv2.MORPH_OPEN,
            vert_kernel,
            iterations=2
        )

        mask = cv2.bitwise_or(
            horiz_lines,
            vert_lines
        )

        # Thicken the mask slightly so inpainting fully covers
        # anti-aliased line edges too.
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )

        mask = cv2.dilate(
            mask,
            dilate_kernel,
            iterations=1
        )

        cleaned = cv2.inpaint(
            image_bgr,
            mask,
            3,
            cv2.INPAINT_TELEA
        )

        return cleaned

    # ========================================================
    # LINE DETECTION - EasyOCR (CRAFT) BASED
    # ========================================================
    #
    # Replaces the old contour/morphology based detect_lines().
    #
    # Steps:
    #   1. Run EasyOCR's detector to get word-level boxes.
    #   2. Cluster words into lines by y-center proximity
    #      (a proper y-band clustering against the line's
    #      running average, not just "compare to previous box").
    #   3. Sort lines top -> bottom.
    #   4. Sort words left -> right within each line.
    #   5. Return one padded bounding box per line, in reading
    #      order.
    # ========================================================

    def detect_lines_easyocr(
        self,
        image_bgr,
        header_cutoff_ratio=None
    ):

        results = self.easyocr.readtext(
            image_bgr,
            detail=1,
            paragraph=False
        )

        if not results:
            return []

        height, width = image_bgr.shape[:2]

        # FIX 1 - SKIP FIXED HEADER FIELD
        # The "Reg/Roll No / Subject / Name" line is a printed
        # form field, not the student's handwritten answer.
        # Anything above header_cutoff_ratio * page height is
        # dropped before it ever reaches OCR.
        header_cutoff_y = (
            height * header_cutoff_ratio
            if header_cutoff_ratio else -1
        )

        words = []

        for bbox, text, confidence in results:

            xs = [point[0] for point in bbox]
            ys = [point[1] for point in bbox]

            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)

            box_w = x2 - x1
            box_h = y2 - y1

            # FIX 3 - DROP TINY / NOISE BOXES
            # Ruled-line remnants, inpainting artifacts and stray
            # specks show up as very small boxes (a few px). Real
            # handwritten words are never this small.
            if box_w < 15 or box_h < 10:
                continue

            if y2 < header_cutoff_y:
                continue

            words.append({
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "yc": (y1 + y2) / 2,
                "h": max(y2 - y1, 1)
            })

        if not words:
            return []

        # Sort by vertical center first so nearby words end up
        # adjacent before clustering.
        words.sort(key=lambda w: w["yc"])

        lines = []

        for w in words:

            placed = False

            for line in lines:

                # Compare against the line's RUNNING average
                # y-center/height, not just the previous word.
                # This is what makes clustering robust to a
                # single stray/short word breaking the line.
                if abs(w["yc"] - line["yc"]) < max(line["h"], w["h"]) * 0.65:

                    line["words"].append(w)

                    n = len(line["words"])

                    line["yc"] = (
                        sum(x["yc"] for x in line["words"]) / n
                    )

                    line["h"] = (
                        sum(x["h"] for x in line["words"]) / n
                    )

                    placed = True
                    break

            if not placed:

                lines.append({
                    "yc": w["yc"],
                    "h": w["h"],
                    "words": [w]
                })

        # FIX 4 - STABLE TOP -> BOTTOM ORDER
        # Sort by y-center first, then by the topmost word's y1
        # as a tiebreaker. Two lines can end up with near-identical
        # yc (e.g. a short line next to a tall one) — the y1
        # tiebreak keeps those in true visual order instead of
        # flipping unpredictably.
        for line in lines:
            line["min_y1"] = min(w["y1"] for w in line["words"])

        lines.sort(key=lambda line: (round(line["yc"] / 8), line["min_y1"]))

        line_boxes = []

        for line in lines:

            # Left -> right within the line
            ordered_words = sorted(
                line["words"],
                key=lambda w: w["x1"]
            )

            x1 = min(w["x1"] for w in ordered_words)
            y1 = min(w["y1"] for w in ordered_words)
            x2 = max(w["x2"] for w in ordered_words)
            y2 = max(w["y2"] for w in ordered_words)

            pad_x = 10
            pad_y = 8

            box = (
                max(0, int(x1 - pad_x)),
                max(0, int(y1 - pad_y)),
                min(width, int(x2 + pad_x)),
                min(height, int(y2 + pad_y))
            )

            line_boxes.append(box)

        return line_boxes

    # ========================================================
    # DIAGRAM REGION DETECTION
    # ========================================================
    #
    # Feeding a diagram straight into TrOCR produces hallucinated
    # text, since TrOCR always tries to output SOME text for
    # whatever crop it's given. Instead: find ink on the page that
    # isn't explained by any detected text-line box — that's very
    # likely a diagram/sketch — group it into regions, save each
    # region as its own image, and exclude those areas from the
    # text OCR pass entirely.
    #
    # This is a heuristic, not a trained layout model. It works
    # well for "diagram in its own space, text elsewhere" answer
    # sheets. It will NOT reliably separate text that is written
    # directly on top of / crammed against diagram lines.
    # ========================================================

    def detect_diagram_regions(
        self,
        image_bgr,
        line_boxes,
        min_region_area_ratio=0.006
    ):

        height, width = image_bgr.shape[:2]
        page_area = height * width

        gray = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2GRAY
        )

        ink_mask = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        # Mask out everything already claimed by a detected
        # text line (dilated a bit so we don't pick up the
        # edges/serifs of real handwriting as "diagram ink").
        text_mask = np.zeros_like(ink_mask)

        for (x1, y1, x2, y2) in line_boxes:
            cv2.rectangle(
                text_mask,
                (x1, y1),
                (x2, y2),
                255,
                thickness=-1
            )

        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (9, 9)
        )

        text_mask = cv2.dilate(
            text_mask,
            dilate_kernel,
            iterations=1
        )

        remaining_ink = cv2.bitwise_and(
            ink_mask,
            cv2.bitwise_not(text_mask)
        )

        # Close small gaps so a diagram's separate strokes
        # (box edges, arrows, labels) merge into one region.
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (25, 25)
        )

        closed = cv2.morphologyEx(
            remaining_ink,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = page_area * min_region_area_ratio

        diagram_boxes = []

        for contour in contours:

            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area < min_area:
                continue

            pad = 12

            box = (
                max(0, x - pad),
                max(0, y - pad),
                min(width, x + w + pad),
                min(height, y + h + pad)
            )

            diagram_boxes.append(box)

        return diagram_boxes

    def _box_overlap_ratio(self, box, other):
        """Fraction of `box`'s area that overlaps `other`."""

        x1 = max(box[0], other[0])
        y1 = max(box[1], other[1])
        x2 = min(box[2], other[2])
        y2 = min(box[3], other[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        inter_area = (x2 - x1) * (y2 - y1)
        box_area = max(
            (box[2] - box[0]) * (box[3] - box[1]),
            1
        )

        return inter_area / box_area

    def save_diagram_crops(
        self,
        image_bgr,
        diagram_boxes,
        image_path,
        page_number
    ):
        """Save each diagram region as its own image, next to the
        source page image, in a 'diagrams' sub-folder. Returns the
        list of saved file paths (for logging / including in
        results if you want to reference them later)."""

        if not diagram_boxes:
            return []

        source_dir = Path(image_path).parent
        diagrams_dir = source_dir / "diagrams"
        diagrams_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for i, (x1, y1, x2, y2) in enumerate(diagram_boxes, start=1):

            crop = image_bgr[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            out_path = (
                diagrams_dir /
                f"page{page_number}_diagram{i}.png"
            )

            cv2.imwrite(str(out_path), crop)
            saved_paths.append(str(out_path))

        return saved_paths

    # ========================================================
    # TrOCR
    # ========================================================

    def trocr(self, image):

        # Grayscale → RGB
        if len(image.shape) == 2:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_GRAY2RGB
            )

        else:

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

        pil_image = Image.fromarray(rgb)

        pixel_values = self.processor(
            images=pil_image,
            return_tensors="pt"
        ).pixel_values

        pixel_values = pixel_values.to(
            self.device
        )

        with torch.no_grad():

            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=80,
                num_beams=4
            )

        text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        return text.strip()

    # ========================================================
    # EasyOCR FALLBACK
    # ========================================================

    def easyocr_fallback(self, image):

        results = self.easyocr.readtext(
            image,
            detail=1,
            paragraph=False
        )

        if not results:
            return "", 0.0

        texts = []
        confidences = []

        for bbox, text, confidence in results:

            text = text.strip()

            if confidence < 0.35:
                continue

            if len(text) < 2:
                continue

            texts.append(text)
            confidences.append(confidence)

        if not texts:
            return "", 0.0

        final_text = " ".join(texts)

        average_confidence = (
            sum(confidences) /
            len(confidences)
        )

        return final_text, average_confidence

    # ========================================================
    # TEXT QUALITY CHECK
    # ========================================================

    def is_bad_text(self, text):

        if not text:
            return True

        text = text.strip()

        if len(text) < 2:
            return True

        # Alphanumeric ratio
        alphanumeric = sum(
            c.isalnum()
            for c in text
        )

        ratio = (
            alphanumeric /
            max(len(text), 1)
        )

        if ratio < 0.35:
            return True

        # Repeated words
        words = text.split()

        if len(words) >= 3:

            unique_ratio = (
                len(set(words)) /
                len(words)
            )

            if unique_ratio < 0.35:
                return True

        # Known garbage
        garbage_patterns = [
            "not logged in",
            "talk contributions",
            "create account",
            "log in",
            "withdrawn",
            "commonwealth"
        ]

        lowered = text.lower()

        for pattern in garbage_patterns:

            if pattern in lowered:
                return True

        return False

    # ========================================================
    # PROCESS SINGLE LINE
    # ========================================================

    def process_line(
        self,
        line_image,
        line_number
    ):

        # Resize
        resized = cv2.resize(
            line_image,
            None,
            fx=2.5,
            fy=2.5,
            interpolation=cv2.INTER_CUBIC
        )

        # ====================================================
        # PRIMARY OCR → TrOCR
        # ====================================================

        trocr_text = self.trocr(
            resized
        )

        print(
            f"[Line {line_number}] "
            f"TrOCR: {trocr_text}"
        )

        # ====================================================
        # QUALITY CHECK
        # ====================================================

        if not self.is_bad_text(
            trocr_text
        ):

            return {
                "text": trocr_text,
                "source": "TrOCR",
                "confidence": None
            }

        # ====================================================
        # FALLBACK → EasyOCR
        # ====================================================

        print(
            f"[Line {line_number}] "
            f"TrOCR rejected → EasyOCR"
        )

        easy_text, confidence = (
            self.easyocr_fallback(
                resized
            )
        )

        print(
            f"[Line {line_number}] "
            f"EasyOCR: {easy_text} "
            f"(confidence={confidence:.2f})"
        )

        if not self.is_bad_text(
            easy_text
        ):

            return {
                "text": easy_text,
                "source": "EasyOCR",
                "confidence": confidence
            }

        return {
            "text": "",
            "source": "Rejected",
            "confidence": 0.0
        }

    # ========================================================
    # PROCESS SINGLE PAGE
    # ========================================================

    # ========================================================
    # FIX 2 - QUESTION NUMBER CLEANUP
    # ========================================================
    #
    # OCR frequently misreads "Q1)", "Q4)" etc as "a)", "QW",
    # "QSD", "QMJ", "(Q6)" and similar. Since question numbers
    # are guaranteed sequential in a real answer script, we
    # detect anything that LOOKS like a question-number line
    # (short line, starts with an optional bracket + a token
    # containing "Q" or a lone letter, followed by ")" ) and
    # rewrite it with a running counter instead of trusting the
    # raw OCR text.
    # ========================================================

    # Every question prompt in this template starts with one of
    # these verbs (see the question bank). OCR mangles the "Qn)"
    # marker in all kinds of ways ("a)", "QW", "QSD", "(Q6)",
    # "QMJ"...) but it never mangles the verb that follows nearly
    # as badly, so we anchor on the verb instead of the marker.
    QUESTION_VERBS = {
        "define", "explain", "differentiate",
        "write", "describe", "discuss", "list"
    }

    def fix_question_numbers(self, lines, start_counter=0):

        counter = start_counter

        fixed_lines = []

        for line in lines:

            text = line["text"]
            words = text.split()

            verb_idx = None

            # The garbled marker is at most a few tokens
            # ("(", "Q2", ")") so only look in the first 4 words.
            for i, w in enumerate(words[:4]):

                cleaned = re.sub(r"[^a-zA-Z]", "", w).lower()

                if cleaned in self.QUESTION_VERBS:
                    verb_idx = i
                    break

            if verb_idx is not None:

                counter += 1

                rest = " ".join(words[verb_idx:])

                line = dict(line)
                line["text"] = f"Q{counter}) {rest}"
                line["source"] = line.get("source", "") + "+QFix"

            fixed_lines.append(line)

        return fixed_lines, counter

    def process_page(self, image_path, page_number=None, q_counter_start=0):

        print("\n==============================================")
        print(
            f"Processing page: {image_path}"
        )
        print("==============================================")

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                "ERROR: Could not read image."
            )

            return [], q_counter_start

        # ====================================================
        # DESKEW (real photographed/scanned pages are rarely
        # perfectly straight; do this before anything else)
        # ====================================================

        image = self.deskew_image(image)

        # ====================================================
        # REMOVE RULED LINES (helps both detection & recognition)
        # ====================================================

        delined = self.remove_ruled_lines(
            image
        )

        # ====================================================
        # LINE DETECTION - EasyOCR/CRAFT, on the delined image.
        # Detection deliberately does NOT run on the hard
        # adaptive-threshold binary — thresholding thins strokes
        # and hurts CRAFT's detector.
        # ====================================================

        # Header field only exists on page 1 in this template.
        # Anything in the top ~6% of the page there is the
        # printed Reg No / Subject / Name line, not an answer.
        boxes = self.detect_lines_easyocr(
            delined,
            header_cutoff_ratio=0.06 if page_number == 1 else None
        )

        print(
            f"Detected {len(boxes)} "
            f"lines (EasyOCR/CRAFT detector)."
        )

        # ====================================================
        # DIAGRAM DETECTION - find ink not explained by any
        # text line, save it separately, and drop any text-line
        # box that mostly overlaps a diagram region (avoids
        # feeding diagram scribbles into TrOCR as "text").
        # ====================================================

        diagram_boxes = self.detect_diagram_regions(
            delined,
            boxes
        )

        if diagram_boxes:

            saved = self.save_diagram_crops(
                image,
                diagram_boxes,
                image_path,
                page_number
            )

            print(
                f"Detected {len(diagram_boxes)} diagram "
                f"region(s), saved to: {saved}"
            )

            boxes = [
                box for box in boxes
                if not any(
                    self._box_overlap_ratio(box, d) > 0.5
                    for d in diagram_boxes
                )
            ]

        # ====================================================
        # ENHANCEMENT (for recognition crops only)
        # ====================================================

        enhanced = self.enhance_image(
            delined
        )

        # ====================================================
        # OCR EACH LINE, IN DETECTED READING ORDER
        # ====================================================

        page_lines = []

        for line_number, box in enumerate(
            boxes,
            start=1
        ):

            x1, y1, x2, y2 = box

            line_crop = enhanced[
                y1:y2,
                x1:x2
            ]

            if line_crop.size == 0:
                continue

            result = self.process_line(
                line_crop,
                line_number
            )

            if result["text"]:

                page_lines.append({
                    "text": result["text"],
                    "source": result["source"],
                    "confidence": result["confidence"]
                })

        # FIX 2 applied here — remap garbled question-number
        # tokens (a), QW, QSD, (Q6)...) to a clean running Q#.
        # Counter continues across pages so numbering doesn't
        # reset at each page break.
        page_lines, q_counter_end = self.fix_question_numbers(
            page_lines,
            start_counter=q_counter_start
        )

        return page_lines, q_counter_end

    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================
    #
    # IMPORTANT:
    # test_ocr.py passes image_paths here.
    #
    # Therefore this function does NOT care about
    # the PDF filename.
    # ========================================================

    def process_document(self, image_paths):

        results = []

        q_counter = 0

        print(
            f"\nProcessing {len(image_paths)} page(s)..."
        )

        for page_number, image_path in enumerate(
            image_paths,
            start=1
        ):

            print(
                "\n=============================================="
            )

            print(
                f"PROCESSING PAGE {page_number}"
            )

            print(
                "=============================================="
            )

            page_lines, q_counter = self.process_page(
                image_path,
                page_number=page_number,
                q_counter_start=q_counter
            )

            results.append({
                "page": page_number,
                "filename": os.path.basename(
                    str(image_path)
                ),
                "lines": page_lines
            })

        return results

    # ========================================================
    # SAVE OCR RESULTS
    # ========================================================

    def save_text(
        self,
        results,
        output_file
    ):

        output_dir = os.path.dirname(
            output_file
        )

        if output_dir:

            os.makedirs(
                output_dir,
                exist_ok=True
            )

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            for result in results:

                file.write(
                    f"\n===== PAGE "
                    f"{result['page']} "
                    f"=====\n\n"
                )

                for line in result["lines"]:

                    file.write(
                        line["text"]
                    )

                    file.write("\n")

        print(
            f"\nOCR output saved to:"
            f"\n{output_file}"
        )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "This file is intended to be used "
        "through tests/test_ocr.py"
    )
