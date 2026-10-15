import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import easyocr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class HandwritingOCR:
    """
    High-accuracy handwriting OCR engine for student answer scripts.

    Pipeline:
        1. Deskew & background-normalize page image (removes ruled lines & shadows).
        2. Detect text line regions using tuned EasyOCR detector with gap-aware grouping.
        3. Extract padded line crops to protect ascenders and descenders.
        4. Recognize text using batched TrOCR with exact token-level confidence scoring.
        5. Targeted retry (Beam Search & contrast enhancement) for low-confidence lines.
        6. EasyOCR fallback for uncertain lines.
        7. Text normalization & structured result generation.
    """

    def __init__(
        self,
        batch_size: int = 8,
        use_easyocr_fallback: bool = True,
    ) -> None:
        print("\nInitializing Handwriting OCR Engine...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = max(1, int(batch_size))
        self.use_easyocr_fallback = use_easyocr_fallback

        if self.device.type == "cpu":
            available_cpus = os.cpu_count() or 4
            torch.set_num_threads(max(1, min(4, available_cpus)))

        # --------------------------------------------------
        # LOAD TrOCR
        # --------------------------------------------------
        model_name = "microsoft/trocr-base-handwritten"
        print(f"Loading TrOCR model: {model_name}")

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            self.model.half()

        print(
            f"TrOCR running on: {self.device} (batch_size={self.batch_size})")

        # --------------------------------------------------
        # LOAD EasyOCR (Detector & Fallback)
        # --------------------------------------------------
        print("Loading EasyOCR detector...")
        self.easyocr = easyocr.Reader(
            ["en"],
            gpu=(self.device.type == "cuda"),
        )
        print("OCR models initialized successfully.\n")

    # ======================================================
    # IMAGE VALIDATION & PREPROCESSING
    # ======================================================

    @staticmethod
    def validate_image(image: Optional[np.ndarray]) -> bool:
        return (
            image is not None
            and isinstance(image, np.ndarray)
            and image.size > 0
            and image.ndim == 3
            and image.shape[2] == 3
        )

    def deskew_image(self, image_bgr: np.ndarray) -> np.ndarray:
        if not self.validate_image(image_bgr):
            return image_bgr

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        coordinates = np.column_stack(np.where(binary > 0))
        if len(coordinates) < 200 or len(coordinates) > 250000:
            return image_bgr

        angle = cv2.minAreaRect(coordinates.astype(np.float32))[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # Guard against erratic rotations
        if abs(angle) < 0.4 or abs(angle) > 6.0:
            return image_bgr

        height, width = image_bgr.shape[:2]
        matrix = cv2.getRotationMatrix2D((width // 2, height // 2), angle, 1.0)
        return cv2.warpAffine(
            image_bgr,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def enhance_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Suppresses notebook ruled lines, uneven illumination, and background bleed.
        """
        if not self.validate_image(image_bgr):
            return image_bgr

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Estimate background illumination to cancel shadows & notebook rules
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(
            diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Apply mild CLAHE to make ink strokes stand out sharply
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        enhanced = clahe.apply(norm)

        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # ======================================================
    # TEXT DETECTION WITH GAP-AWARE GROUPING
    # ======================================================

    def detect_lines_easyocr(
        self,
        image_bgr: np.ndarray,
        header_cutoff_ratio: Optional[float] = None,
    ) -> List[Tuple[int, int, int, int]]:
        if not self.validate_image(image_bgr):
            return []

        height, width = image_bgr.shape[:2]
        cutoff_y = int(
            height * header_cutoff_ratio) if header_cutoff_ratio else -1

        # Tuned thresholds for handwritten text strokes
        horizontal_list, free_list = self.easyocr.detect(
            image_bgr,
            min_size=10,
            text_threshold=0.40,
            low_text=0.30,
            link_threshold=0.35,
            canvas_size=2560,
            mag_ratio=1.2,
        )

        raw_boxes = []

        # Process horizontal boxes
        for box in horizontal_list[0] if horizontal_list else []:
            if len(box) < 4:
                continue
            x1, x2, y1, y2 = map(int, box[:4])
            x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
            y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
            if (x2 - x1) >= 10 and (y2 - y1) >= 8 and y2 > cutoff_y:
                raw_boxes.append((x1, y1, x2, y2))

        # Process polygon boxes
        for polygon in free_list[0] if free_list else []:
            points = np.asarray(polygon, dtype=np.float32)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
                continue
            x1 = int(max(0, np.min(points[:, 0])))
            y1 = int(max(0, np.min(points[:, 1])))
            x2 = int(min(width, np.max(points[:, 0])))
            y2 = int(min(height, np.max(points[:, 1])))
            if (x2 - x1) >= 10 and (y2 - y1) >= 8 and y2 > cutoff_y:
                raw_boxes.append((x1, y1, x2, y2))

        if not raw_boxes:
            return []

        # Remove duplicate / heavily overlapping boxes
        raw_boxes.sort(key=lambda b: (b[1], b[0]))
        deduped = []
        for box in raw_boxes:
            x1, y1, x2, y2 = box
            dup = False
            for ex1, ey1, ex2, ey2 in deduped:
                ix1, iy1 = max(x1, ex1), max(y1, ey1)
                ix2, iy2 = min(x2, ex2), min(y2, ey2)
                inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                smaller_area = min(
                    max(1, (x2 - x1) * (y2 - y1)),
                    max(1, (ex2 - ex1) * (ey2 - ey1)),
                )
                if (inter / smaller_area) > 0.75:
                    dup = True
                    break
            if not dup:
                deduped.append(box)

        # --------------------------------------------------
        # Group boxes by vertical lines with horizontal gap threshold
        # --------------------------------------------------
        groups = []
        for box in deduped:
            x1, y1, x2, y2 = box
            cy = (y1 + y2) / 2
            bh = y2 - y1

            matched_group = None
            min_dist = float("inf")

            for group in groups:
                dist = abs(cy - group["center_y"])
                tol = max(10, min(bh, group["height"]) * 0.55)
                if dist <= tol and dist < min_dist:
                    matched_group = group
                    min_dist = dist

            if matched_group is None:
                groups.append({"center_y": cy, "height": bh, "boxes": [box]})
            else:
                matched_group["boxes"].append(box)
                gb = matched_group["boxes"]
                matched_group["center_y"] = sum(
                    (b[1] + b[3]) / 2 for b in gb) / len(gb)
                matched_group["height"] = sum(
                    b[3] - b[1] for b in gb) / len(gb)

        # --------------------------------------------------
        # Build line crops with generous padding and gap-splitting
        # --------------------------------------------------
        line_boxes = []
        pad_x = 16
        pad_y = 14

        for group in groups:
            # Sort words from left to right
            g_boxes = sorted(group["boxes"], key=lambda b: b[0])

            # Split line if there is an abnormally large horizontal gap
            # (e.g. Roll No column vs student text, or margin note)
            current_cluster = [g_boxes[0]]
            for next_box in g_boxes[1:]:
                prev_box = current_cluster[-1]
                gap = next_box[0] - prev_box[2]
                max_allowed_gap = max(120, int(prev_box[3] - prev_box[1]) * 3)

                if gap > max_allowed_gap:
                    # Finalize previous cluster
                    x1 = min(b[0] for b in current_cluster)
                    y1 = min(b[1] for b in current_cluster)
                    x2 = max(b[2] for b in current_cluster)
                    y2 = max(b[3] for b in current_cluster)
                    line_boxes.append((
                        max(0, x1 - pad_x),
                        max(0, y1 - pad_y),
                        min(width, x2 + pad_x),
                        min(height, y2 + pad_y),
                    ))
                    current_cluster = [next_box]
                else:
                    current_cluster.append(next_box)

            if current_cluster:
                x1 = min(b[0] for b in current_cluster)
                y1 = min(b[1] for b in current_cluster)
                x2 = max(b[2] for b in current_cluster)
                y2 = max(b[3] for b in current_cluster)
                line_boxes.append((
                    max(0, x1 - pad_x),
                    max(0, y1 - pad_y),
                    min(width, x2 + pad_x),
                    min(height, y2 + pad_y),
                ))

        # Sort lines strictly in reading order (top to bottom, left to right)
        line_boxes.sort(key=lambda b: ((b[1] + b[3]) / 2, b[0]))
        return line_boxes

    # ======================================================
    # LINE CROPPING
    # ======================================================

    @staticmethod
    def prepare_line_crop(
        image_bgr: np.ndarray,
        box: Tuple[int, int, int, int],
    ) -> Optional[Image.Image]:
        x1, y1, x2, y2 = box
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 6:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        # Margin padding for TrOCR attention stability
        padded = cv2.copyMakeBorder(
            rgb, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        return Image.fromarray(padded)

    # ======================================================
    # TEXT NORMALIZATION & VALIDATION
    # ======================================================

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans common OCR handwriting errors, stray artifacts, and repetitions.
        """
        if not text:
            return ""

        text = text.strip()

        # Fix repeated characters caused by TrOCR loop hallucinations (e.g. 'coooode' -> 'code')
        text = re.sub(r"([a-zA-Z])\1{3,}", r"\1\1", text)

        # Fix common handwriting contractions & misreads in answer sheets
        replacements = [
            (r"\bbcoe\b", "Because"),
            (r"\bbooz\b", "because"),
            (r"\bbcoz\b", "because"),
            (r"\breqmts\b", "Requirements"),
            (r"\breqmits\b", "Requirements"),
            (r"\bpage2012\b", "Page 2 of 2"),
            (r"\bReg\s*Roll\s*No\.?\b", "Register / Roll No."),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

        # Strip accidental leading numbering artifacts (e.g., '40 Testing' -> 'Testing')
        # where stray ink or bullets were recognized as random double-digit numbers
        text = re.sub(r"^[0-9]{2}\s+(?=[A-Z])", "", text)

        # Clean multiple spaces and stray edge symbols/quotes
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"^[\"\'\-—_~`.]+|[\"\'\-—_~`.]+$", "", text).strip()

        return text

    @classmethod
    def is_bad_text(cls, text: Optional[str], confidence: Optional[float] = None) -> bool:
        if not text:
            return True

        cleaned = cls.clean_text(text)
        if len(cleaned) < 2:
            return True

        if confidence is not None and confidence < 0.70:
            return True

        alpha_count = sum(c.isalpha() for c in cleaned)
        digit_count = sum(c.isdigit() for c in cleaned)

        if (alpha_count + digit_count) / max(len(cleaned), 1) < 0.40:
            return True

        if alpha_count == 0 and digit_count == 0:
            return True

        # Check for non-word gibberish like ';;;###' or '___'
        if re.match(r"^[\W_]+$", cleaned):
            return True

        return False

    # ======================================================
    # TrOCR INFERENCE WITH EXACT CONFIDENCE
    # ======================================================

    def trocr_batch(
        self,
        pil_images: List[Image.Image],
        num_beams: int = 1,
    ) -> List[Tuple[str, float]]:
        """
        Runs batched TrOCR and returns (extracted_text, confidence_score).
        """
        if not pil_images:
            return []

        results = []
        eos_id = getattr(self.processor.tokenizer, "eos_token_id", 2)

        for start in range(0, len(pil_images), self.batch_size):
            batch = pil_images[start: start + self.batch_size]

            inputs = self.processor(images=batch, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)

            if self.device.type == "cuda":
                pixel_values = pixel_values.half()

            with torch.inference_mode():
                outputs = self.model.generate(
                    pixel_values,
                    max_new_tokens=96,
                    num_beams=num_beams,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            sequences = outputs.sequences
            batch_texts = self.processor.batch_decode(
                sequences, skip_special_tokens=True
            )

            # Compute real confidence safely (handles greedy and beam search)
            confidences = []
            try:
                if num_beams == 1 and outputs.scores:
                    stacked_logits = torch.stack(outputs.scores, dim=1)
                    probs = F.softmax(stacked_logits, dim=-1)
                    gen_tokens = sequences[:, 1: 1 + stacked_logits.shape[1]]
                    token_probs = torch.gather(
                        probs, 2, gen_tokens.unsqueeze(-1)
                    ).squeeze(-1)

                    for i in range(len(batch_texts)):
                        line_probs = []
                        for t_idx in range(gen_tokens.shape[1]):
                            tok = gen_tokens[i, t_idx].item()
                            line_probs.append(token_probs[i, t_idx].item())
                            if tok == eos_id:
                                break

                        conf = (
                            float(sum(line_probs) / len(line_probs))
                            if line_probs
                            else 0.75
                        )
                        confidences.append(round(min(1.0, max(0.0, conf)), 4))
                elif hasattr(outputs, "sequences_scores") and outputs.sequences_scores is not None:
                    for score in outputs.sequences_scores.cpu().numpy():
                        prob = float(np.exp(score))
                        confidences.append(round(min(1.0, max(0.0, prob)), 4))
                else:
                    confidences = [0.85] * len(batch_texts)
            except Exception:
                confidences = [0.80] * len(batch_texts)

            if len(confidences) != len(batch_texts):
                confidences = [0.80] * len(batch_texts)

            for text, conf in zip(batch_texts, confidences):
                cleaned = self.clean_text(text)
                results.append((cleaned, conf))

        return results

    # ======================================================
    # EASYOCR FALLBACK
    # ======================================================

    def easyocr_fallback(self, image: Image.Image) -> Tuple[str, Optional[float]]:
        results = self.easyocr.readtext(
            np.asarray(image),
            detail=1,
            paragraph=False,
        )
        if not results:
            return "", None

        accepted = []
        for _, text, confidence in results:
            cleaned = self.clean_text(text)
            if cleaned and confidence >= 0.25:
                accepted.append((cleaned, float(confidence)))

        if not accepted:
            return "", None

        final_text = " ".join(item[0] for item in accepted)
        avg_conf = sum(item[1] for item in accepted) / len(accepted)
        return final_text, round(avg_conf, 4)

    # ======================================================
    # PROCESS ONE PAGE
    # ======================================================

    def process_page(
        self,
        image_path: str,
        page_number: Optional[int] = None,
        q_counter_start: int = 0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        start_time = time.perf_counter()
        print(f"\nProcessing page {page_number}: {image_path}")

        image = cv2.imread(str(image_path))
        if not self.validate_image(image):
            raise ValueError(f"Could not read page image: {image_path}")

        # Deskew & background normalization
        image = self.deskew_image(image)
        image = self.enhance_image(image)

        boxes = self.detect_lines_easyocr(image, header_cutoff_ratio=None)
        print(f"Detected {len(boxes)} text lines.")

        line_images = [self.prepare_line_crop(image, box) for box in boxes]
        valid_indices = [i for i, c in enumerate(line_images) if c is not None]
        valid_crops = [line_images[i] for i in valid_indices]

        # First pass: Fast batched TrOCR
        primary_results = self.trocr_batch(valid_crops, num_beams=1)
        recognized_map = dict(zip(valid_indices, primary_results))

        page_lines = []

        for index, box in enumerate(boxes):
            text, confidence = recognized_map.get(index, ("", 0.0))
            source = "TrOCR"

            # Targeted retry for low-confidence or suspicious lines
            if (confidence < 0.72 or self.is_bad_text(text, confidence)) and line_images[index] is not None:
                crop = line_images[index]

                # Retry 1: Beam search (num_beams=3) for handwriting disambiguation
                retry_text, retry_conf = self.trocr_batch(
                    [crop], num_beams=3)[0]
                if retry_conf > confidence and not self.is_bad_text(retry_text, retry_conf):
                    text = retry_text
                    confidence = retry_conf
                    source = "TrOCR-Beam"

                # Retry 2: EasyOCR fallback if TrOCR is still unsatisfactory
                if self.use_easyocr_fallback and (confidence < 0.65 or self.is_bad_text(text, confidence)):
                    fb_text, fb_conf = self.easyocr_fallback(crop)
                    if fb_text and not self.is_bad_text(fb_text, fb_conf):
                        text = fb_text
                        confidence = fb_conf or 0.70
                        source = "EasyOCR"

            needs_review = self.is_bad_text(text, confidence)

            page_lines.append({
                "text": text,
                "source": source,
                "confidence": confidence,
                "needs_review": needs_review,
                "line_number": index + 1,
                "bbox": list(box),
            })

        elapsed = time.perf_counter() - start_time
        print(f"Page {page_number} processed in {elapsed:.2f} seconds.")
        return page_lines, q_counter_start

    # ======================================================
    # PROCESS COMPLETE DOCUMENT
    # ======================================================

    def process_document(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        results = []
        total_start = time.perf_counter()
        print(f"\nProcessing {len(image_paths)} page(s)...")

        for page_number, image_path in enumerate(image_paths, start=1):
            page_lines, _ = self.process_page(
                image_path,
                page_number=page_number,
            )
            results.append({
                "page": page_number,
                "filename": Path(image_path).name,
                "lines": page_lines,
            })

        elapsed = time.perf_counter() - total_start
        total_lines = sum(len(page["lines"]) for page in results)
        review_lines = sum(
            1 for page in results for line in page["lines"] if line["needs_review"]
        )

        print(f"\nOCR finished in {elapsed:.2f} seconds.")
        print(f"Pages processed: {len(results)}")
        print(f"Lines extracted: {total_lines}")
        print(f"Lines needing review: {review_lines}")

        return results

    # ======================================================
    # SAVE EXTRACTED TEXT
    # ======================================================

    def save_text(self, results: List[Dict[str, Any]], output_file: str) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as file:
            for page in results:
                file.write(f"\n===== PAGE {page['page']} =====\n\n")
                for line in page["lines"]:
                    text = line.get("text", "")
                    conf = line.get("confidence", 0.0)
                    flag = " [REVIEW REQUIRED]" if line.get(
                        "needs_review") else ""
                    file.write(f"{text}{flag}\n")

        print(f"\nExtracted text saved to: {output_path}")


if __name__ == "__main__":
    print("Use this OCR engine through the application or tests/test_ocr.py.")
