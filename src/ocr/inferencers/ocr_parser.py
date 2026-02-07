from __future__ import annotations

import ast
import logging
import re
from typing import List

import numpy as np

from ocr.types.ocr_results import OCRResult, OCRResults

logger = logging.getLogger(__name__)


def parse_raw_str(raw_text: str) -> OCRResults:
    """Parse OCR raw text into OCRResult list.

    Raw fragment format:
    <|ref|>...<|/ref|><|det|>[[x1, y1, x2, y2], ...]<|/det|>
    """
    pattern = re.compile(
        r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>",
        re.DOTALL,
    )

    results: OCRResults = []
    for ref, det_str in pattern.findall(raw_text):
        try:
            det_obj = ast.literal_eval(det_str.strip())
            det = np.array(det_obj, dtype=np.int32) / 999.0
            det = det[0]
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse det: {det_str.strip()}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error parsing det: {e}")
            continue
        results.append(OCRResult(ref=ref.strip(), det=det))

    return results


def get_plain_text(ocr_results: OCRResults) -> str:
    """Extract plain text from OCRResults."""
    return "\n".join([item.ref for item in ocr_results])


def parse_plain_text(plain_text: str, ocr_results: OCRResults) -> OCRResults:
    """Parse plain text back into OCRResults format."""
    text_list = plain_text.splitlines()
    if len(text_list) != len(ocr_results):
        raise ValueError("Plain text lines count must match OCR results count.")
    results = []
    for text, ocr_item in zip(text_list, ocr_results):
        results.append(OCRResult(ref=text.strip(), det=ocr_item.det))
    return results
