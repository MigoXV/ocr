from __future__ import annotations

import ast
import logging
import re

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
