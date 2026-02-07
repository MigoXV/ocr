from __future__ import annotations

import ast
import logging
import re
from typing import AsyncIterator

import numpy as np

from ocr.types.ocr_results import OCRResult, OCRResults

logger = logging.getLogger(__name__)

REF_OPEN = "<|ref|>"
REF_CLOSE = "<|/ref|>"
DET_OPEN = "<|det|>"
DET_CLOSE = "<|/det|>"


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
            det = (np.array(det_obj, dtype=np.float32) / 999.0)[0].tolist()
        except (ValueError, SyntaxError):
            logger.warning(f"Failed to parse det: {det_str.strip()}")
            continue
        except Exception as e:
            logger.error(f"Unexpected error parsing det: {e}")
            continue
        results.append(OCRResult(ref=ref.strip(), det=det))

    return results


async def parse_raw_str_stream(
    text_stream: AsyncIterator[str],
) -> AsyncIterator[OCRResult]:
    """Parse streaming OCR raw text into dict items.

    Input fragments are accumulated in a buffer until a complete segment is found:
    <|ref|>...<|/ref|><|det|>...<|/det|>
    """
    buffer = ""

    async for chunk in text_stream:
        if not chunk:
            continue
        buffer += chunk

        while True:
            ref_start = buffer.find(REF_OPEN)
            if ref_start < 0:
                # Keep only a short suffix to avoid unbounded buffer growth.
                buffer = buffer[-len(REF_OPEN) :]
                break

            if ref_start > 0:
                buffer = buffer[ref_start:]
                ref_start = 0

            ref_end = buffer.find(REF_CLOSE, ref_start + len(REF_OPEN))
            if ref_end < 0:
                break

            det_start = buffer.find(DET_OPEN, ref_end + len(REF_CLOSE))
            if det_start < 0:
                break

            det_end = buffer.find(DET_CLOSE, det_start + len(DET_OPEN))
            if det_end < 0:
                break

            ref_text = buffer[ref_start + len(REF_OPEN) : ref_end].strip()
            det_text = buffer[det_start + len(DET_OPEN) : det_end].strip()
            segment_end = det_end + len(DET_CLOSE)

            try:
                det_obj = ast.literal_eval(det_text)
            except (ValueError, SyntaxError):
                logger.warning("Failed to parse det in stream: %s", det_text)
            except Exception as exc:
                logger.error("Unexpected det parse error in stream: %s", exc)
            else:
                yield OCRResult(ref=ref_text, det=det_obj)

            buffer = buffer[segment_end:]


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
