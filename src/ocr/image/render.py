import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from ocr.image.text_adaptive import render_text_adaptive_cv2
from ocr.types.ocr_results import OCRResults

logger = logging.getLogger(__name__)


def _extract_box_from_det(
    det: np.ndarray, img_w: int, img_h: int
) -> Tuple[int, int, int, int] | None:
    """Extract pixel box (x1,y1,x2,y2) from normalized det in supported shapes."""
    x1f, y1f, x2f, y2f = det
    x1 = int(np.floor(np.clip(min(x1f, x2f), 0.0, 1.0) * img_w))
    x2 = int(np.ceil(np.clip(max(x1f, x2f), 0.0, 1.0) * img_w))
    y1 = int(np.floor(np.clip(min(y1f, y2f), 0.0, 1.0) * img_h))
    y2 = int(np.ceil(np.clip(max(y1f, y2f), 0.0, 1.0) * img_h))
    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w))
    y2 = max(0, min(y2, img_h))
    return x1, y1, x2, y2


def render_ocr_results(image: Image.Image, ocr_results: OCRResults) -> Image.Image:
    """Render OCR text by whitening each OCR region and fitting text into it."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    canvas = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img_h, img_w = canvas.shape[:2]
    # 遍历ocr结果
    for ocr_item in ocr_results:
        try:
            x1, y1, x2, y2 = _extract_box_from_det(ocr_item.det, img_w, img_h)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)
            render_text_adaptive_cv2(canvas, ocr_item.ref, (x1, y1, x2, y2))
        except Exception as e:
            logger.warning("Failed to render OCR item '%s': %s", ocr_item.ref, e)
            continue
    result_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)
