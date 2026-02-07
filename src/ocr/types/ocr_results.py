from __future__ import annotations

from pydantic import BaseModel


class OCRResult(BaseModel):
    ref: str
    # Normalized box for render path: [x1, y1, x2, y2]
    # Stream parse path may carry raw shape: [[x1, y1, x2, y2], ...]
    det: list[float] | list[list[int]]


OCRResults = list[OCRResult]
