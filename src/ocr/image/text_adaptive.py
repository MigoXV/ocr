import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_SIZE_BASE = 32
_LOGGER = logging.getLogger(__name__)
_FONT_FALLBACK_WARNED = False


@lru_cache(maxsize=1)
def _find_cjk_font_path() -> str | None:
    env_path = os.environ.get("OCR_RENDER_FONT_PATH")
    if env_path and Path(env_path).is_file():
        return env_path

    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/arphic/uming.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return path
    return None


def _load_font(font_size: int) -> ImageFont.ImageFont:
    global _FONT_FALLBACK_WARNED
    font_path = _find_cjk_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, font_size)
        except OSError:
            if not _FONT_FALLBACK_WARNED:
                _LOGGER.warning(
                    "OCR render font load failed: %s. Falling back to default font, "
                    "which may not support Chinese.",
                    font_path,
                )
                _FONT_FALLBACK_WARNED = True
    elif not _FONT_FALLBACK_WARNED:
        _LOGGER.warning(
            "No CJK font found for OCR rendering. Set OCR_RENDER_FONT_PATH to a "
            "valid .ttf/.ttc font file to render Chinese text correctly."
        )
        _FONT_FALLBACK_WARNED = True
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    probe = text if text else " "
    left, top, right, bottom = draw.textbbox((0, 0), probe, font=font)
    return right - left, bottom - top


def _wrap_text_for_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_w: int,
    font_size: int,
    line_spacing_ratio: float,
) -> Tuple[List[str], int, int, int]:
    paragraphs = (text or "").splitlines() or [text]
    font = _load_font(font_size)
    lines: List[str] = []
    _, line_height = _text_size(draw, "Ag", font)
    line_gap = max(1, int(line_height * line_spacing_ratio))
    total_h = 0

    for para in paragraphs:
        if not para:
            lines.append("")
            total_h += line_height + line_gap
            continue

        current = ""
        for ch in para:
            candidate = current + ch
            cand_w, _ = _text_size(draw, candidate, font)
            if cand_w <= max_w or not current:
                current = candidate
            else:
                lines.append(current)
                total_h += line_height + line_gap
                current = ch
        lines.append(current)
        total_h += line_height + line_gap

    if lines:
        total_h -= line_gap

    return lines, line_height, line_gap, total_h


def _choose_best_font_scale(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_w: int,
    max_h: int,
    min_font_scale: float,
    max_font_scale: float,
    line_spacing_ratio: float,
) -> int:
    lo = max(1, int(round(min_font_scale * _FONT_SIZE_BASE)))
    hi = max(lo, int(round(max_font_scale * _FONT_SIZE_BASE)))
    best_size = lo
    for _ in range(22):
        mid = (lo + hi) // 2
        lines, _, _, total_h = _wrap_text_for_width(
            draw=draw,
            text=text,
            max_w=max_w,
            font_size=mid,
            line_spacing_ratio=line_spacing_ratio,
        )
        font = _load_font(mid)
        max_line_w = 0
        for line in lines:
            line_w, _ = _text_size(draw, line, font)
            max_line_w = max(max_line_w, line_w)
        fits = max_line_w <= max_w and total_h <= max_h
        if fits:
            best_size = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best_size


def render_text_adaptive_cv2(
    canvas: np.ndarray,
    text: str,
    box: Tuple[int, int, int, int],
    text_color: Tuple[int, int, int] = (0, 0, 0),
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    min_font_scale: float = 0.3,
    max_font_scale: float = 3.0,
    thickness: int = 1,
    padding_ratio: float = 0.08,
    line_spacing_ratio: float = 0.25,
) -> None:
    """Render text into box with auto line-wrap and adaptive font scale."""
    _ = font_face, thickness
    if not text:
        return

    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 2 or box_h <= 2:
        return

    pad_x = max(1, int(box_w * padding_ratio))
    pad_y = max(1, int(box_h * padding_ratio))
    max_w = box_w - pad_x * 2
    max_h = box_h - pad_y * 2
    if max_w <= 2 or max_h <= 2:
        return

    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(canvas_rgb)
    draw = ImageDraw.Draw(pil_image)

    best_font_size = _choose_best_font_scale(
        draw=draw,
        text=text,
        max_w=max_w,
        max_h=max_h,
        min_font_scale=min_font_scale,
        max_font_scale=max_font_scale,
        line_spacing_ratio=line_spacing_ratio,
    )
    lines, line_h, line_gap, total_h = _wrap_text_for_width(
        draw=draw,
        text=text,
        max_w=max_w,
        font_size=best_font_size,
        line_spacing_ratio=line_spacing_ratio,
    )
    font = _load_font(best_font_size)
    start_y = y1 + pad_y + max(0, (max_h - total_h) // 2) + line_h
    for idx, line in enumerate(lines):
        line_w, _ = _text_size(draw, line, font)
        draw_x = x1 + pad_x + max(0, (max_w - line_w) // 2)
        draw_y = start_y + idx * (line_h + line_gap)
        if draw_y > y2 - pad_y:
            break
        draw.text((draw_x, draw_y - line_h), line, fill=text_color[::-1], font=font)

    canvas[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
