"""OCR parser demo: run OCR once and parse structured tags."""

import asyncio
import time

from ocr.inferencers import OpenAIOCRInferencer
from ocr.inferencers.ocr_parser import parse_raw_str

INPUT_PATH = "data-bin/inputs/test01.jpg"


async def main() -> None:
    with open(INPUT_PATH, "rb") as f:
        image_bytes = f.read()

    inferencer = OpenAIOCRInferencer(model="deepseek-ocr2")

    print("=" * 15 + " 调用 OCR API(推理器) " + "=" * 15)
    raw_outputs = ""
    start = time.time()
    async for chunk in inferencer.ocr_stream(image_bytes):
        print(chunk, end="", flush=True)
        raw_outputs += chunk
    print("\n")
    print(f"OCR耗时: {time.time() - start:.2f}s")

    print("=" * 15 + " 解析 OCR 结构化结果 " + "=" * 15)
    parsed = parse_raw_str(raw_outputs)
    print(f"共解析到 {len(parsed)} 条 OCRResult")

    for idx, item in enumerate(parsed):
        print(f"[{idx}] ref: {item.ref}\t\t det: {item.det}")
        # print(f"[{idx}] det shape: {item.det.shape}, det: {item.det.tolist()}")


if __name__ == "__main__":
    asyncio.run(main())
