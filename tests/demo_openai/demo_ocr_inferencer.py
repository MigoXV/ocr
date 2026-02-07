"""
OCR inferencer 调用示例

复用 demo_ocr 的后处理流程，仅将 OCR 请求封装到推理器类。
"""

import asyncio
import os
import time

from ocr.image.render import render_ocr_results
from ocr.inferencers import OpenAIOCRInferencer
from ocr.inferencers.ocr_parser import get_plain_text, parse_plain_text, parse_raw_str
from ocr.utils import load_image

TRANSLATE_MODEL = "tencent/Hunyuan-MT-7B"
TARGET_LANGUAGE = "法语"
INPUT_PATH = "data-bin/inputs/test01.jpg"
OUTPUT_PATH = "data-bin/test-deepseek-ocr-inferencer" + time.strftime(
    "_%Y%m%d_%H%M%S", time.localtime()
)


async def main() -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    image = load_image(INPUT_PATH).convert("RGB")
    with open(INPUT_PATH, "rb") as f:
        image_bytes = f.read()
    inferencer = OpenAIOCRInferencer(
        model="deepseek-ocr2",
        translate_model=TRANSLATE_MODEL,
    )
    # 启动ocr
    outputs = ""
    async for content in inferencer.ocr_stream(image_bytes):
        print(content, end="", flush=True)
        outputs += content
    print("\n")
    # 解析OCR结果并渲染图片
    ocr_results = parse_raw_str(outputs)
    rendered_image = render_ocr_results(image, ocr_results)
    rendered_image.save(f"{OUTPUT_PATH}/result_covered_original.jpg")
    print(f"结果已保存到: {OUTPUT_PATH}/result_covered_original.jpg")
    # 翻译OCR结果
    # full_context = "\n".join([f"{ocr_item.ref}" for ocr_item in ocr_results])
    plain_text = get_plain_text(ocr_results)
    translate_result_itr = inferencer.translate_stream(
        text=plain_text, target_language=TARGET_LANGUAGE
    )
    outputs = ""
    async for chunk in translate_result_itr:
        print(chunk, end="", flush=True)
        outputs += chunk
    print()
    # 将翻译结果渲染到图片上
    translate_results = parse_plain_text(outputs, ocr_results)
    rendered_translated_image = render_ocr_results(image, translate_results)
    rendered_translated_image.save(f"{OUTPUT_PATH}/result_covered_translated.jpg")
    print(f"结果已保存到: {OUTPUT_PATH}/result_covered_translated.jpg")


if __name__ == "__main__":
    asyncio.run(main())
