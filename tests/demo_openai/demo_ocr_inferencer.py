"""
OCR inferencer 调用示例

复用 demo_ocr 的后处理流程，仅将 OCR 请求封装到推理器类。
"""

import os
import time

from ocr.inferencers import OpenAIOCRInferencer
from ocr.inferencers.ocr_parser import parse_raw_str
from ocr.image.render import render_ocr_results
from ocr.utils import load_image

TRANSLATE_MODEL = "tencent/Hunyuan-MT-7B"
TARGET_LANGUAGE = "法语"
INPUT_PATH = "data-bin/inputs/test01.jpg"
OUTPUT_PATH = "data-bin/test-deepseek-ocr-inferencer" + time.strftime(
    "_%Y%m%d_%H%M%S", time.localtime()
)


def create_translator(inferencer, model, target_language, full_context=None):
    cache = {}

    def translate_func(text):
        if not text or len(text.strip()) == 0:
            return text

        if text in cache:
            return cache[text]

        try:
            translated = ""
            for chunk in inferencer.translate_stream(
                text=text,
                target_language=target_language,
                full_context=full_context,
                model=model,
            ):
                print(chunk, end="", flush=True)
                translated += chunk
            print()
            translated = translated.strip()
            cache[text] = translated
            return translated
        except Exception as e:
            print(f"翻译失败: {text[:20]}... -> {e}")
            return text

    return translate_func


if __name__ == "__main__":
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)

    image = load_image(INPUT_PATH).convert("RGB")
    with open(INPUT_PATH, "rb") as f:
        image_bytes = f.read()

    inferencer = OpenAIOCRInferencer(
        model="deepseek-ocr2",
        translate_model=TRANSLATE_MODEL,
    )

    print("=" * 15 + " 调用 OCR API(推理器) " + "=" * 15)
    outputs = ""
    for content in inferencer.ocr_stream(image_bytes):
        print(content, end="", flush=True)
        outputs += content
    print("\n")

    print("=" * 15 + " 保存结果 " + "=" * 15)

    with open(f"{OUTPUT_PATH}/result_ori.mmd", "w", encoding="utf-8") as afile:
        afile.write(outputs)

    ocr_results = parse_raw_str(outputs)
    print(f"共解析到 {len(ocr_results)} 条 OCRResult")

    print("=" * 15 + " 渲染原文(白底覆盖+自适应文字) " + "=" * 15)
    rendered_image = render_ocr_results(image, ocr_results)
    rendered_image.save(f"{OUTPUT_PATH}/result_covered_original.jpg")
    print(f"结果已保存到: {OUTPUT_PATH}/result_covered_original.jpg")

    # translate_func = create_translator(
    #     inferencer,
    #     TRANSLATE_MODEL,
    #     TARGET_LANGUAGE,
    #     full_context=outputs,
    # )



    # print("=" * 15 + " 渲染原文 " + "=" * 15)
    # result_covered_original = cover_and_render_text(image, matches_ref)

    # print("=" * 15 + " 翻译并渲染文字 " + "=" * 15)
    # result_covered_translated = cover_and_render_text(
    #     image, matches_ref, text_transform=translate_func
    # )

    # for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
    #     outputs = outputs.replace(a_match_image, f"![](images/{idx}.jpg)\n")

    # for _, a_match_other in enumerate(tqdm(matches_other, desc="other")):
    #     outputs = (
    #         outputs.replace(a_match_other, "")
    #         .replace("\\coloneqq", ":=")
    #         .replace("\\eqqcolon", "=:")
    #     )

    # with open(f"{OUTPUT_PATH}/result.mmd", "w", encoding="utf-8") as afile:
    #     afile.write(outputs)

    # result.save(f"{OUTPUT_PATH}/result_with_boxes.jpg")
    # result_covered_original.save(f"{OUTPUT_PATH}/result_covered_original.jpg")
    # result_covered_translated.save(f"{OUTPUT_PATH}/result_covered_translated.jpg")

    # print(f"结果已保存到: {OUTPUT_PATH}")
