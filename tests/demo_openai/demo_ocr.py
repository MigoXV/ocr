"""
DeepSeek OCR2 OpenAI 客户端调用示例

使用 OpenAI 兼容的 API 客户端调用 DeepSeek OCR2 服务
"""

import base64
import os
import time

from openai import OpenAI
from tqdm import tqdm

from ocr.utils import (
    cover_and_render_text,
    load_image,
    process_image_with_refs,
    re_match,
)

# ========== 配置 ==========
# OCR API 服务地址

# # 翻译 API 服务地址（新的 OpenAI 客户端实例）
TRANSLATE_MODEL = "MedAIBase/Tencent-HY-MT1.5:7b"  # 翻译使用的模型
TARGET_LANGUAGE = "法语"  # 目标翻译语言

# 输入图片路径
INPUT_PATH = "data-bin/inputs/test01.jpg"

# 输出目录路径（包含时间戳，避免覆盖）
OUTPUT_PATH = "data-bin/test-deepseek-ocr" + time.strftime(
    "_%Y%m%d_%H%M%S", time.localtime()
)


def image_to_base64(image_path: str) -> str:
    """将图片文件转换为 base64 data URI"""
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    # 简单判断图片格式
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"
    elif image_path.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/png"
    return f"data:{mime_type};base64,{base64_data}"


def create_translator(client, model, target_language, full_context=None):
    """
    创建翻译函数的工厂函数

    Args:
        client: OpenAI 客户端实例
        model: 使用的模型名称
        target_language: 目标翻译语言
        full_context: 全文上下文，用于提供翻译时的语境参考

    Returns:
        translate_func: 翻译函数，接受文本返回翻译结果
    """
    # 缓存已翻译的文本，避免重复调用 API
    cache = {}

    # 构建系统提示词
    system_prompt = f"""你是一个专业的翻译助手。请将用户提供的文本翻译成{target_language}。

翻译要求：
1. 只返回翻译结果，不要添加任何解释、注释或额外内容
2. 保持原文的格式和标点符号风格
3. 如果文本无法翻译（如纯数字、符号、公式等），则原样返回
4. 专业术语请使用{target_language}中的标准译法
5. 保持翻译的简洁性，译文长度应尽量与原文相近"""

    # 如果提供了全文上下文，添加到系统提示词中
    if full_context:
        system_prompt += f"""

以下是完整的文档内容，供你理解上下文和语境：
<document>
{full_context}
</document>

请根据上述文档的上下文，准确翻译用户提供的片段。"""

    def translate_func(text):
        """翻译单个文本"""
        if not text or len(text.strip()) == 0:
            return text

        # 检查缓存
        if text in cache:
            return cache[text]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            translated = response.choices[0].message.content.strip()
            cache[text] = translated
            return translated
        except Exception as e:
            print(f"翻译失败: {text[:20]}... -> {e}")
            return text  # 翻译失败时返回原文

    return translate_func


if __name__ == "__main__":
    # ========== 1. 创建输出目录 ==========
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(f"{OUTPUT_PATH}/images", exist_ok=True)

    # ========== 2. 加载图片并转换为 base64 ==========
    image = load_image(INPUT_PATH).convert("RGB")
    image_base64 = image_to_base64(INPUT_PATH)

    # ========== 3. 创建 OpenAI 客户端 ==========
    # OCR 客户端
    client = OpenAI()

    # 翻译客户端（新的独立实例，翻译函数将在获取 OCR 结果后创建）
    # translate_client = OpenAI()

    # ========== 4. 调用 OCR API ==========
    print("=" * 15 + " 调用 OCR API " + "=" * 15)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "OCR this image."},
                    {"type": "image_url", "image_url": {"url": image_base64}},
                ],
            }
        ],
        model="deepseek-ocr2",
        stream=True,
    )

    # ========== 5. 流式接收结果 ==========
    outputs = ""
    for chunk in response:
        if not chunk.choices:
            continue
        if chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            outputs += content
    print("\n")

    # ========== 6. 后处理和保存结果 ==========
    print("=" * 15 + " 保存结果 " + "=" * 15)

    # ----- 6.1 保存原始输出 -----
    with open(f"{OUTPUT_PATH}/result_ori.mmd", "w", encoding="utf-8") as afile:
        afile.write(outputs)

    # ----- 6.2 解析检测结果 -----
    matches_ref, matches_images, matches_other = re_match(outputs)

    # ----- 6.3 创建翻译函数（传入全文上下文） -----
    # 将 OCR 识别的全文作为上下文传给翻译模型，提供语境参考
    translate_func = create_translator(
        client,
        TRANSLATE_MODEL,
        TARGET_LANGUAGE,
        full_context=outputs,  # 传入 OCR 识别的全文
    )

    # ----- 6.4 绘制检测框并裁剪图片 -----
    image_draw = image.copy()
    result = process_image_with_refs(image_draw, matches_ref, OUTPUT_PATH)

    # ----- 6.5 白色方块覆盖 + 原文渲染（不翻译） -----
    print("=" * 15 + " 渲染原文 " + "=" * 15)
    result_covered_original = cover_and_render_text(
        image,
        matches_ref,
        # 不传 text_transform，使用 OCR 识别的原文
    )

    # ----- 6.6 白色方块覆盖 + 翻译文字渲染 -----
    # 用白色不透明方块覆盖原始检测区域，并在其上渲染翻译后的文字
    print("=" * 15 + " 翻译并渲染文字 " + "=" * 15)
    result_covered_translated = cover_and_render_text(
        image, matches_ref, text_transform=translate_func  # 传入翻译函数
    )

    # ----- 6.7 替换输出中的图片引用 -----
    for idx, a_match_image in enumerate(tqdm(matches_images, desc="image")):
        outputs = outputs.replace(a_match_image, f"![](images/{idx}.jpg)\n")

    # ----- 6.8 移除其他检测标签并修复特殊字符 -----
    for idx, a_match_other in enumerate(tqdm(matches_other, desc="other")):
        outputs = (
            outputs.replace(a_match_other, "")
            .replace("\\coloneqq", ":=")
            .replace("\\eqqcolon", "=:")
        )

    # ----- 6.9 保存处理后的输出 -----
    with open(f"{OUTPUT_PATH}/result.mmd", "w", encoding="utf-8") as afile:
        afile.write(outputs)

    # ----- 6.10 保存带检测框的图片 -----
    result.save(f"{OUTPUT_PATH}/result_with_boxes.jpg")

    # ----- 6.11 保存白色覆盖+原文渲染的图片 -----
    result_covered_original.save(f"{OUTPUT_PATH}/result_covered_original.jpg")

    # ----- 6.12 保存白色覆盖+翻译文字渲染的图片 -----
    result_covered_translated.save(f"{OUTPUT_PATH}/result_covered_translated.jpg")

    print(f"结果已保存到: {OUTPUT_PATH}")
