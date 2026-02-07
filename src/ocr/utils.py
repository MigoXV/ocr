"""
DeepSeek OCR2 工具函数模块

本模块提供了 OCR 结果处理相关的工具函数，包括：
- 图像加载和预处理
- OCR 输出文本解析
- 检测框可视化绘制
"""

import re

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def load_image(image_path):
    """
    加载图片并自动校正 EXIF 方向

    很多手机拍摄的照片会在 EXIF 元数据中记录拍摄方向，
    而实际的像素数据可能是旋转过的。这个函数会自动校正这种情况。

    Args:
        image_path (str): 图片文件的路径

    Returns:
        PIL.Image: 加载并校正方向后的图片对象
        None: 如果加载失败

    示例:
        >>> image = load_image("/path/to/image.jpg")
        >>> if image:
        ...     image = image.convert("RGB")  # 转换为 RGB 模式
    """
    try:
        # 使用 PIL 打开图片
        image = Image.open(image_path)

        # exif_transpose 会读取 EXIF 中的方向信息，
        # 并对图片进行相应的旋转/翻转，使其显示正确
        corrected_image = ImageOps.exif_transpose(image)

        return corrected_image

    except Exception as e:
        print(f"error: {e}")
        # 如果 EXIF 校正失败，尝试直接打开图片
        try:
            return Image.open(image_path)
        except:
            return None


def extract_coordinates_and_label(ref_text, image_width, image_height):
    """
    从单个匹配结果中提取标签类型和坐标列表

    Args:
        ref_text (tuple): re_match 返回的单个匹配元组 (完整匹配, 类型, 坐标字符串)
        image_width (int): 图片宽度（用于坐标转换，目前未使用）
        image_height (int): 图片高度（用于坐标转换，目前未使用）

    Returns:
        tuple: (标签类型, 坐标列表)
            - 标签类型: str，如 "image", "title", "text" 等
            - 坐标列表: list，格式为 [[x1,y1,x2,y2], ...]
        None: 如果解析失败

    注意:
        坐标值是归一化到 0-999 范围的，需要在绘制时转换为实际像素坐标
    """
    try:
        # ref_text[1] 是类型（如 "image", "title"）
        label_type = ref_text[1]

        # ref_text[2] 是坐标字符串，如 "[[100,200,300,400]]"
        # 使用 eval 将字符串转换为 Python 列表
        # 注意: eval 有安全风险，但这里输入来自模型输出，相对可控
        cor_list = eval(ref_text[2])

    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def draw_bounding_boxes(image, refs, output_path=None):
    """
    在图片上绘制检测到的边界框

    这个函数会在原图上绘制 OCR 检测到的所有元素的边界框，
    并为不同类型的元素使用不同的样式（如标题用粗边框）。

    Args:
        image (PIL.Image): 原始图片对象
        refs (list): re_match 返回的匹配列表
        output_path (str, optional): 输出路径，用于保存裁剪的图片区域

    Returns:
        PIL.Image: 绘制了边界框的图片

    绘制规则:
        - title 类型: 4像素宽的边框
        - 其他类型: 2像素宽的边框
        - image 类型: 会额外裁剪保存图片区域
        - 每个框使用随机颜色，便于区分
        - 框上方显示类型标签
    """
    # 获取图片尺寸，用于坐标转换
    image_width, image_height = image.size

    # 复制原图，避免修改原始图片
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)

    # 创建一个透明的覆盖层，用于绘制半透明的填充效果
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))
    draw2 = ImageDraw.Draw(overlay)

    font = ImageFont.truetype(
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size=32
    )

    # 图片区域计数器，用于命名保存的裁剪图片
    img_idx = 0

    # 遍历所有检测到的元素
    for i, ref in enumerate(refs):
        try:
            # 提取标签类型和坐标
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result:
                label_type, points_list = result

                # 为当前元素生成随机颜色
                # RGB 值限制在 0-200 范围，避免太亮看不清
                color = (
                    np.random.randint(0, 200),
                    np.random.randint(0, 200),
                    np.random.randint(0, 255),
                )

                # 带透明度的颜色，用于半透明填充
                # 20 是 alpha 值（0-255），值越小越透明
                color_a = color + (20,)

                # 一个元素可能有多个边界框（如分散的文本块）
                for points in points_list:
                    # 获取归一化坐标 (0-999 范围)
                    x1, y1, x2, y2 = points

                    # ===== 坐标转换 =====
                    # 模型输出的坐标是归一化到 0-999 范围的
                    # 需要转换为实际的像素坐标
                    # 公式: 实际坐标 = 归一化坐标 / 999 * 图片尺寸
                    x1 = int(x1 / 999 * image_width)
                    y1 = int(y1 / 999 * image_height)
                    x2 = int(x2 / 999 * image_width)
                    y2 = int(y2 / 999 * image_height)

                    # ===== 处理图片类型元素 =====
                    # 如果是图片区域，裁剪并保存
                    if label_type == "image" and output_path:
                        try:
                            # 从原图裁剪该区域
                            cropped = image.crop((x1, y1, x2, y2))
                            # 保存到输出目录
                            cropped.save(f"{output_path}/images/{img_idx}.jpg")
                        except Exception as e:
                            print(e)
                            pass
                        img_idx += 1

                    # ===== 绘制边界框 =====
                    try:
                        if label_type == "title":
                            # 标题使用粗边框 (4像素)
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )
                        else:
                            # 其他元素使用普通边框 (2像素)
                            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                            draw2.rectangle(
                                [x1, y1, x2, y2],
                                fill=color_a,
                                outline=(0, 0, 0, 0),
                                width=1,
                            )

                        # ===== 绘制标签文字 =====
                        # 标签位置：边界框左上角上方
                        text_x = x1
                        text_y = max(0, y1 - 15)  # 确保不超出图片顶部

                        # 计算文字尺寸，用于绘制背景
                        text_bbox = draw.textbbox((0, 0), label_type, font=font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]

                        # 绘制文字背景（半透明白色）
                        draw.rectangle(
                            [text_x, text_y, text_x + text_width, text_y + text_height],
                            fill=(255, 255, 255, 30),
                        )

                        # 绘制标签文字
                        draw.text((text_x, text_y), label_type, font=font, fill=color)
                    except:
                        pass
        except:
            continue

    # 将透明覆盖层合并到主图上
    img_draw.paste(overlay, (0, 0), overlay)
    return img_draw


def process_image_with_refs(image, ref_texts, output_path=None):
    """
    处理图片并绘制所有检测到的引用框

    这是一个便捷的包装函数，封装了 draw_bounding_boxes。

    Args:
        image (PIL.Image): 原始图片对象
        ref_texts (list): re_match 返回的匹配列表
        output_path (str, optional): 输出路径，用于保存裁剪的图片

    Returns:
        PIL.Image: 绘制了边界框的图片

    使用示例:
        >>> image = load_image("input.jpg")
        >>> matches, _, _ = re_match(ocr_output)
        >>> result = process_image_with_refs(image, matches, "output/")
        >>> result.save("result_with_boxes.jpg")
    """
    result_image = draw_bounding_boxes(image, ref_texts, output_path)
    return result_image


# =====================================================================
# 白色方块覆盖 + 自适应文字渲染功能
# =====================================================================


def wrap_text_to_box(text, font, max_width):
    """
    将文本按指定宽度换行，支持中英文混排

    逐字符检测宽度，当超出最大宽度时自动换行。

    Args:
        text (str): 要换行的文本
        font (ImageFont): PIL 字体对象
        max_width (int): 最大允许宽度（像素）

    Returns:
        list: 换行后的文本行列表

    示例:
        >>> font = ImageFont.truetype("font.ttc", 20)
        >>> lines = wrap_text_to_box("这是一段很长的文本", font, 100)
        >>> print(lines)
        ['这是一段', '很长的文', '本']
    """
    if not text:
        return []

    lines = []
    current_line = ""

    for char in text:
        # 处理换行符
        if char == "\n":
            if current_line:
                lines.append(current_line)
            current_line = ""
            continue

        test_line = current_line + char

        # 计算当前行宽度
        bbox = font.getbbox(test_line)
        width = bbox[2] - bbox[0]

        if width <= max_width:
            current_line = test_line
        else:
            # 当前行已满，保存并开始新行
            if current_line:
                lines.append(current_line)
            current_line = char

    # 添加最后一行
    if current_line:
        lines.append(current_line)

    return lines if lines else [text]


def find_optimal_font_size(
    text, box_width, box_height, font_path, min_size=8, max_size=72, padding_ratio=0.1
):
    """
    二分查找能够容纳文本的最大字号

    在给定的边界框内，找到能够完整显示文本的最大字体大小。

    Args:
        text (str): 要显示的文本
        box_width (int): 边界框宽度（像素）
        box_height (int): 边界框高度（像素）
        font_path (str): 字体文件路径
        min_size (int): 最小字号
        max_size (int): 最大字号
        padding_ratio (float): 内边距比例（0-1）

    Returns:
        int: 最优字号

    算法说明:
        使用二分查找，每次检测当前字号是否能让文本完整显示在框内。
        考虑换行后的总高度和最大行宽。
    """
    # 计算实际可用区域（去除内边距）
    available_width = int(box_width * (1 - padding_ratio * 2))
    available_height = int(box_height * (1 - padding_ratio * 2))

    if available_width <= 0 or available_height <= 0:
        return min_size

    low, high = min_size, max_size
    best_size = min_size

    while low <= high:
        mid = (low + high) // 2

        try:
            font = ImageFont.truetype(font_path, mid)
        except:
            # 字体加载失败，使用默认字体
            font = ImageFont.load_default()

        # 获取换行后的文本行
        wrapped_lines = wrap_text_to_box(text, font, available_width)

        if not wrapped_lines:
            low = mid + 1
            continue

        # 计算行高（使用 'Ay中' 测试，包含上下突出部分）
        line_height_bbox = font.getbbox("Ay中文")
        line_height = (line_height_bbox[3] - line_height_bbox[1]) * 1.2  # 1.2 倍行距

        # 计算总高度
        total_height = len(wrapped_lines) * line_height

        # 计算最大行宽
        max_line_width = 0
        for line in wrapped_lines:
            line_bbox = font.getbbox(line)
            line_width = line_bbox[2] - line_bbox[0]
            max_line_width = max(max_line_width, line_width)

        # 检查是否能容纳
        if max_line_width <= available_width and total_height <= available_height:
            best_size = mid
            low = mid + 1
        else:
            high = mid - 1

    return best_size


def cover_and_render_text(
    image,
    refs,
    font_path=None,
    padding_ratio=0.05,
    min_font_size=8,
    max_font_size=72,
    text_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    skip_types=None,
    text_transform=None,
):
    """
    用白色方块覆盖检测区域并渲染自适应文字

    对每个检测到的文本区域：
    1. 用白色不透明矩形覆盖原始内容
    2. 计算能容纳文本的最大字号
    3. 处理文本换行
    4. 在覆盖区域内居中渲染文字

    Args:
        image (PIL.Image): 原始图片对象
        refs (list): re_match 返回的匹配列表，格式为 [(完整匹配, 类型, 坐标), ...]
        font_path (str, optional): 字体文件路径，默认使用 Noto Sans CJK
        padding_ratio (float): 内边距比例，默认 0.05
        min_font_size (int): 最小字号，默认 8
        max_font_size (int): 最大字号，默认 72
        text_color (tuple): 文字颜色 RGB，默认黑色 (0, 0, 0)
        bg_color (tuple): 背景颜色 RGB，默认白色 (255, 255, 255)
        skip_types (list, optional): 要跳过的类型列表，默认跳过 ["image"]
        text_transform (callable, optional): 文本转换函数，如翻译函数。
            接受原文本作为参数，返回转换后的文本。默认为 None（不转换）。

    Returns:
        PIL.Image: 处理后的图片

    示例:
        >>> image = load_image("input.jpg")
        >>> matches, _, _ = re_match(ocr_output)
        >>> # 不翻译，直接渲染原文
        >>> result = cover_and_render_text(image, matches)
        >>> # 使用翻译函数
        >>> result = cover_and_render_text(image, matches, text_transform=translate_func)
        >>> result.save("result_covered.jpg")
    """
    # 默认字体路径
    if font_path is None:
        font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

    # 默认跳过图片类型
    if skip_types is None:
        skip_types = ["image"]

    # 获取图片尺寸
    image_width, image_height = image.size

    # 复制原图
    img_result = image.copy()

    # 确保图片是 RGB 模式
    if img_result.mode != "RGB":
        img_result = img_result.convert("RGB")

    draw = ImageDraw.Draw(img_result)

    # 遍历所有检测到的元素
    for ref in refs:
        try:
            # 提取标签类型和坐标
            result = extract_coordinates_and_label(ref, image_width, image_height)
            if result is None:
                continue

            label_type, points_list = result

            # 跳过指定类型（如图片区域）
            if label_type in skip_types:
                continue

            # 获取要渲染的文本（类型即为识别的文字内容）
            text = label_type

            # 如果提供了文本转换函数（如翻译），则转换文本
            if text_transform is not None:
                text = text_transform(text)

            # 处理每个边界框
            for points in points_list:
                # 获取归一化坐标 (0-999 范围)
                x1, y1, x2, y2 = points

                # 坐标转换：归一化 → 像素
                x1 = int(x1 / 999 * image_width)
                y1 = int(y1 / 999 * image_height)
                x2 = int(x2 / 999 * image_width)
                y2 = int(y2 / 999 * image_height)

                # 确保坐标有效
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                box_width = x2 - x1
                box_height = y2 - y1

                if box_width <= 0 or box_height <= 0:
                    continue

                # ===== 1. 绘制白色不透明背景 =====
                draw.rectangle([x1, y1, x2, y2], fill=bg_color)

                # ===== 2. 计算最优字号 =====
                font_size = find_optimal_font_size(
                    text,
                    box_width,
                    box_height,
                    font_path,
                    min_font_size,
                    max_font_size,
                    padding_ratio,
                )

                try:
                    font = ImageFont.truetype(font_path, font_size)
                except:
                    font = ImageFont.load_default()

                # ===== 3. 处理文字换行 =====
                available_width = int(box_width * (1 - padding_ratio * 2))
                wrapped_lines = wrap_text_to_box(text, font, available_width)

                if not wrapped_lines:
                    continue

                # ===== 4. 计算文字位置（居中） =====
                # 计算行高
                line_height_bbox = font.getbbox("Ay中文")
                line_height = int((line_height_bbox[3] - line_height_bbox[1]) * 1.2)

                # 计算总文字高度
                total_text_height = len(wrapped_lines) * line_height

                # 计算起始 Y 坐标（垂直居中）
                padding_y = int(box_height * padding_ratio)
                start_y = y1 + max(padding_y, (box_height - total_text_height) // 2)

                # ===== 5. 逐行渲染文字 =====
                for i, line in enumerate(wrapped_lines):
                    # 计算该行宽度
                    line_bbox = font.getbbox(line)
                    line_width = line_bbox[2] - line_bbox[0]

                    # 计算 X 坐标（水平居中）
                    padding_x = int(box_width * padding_ratio)
                    text_x = x1 + max(padding_x, (box_width - line_width) // 2)
                    text_y = start_y + i * line_height

                    # 确保不超出边界框
                    if text_y + line_height > y2:
                        break

                    # 绘制文字
                    draw.text((text_x, text_y), line, font=font, fill=text_color)

        except Exception as e:
            # 跳过处理失败的元素
            continue

    return img_result
