"""
极简图像编辑示例：
在输入图上添加一只简笔画小猫。
"""

import base64
import time
from pathlib import Path

from openai import OpenAI

INPUT_PATH = Path("data-bin/inputs/test01.jpg")
OUTPUT_DIR = Path("data-bin/outputs")
OUTPUT_PATH = OUTPUT_DIR / f"image_edit_cat_{time.strftime('%Y%m%d_%H%M%S')}.png"


def main() -> None:
    client = OpenAI()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("rb") as image_file:
        result = client.images.edit(
            model="gpt-image-1",
            image=image_file,
            prompt="请在原图内容基础上，在明显但不遮挡主体的位置画一只可爱的简笔画小猫，线条清晰。",
            size="1024x1024",
        )

    image_base64 = result.data[0].b64_json
    image_bytes = base64.b64decode(image_base64)
    OUTPUT_PATH.write_bytes(image_bytes)
    print(f"已保存编辑结果: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
