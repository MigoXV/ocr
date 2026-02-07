"""
极简图像编辑示例：
在输入图上添加一只简笔画小猫。
"""

import asyncio
import base64
import time
from pathlib import Path

from openai import AsyncOpenAI

INPUT_PATH = Path("data-bin/inputs/test01.jpg")
OUTPUT_DIR = Path("data-bin/outputs")
OUTPUT_PATH = OUTPUT_DIR / f"image_edit_cat_{time.strftime('%Y%m%d_%H%M%S')}.png"


async def main() -> None:
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="test-api-key",
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with INPUT_PATH.open("rb") as image_file:
            result = await client.images.edit(
                model="gpt-image-1",
                image=image_file,
                prompt="给我把图片里面的文字翻译为英语",
                size="1024x1024",
            )

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        OUTPUT_PATH.write_bytes(image_bytes)
        print(f"已保存编辑结果: {OUTPUT_PATH}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
