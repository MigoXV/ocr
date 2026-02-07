import asyncio
import base64

from openai import AsyncOpenAI
from pydantic import TypeAdapter

from ocr.types.ocr_results import OCRResult

INPUT_PATH = "data-bin/inputs/test01.jpg"
MODEL = "deepseek-ocr2"
PROMPT = "OCR this image."


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    if image_path.lower().endswith(".png"):
        mime_type = "image/png"
    elif image_path.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    else:
        mime_type = "image/png"
    return f"data:{mime_type};base64,{base64_data}"


async def main() -> None:
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="test-api-key",
    )
    try:
        image_base64 = image_to_base64(INPUT_PATH)

        stream = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": image_base64}},
                    ],
                }
            ],
            model=MODEL,
            stream=True,
        )

        output = ""
        async for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
                output += content

        print("\n")

        adapter = TypeAdapter(list[OCRResult])
        results = adapter.validate_json(output)

        print(f"OCR Result count: {len(results)}")
        for idx, item in enumerate(results):
            print(f"[{idx}] {item.model_dump_json(ensure_ascii=False)}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
