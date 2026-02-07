import asyncio
import base64
import json

from openai import AsyncOpenAI, BadRequestError
from pydantic import TypeAdapter

from ocr.types.ocr_results import OCRResult

INPUT_PATH = "data-bin/inputs/test01.jpg"
MODEL = "deepseek-ocr2"
OCR_PROMPT = "OCR this image."
TRANSLATE_LANGUAGE = "法语"
TRANSLATE_CONCURRENCY = 4


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


async def stream_to_text(stream, *, echo: bool = False) -> str:
    output = ""
    async for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            if echo:
                print(content, end="", flush=True)
            output += content
    return output


def parse_translate_output(output: str) -> str:
    translated_text = output.strip()
    try:
        parsed = TypeAdapter(list[str]).validate_json(output)
        if parsed:
            translated_text = parsed[0]
    except Exception:
        pass
    return translated_text


async def translate_one(async_client: AsyncOpenAI, semaphore: asyncio.Semaphore, idx: int, text: str) -> tuple[int, str]:
    primary_prompt = f"/translate {TRANSLATE_LANGUAGE} {text}"

    async with semaphore:
        try:
            response = await async_client.chat.completions.create(
                messages=[{"role": "user", "content": primary_prompt}],
                model=MODEL,
            )
        except BadRequestError as exc:
            detail = str(exc)
            if "request is not a translation task" not in detail:
                return idx, text
            return idx, text

    content = response.choices[0].message.content
    if isinstance(content, str):
        return idx, parse_translate_output(content)
    return idx, parse_translate_output(str(content))


async def translate_with_pool(async_client: AsyncOpenAI, ocr_results: list[OCRResult]) -> None:
    semaphore = asyncio.Semaphore(TRANSLATE_CONCURRENCY)
    tasks = [
        asyncio.create_task(translate_one(async_client, semaphore, idx, item.ref))
        for idx, item in enumerate(ocr_results)
    ]
    for done in asyncio.as_completed(tasks):
        idx, translated_text = await done
        print(f"TRANSLATE_ITEM[{idx}] {json.dumps(translated_text, ensure_ascii=False)}", flush=True)


async def main() -> None:
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="test-api-key",
    )
    try:
        image_base64 = image_to_base64(INPUT_PATH)

        ocr_stream = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_base64}},
                    ],
                }
            ],
            model=MODEL,
            stream=True,
        )

        adapter = TypeAdapter(list[OCRResult])
        print("=== STAGE: OCR_STREAM ===")
        ocr_results = adapter.validate_json(await stream_to_text(ocr_stream, echo=True))
        print("\n")

        print("=== STAGE: OCR ===")
        for idx, item in enumerate(ocr_results):
            print(f"OCR_ITEM[{idx}] {item.model_dump_json(ensure_ascii=False)}", flush=True)

        print("=== STAGE: TRANSLATE ===")
        await translate_with_pool(client, ocr_results)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
