import asyncio

from openai import AsyncOpenAI
from pydantic import TypeAdapter

INPUT_TEXT = "/translate 法语 今天阳光很好，我们去散步吧。"
MODEL = "tencent/Hunyuan-MT-7B"


async def main() -> None:
    client = AsyncOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="test-api-key",
    )
    try:
        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": INPUT_TEXT}],
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

        adapter = TypeAdapter(list[str])
        results = adapter.validate_json(output)
        print("Translate Result:", results)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
