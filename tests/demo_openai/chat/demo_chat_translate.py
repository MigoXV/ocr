from openai import OpenAI
from pydantic import TypeAdapter

INPUT_TEXT = "请把这句话翻译成法语：今天阳光很好，我们去散步吧。"
MODEL = "deepseek-ocr2"


if __name__ == "__main__":
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="test-api-key",
    )

    stream = client.chat.completions.create(
        messages=[{"role": "user", "content": INPUT_TEXT}],
        model=MODEL,
        stream=True,
    )

    output = ""
    for chunk in stream:
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

