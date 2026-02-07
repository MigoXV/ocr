import base64
import json
import re
import time
from typing import Any, AsyncIterator

from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion

from ocr.types.ocr_results import OCRResults


def message_get(message: ChatCompletionMessageParam, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    return getattr(message, key, None)


def has_image(messages: list[ChatCompletionMessageParam]) -> bool:
    for message in messages:
        content = message_get(message, "content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def extract_last_role_text(
    messages: list[ChatCompletionMessageParam],
    role: str,
) -> str:
    for message in reversed(messages):
        if message_get(message, "role") != role:
            continue
        content = message_get(message, "content")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            joined = "\n".join(part for part in parts if part.strip())
            return joined.strip()
    return ""


def serialize_ocr_results(ocr_results: OCRResults) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in ocr_results],
        ensure_ascii=False,
    )


def build_chat_completion(content: str, model: str) -> dict[str, Any]:
    created = int(time.time())
    return {
        "id": f"chatcmpl-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


async def iter_single_json_item(text: str) -> AsyncIterator[str]:
    yield json.dumps(text, ensure_ascii=False)


async def iter_ocr_item_jsons(items: OCRResults) -> AsyncIterator[str]:
    for item in items:
        yield json.dumps(item.model_dump(mode="json"), ensure_ascii=False)


async def iter_json_array_sse(
    item_json_iter: AsyncIterator[str],
    model: str,
) -> AsyncIterator[str]:
    chunk_id = f"chatcmpl-{int(time.time())}"
    created = int(time.time())

    start_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(start_payload, ensure_ascii=False)}\\n\\n"
    yield _format_chunk_content(chunk_id, created, model, "[")

    first_item = True
    async for item_json in item_json_iter:
        content = item_json if first_item else f",{item_json}"
        first_item = False
        yield _format_chunk_content(chunk_id, created, model, content)

    yield _format_chunk_content(chunk_id, created, model, "]")

    end_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(end_payload, ensure_ascii=False)}\\n\\n"
    yield "data: [DONE]\\n\\n"


def _format_chunk_content(chunk_id: str, created: int, model: str, content: str) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content},
                "finish_reason": None,
            }
        ],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\\n\\n"


def parse_translate_command(user_prompt: str) -> tuple[str, str] | None:
    """
    Parse '/translate <language>' command at the beginning of prompt.
    Returns (language, content_after_command) when matched, else None.
    """
    if not user_prompt:
        return None
    match = re.match(
        r"^\s*/translate\s+([^\s]+)(?:\s+([\s\S]*))?\s*$",
        user_prompt,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    language = (match.group(1) or "").strip()
    content = (match.group(2) or "").strip()
    if not language:
        return None
    return language, content


def resolve_request_mode(model: str, ocr_ux_model: str) -> str:
    normalized = (model or "").strip()
    if normalized == ocr_ux_model:
        return "ocr_ux"
    if "-MT" in normalized.upper():
        return "mt"
    raise ValueError(
        f"Unsupported model '{model}'. Use '{ocr_ux_model}' for OCR/OCR+translate, "
        "or a model containing '-MT' for text machine translation."
    )


def map_upstream_model(model: str, ocr_ux_model: str, internal_ocr_model: str) -> str:
    if model == ocr_ux_model:
        return internal_ocr_model
    return model


def extract_completion_text(response: ChatCompletion) -> str:
    if not response.choices:
        return ""
    message = response.choices[0].message
    content = message.content if message else None
    if not isinstance(content, str):
        return ""
    return content


def bytes_to_data_uri(image_bytes: bytes) -> str:
    mime_type = detect_mime_type(image_bytes)
    base64_data = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"


def detect_mime_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"
