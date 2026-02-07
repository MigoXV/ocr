import json
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIStatusError, AsyncOpenAI
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from ocr.inferencers.ocr_parser import parse_raw_str_stream

router = APIRouter()
client = AsyncOpenAI()


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    payload: dict[str, Any] = Body(...),
) -> Any:
    try:
        _require_image_input(payload)
        stream = bool(payload.get("stream"))
        response = await client.chat.completions.create(**payload)
        if not stream:
            return response

        return StreamingResponse(
            _iter_ocr_sse(response),
            media_type="text/event-stream",
        )
    except APIStatusError as exc:
        message = ""
        try:
            if exc.response and getattr(exc.response, "text", None):
                message = exc.response.text
        except Exception:
            message = ""
        if not message:
            message = str(exc)
        raise HTTPException(status_code=exc.status_code, detail=message) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _require_image_input(payload: dict[str, Any]) -> None:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list.")

    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return

    raise HTTPException(
        status_code=400,
        detail="OCR endpoint requires image input in messages[].content[].type=image_url.",
    )


async def _iter_ocr_text(chunks: AsyncIterator[ChatCompletionChunk]) -> AsyncIterator[str]:
    async for chunk in chunks:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


async def _iter_ocr_sse(chunks: AsyncIterator[ChatCompletionChunk]) -> AsyncIterator[str]:
    chunk_id = "chatcmpl-ocr-parser"
    model = "deepseek-ocr2"
    created = int(time.time())

    # Initial role delta chunk for OpenAI-compatible stream parsers.
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
    yield f"data: {json.dumps(start_payload, ensure_ascii=False)}\n\n"

    open_array_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "["},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(open_array_payload, ensure_ascii=False)}\n\n"

    first_item = True
    async for item in parse_raw_str_stream(_iter_ocr_text(chunks)):
        content = json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
        if not first_item:
            content = "," + content
        first_item = False

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
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    close_array_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "]"},
                "finish_reason": None,
            }
        ],
    }
    yield f"data: {json.dumps(close_array_payload, ensure_ascii=False)}\n\n"

    end_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }
        ],
    }
    yield f"data: {json.dumps(end_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
