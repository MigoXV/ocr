import json
import logging
import time
from typing import Any, AsyncIterator

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIStatusError, AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from ocr.inferencers import OpenAIOCRInferencer
from ocr.inferencers.ocr_parser import parse_raw_str, parse_raw_str_stream
from ocr.types.ocr_results import OCRResult, OCRResults

router = APIRouter()
client = AsyncOpenAI()
inferencer = OpenAIOCRInferencer(client=client)
logger = logging.getLogger(__name__)

DEFAULT_TARGET_LANGUAGE = "法语"
DEFAULT_PROMPT_PARSE_MODEL = "Qwen/Qwen3-8B"


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    payload: dict[str, Any] = Body(...),
) -> Any:
    try:
        messages = _require_messages(payload)
        has_image = _has_image(messages)
        user_text = _extract_last_user_text(messages)
        stream = bool(payload.get("stream"))
        model = str(payload.get("model") or "deepseek-ocr2")

        task = await inferencer.resolve_chat_task(
            user_prompt=user_text,
            has_image=has_image,
            model=DEFAULT_PROMPT_PARSE_MODEL,
        )

        if not has_image:
            if task != "translate":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "No image provided and request is not a translation task. "
                        "This endpoint does not support general chat."
                    ),
                )
            translated = await _run_translate_pipeline(user_text=user_text)
            content = json.dumps([translated], ensure_ascii=False)
            if stream:
                return StreamingResponse(
                    _iter_json_array_sse(_iter_single_json_item(translated), model=model),
                    media_type="text/event-stream",
                )
            return _build_chat_completion(content=content, model=model)

        if task == "ocr_translate":
            if stream:
                upstream_stream = await _create_upstream_stream(payload)
                ocr_results = await _collect_ocr_results_from_stream(upstream_stream)
                translated = await _translate_ocr_results(ocr_results, user_text=user_text)
                return StreamingResponse(
                    _iter_json_array_sse(_iter_ocr_item_jsons(translated), model=model),
                    media_type="text/event-stream",
                )

            upstream_response = await _create_upstream_non_stream(payload)
            raw_text = _extract_completion_text(upstream_response)
            ocr_results = parse_raw_str(raw_text)
            translated = await _translate_ocr_results(ocr_results, user_text=user_text)
            content = _serialize_ocr_results(translated)
            return _build_chat_completion(content=content, model=model)

        # Default: OCR pipeline
        if stream:
            upstream_stream = await _create_upstream_stream(payload)
            return StreamingResponse(
                _iter_json_array_sse(_iter_ocr_item_jsons_from_stream(upstream_stream), model=model),
                media_type="text/event-stream",
            )

        upstream_response = await _create_upstream_non_stream(payload)
        raw_text = _extract_completion_text(upstream_response)
        ocr_results = parse_raw_str(raw_text)
        content = _serialize_ocr_results(ocr_results)
        return _build_chat_completion(content=content, model=model)
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
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _require_messages(payload: dict[str, Any]) -> list[dict[str, Any]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise HTTPException(status_code=400, detail="messages must be a list.")
    return [msg for msg in messages if isinstance(msg, dict)]


def _has_image(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
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


async def _create_upstream_stream(
    payload: dict[str, Any],
) -> AsyncIterator[ChatCompletionChunk]:
    request_payload = dict(payload)
    request_payload["stream"] = True
    return await client.chat.completions.create(**request_payload)


async def _create_upstream_non_stream(payload: dict[str, Any]) -> ChatCompletion:
    request_payload = dict(payload)
    request_payload["stream"] = False
    return await client.chat.completions.create(**request_payload)


def _extract_completion_text(response: ChatCompletion) -> str:
    if not response.choices:
        return ""
    message = response.choices[0].message
    content = message.content if message else None
    if not isinstance(content, str):
        return ""
    return content


def _serialize_ocr_results(ocr_results: OCRResults) -> str:
    return json.dumps(
        [item.model_dump(mode="json") for item in ocr_results],
        ensure_ascii=False,
    )


async def _run_translate_pipeline(user_text: str) -> str:
    target_language = await inferencer.resolve_target_language(
        user_prompt=user_text,
        default_language=DEFAULT_TARGET_LANGUAGE,
        model=DEFAULT_PROMPT_PARSE_MODEL,
    )
    return await inferencer.translate_text(
        text=user_text,
        target_language=target_language,
    )


async def _collect_ocr_results_from_stream(
    chunks: AsyncIterator[ChatCompletionChunk],
) -> OCRResults:
    results: OCRResults = []
    async for item in parse_raw_str_stream(_iter_ocr_text(chunks)):
        results.append(item)
    return results


async def _translate_ocr_results(
    ocr_results: OCRResults,
    user_text: str,
) -> OCRResults:
    target_language = await inferencer.resolve_target_language(
        user_prompt=user_text,
        default_language=DEFAULT_TARGET_LANGUAGE,
        model=DEFAULT_PROMPT_PARSE_MODEL,
    )
    full_context = "\n".join(item.ref for item in ocr_results)
    system_prompt = await inferencer.summarize_translation_context(
        text=full_context,
        target_language=target_language,
        model=DEFAULT_PROMPT_PARSE_MODEL,
    )

    cache: dict[str, str] = {}
    translated: OCRResults = []
    for item in ocr_results:
        source_text = item.ref
        if source_text in cache:
            target_text = cache[source_text]
        else:
            try:
                target_text = await inferencer.translate_text(
                    text=source_text,
                    target_language=target_language,
                    system_prompt=system_prompt,
                )
            except Exception as exc:
                logger.warning("Translate failed for item '%s': %s", source_text, exc)
                target_text = source_text
            cache[source_text] = target_text
        translated.append(OCRResult(ref=target_text, det=item.det))
    return translated


async def _iter_ocr_text(chunks: AsyncIterator[ChatCompletionChunk]) -> AsyncIterator[str]:
    async for chunk in chunks:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            yield content


async def _iter_ocr_item_jsons_from_stream(
    chunks: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[str]:
    async for item in parse_raw_str_stream(_iter_ocr_text(chunks)):
        yield json.dumps(item.model_dump(mode="json"), ensure_ascii=False)


async def _iter_ocr_item_jsons(items: OCRResults) -> AsyncIterator[str]:
    for item in items:
        yield json.dumps(item.model_dump(mode="json"), ensure_ascii=False)


async def _iter_single_json_item(text: str) -> AsyncIterator[str]:
    yield json.dumps(text, ensure_ascii=False)


def _build_chat_completion(content: str, model: str) -> dict[str, Any]:
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


async def _iter_json_array_sse(
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
    yield f"data: {json.dumps(start_payload, ensure_ascii=False)}\n\n"
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
    yield f"data: {json.dumps(end_payload, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


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
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

