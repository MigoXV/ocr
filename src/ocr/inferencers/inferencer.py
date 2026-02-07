import json
import logging
from typing import Any, AsyncIterator, Literal, Optional

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from ocr.types.ocr_results import OCRResult, OCRResults

from .ocr_parser import parse_raw_str, parse_raw_str_stream
from .prompts import build_translate_system_prompt
from .utils import (
    bytes_to_data_uri,
    build_chat_completion,
    extract_completion_text,
    extract_message_text,
    has_image,
    iter_json_array_sse,
    iter_ocr_item_jsons,
    iter_single_json_item,
    map_upstream_model,
    parse_translate_command,
    resolve_request_mode,
    serialize_ocr_results,
)

logger = logging.getLogger(__name__)

DEFAULT_TARGET_LANGUAGE = "法语"
DEFAULT_PROMPT_PARSE_MODEL = "Qwen/Qwen3-8B"
OCR_UX_MODEL = "deepseek-ocr2-ux"
INTERNAL_OCR_MODEL = "deepseek-ocr2"


class OpenAIOCRInferencer:
    """OpenAI-compatible OCR inferencer."""

    def __init__(
        self,
        model: str = "deepseek-ocr2",
        prompt: str = "OCR this image.",
        translate_model: str = "tencent/Hunyuan-MT-7B",
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.translate_model = translate_model
        self.client = client or AsyncOpenAI()

    async def create_chat_completion_response(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[str]:
        has_image_input = has_image(messages)
        request_mode = resolve_request_mode(model, ocr_ux_model=OCR_UX_MODEL)
        system_text = extract_message_text(messages[0])
        user_text = extract_message_text(messages[1])
        should_translate = self._should_translate(
            system_text=system_text,
            user_text=user_text,
        )

        if request_mode == "mt":
            if has_image_input:
                raise ValueError(
                    f"Model '{model}' is text MT only and does not support image input."
                )

            translated = await self._run_translate_pipeline(
                user_text=user_text,
                system_text=system_text,
                translate_model=model,
            )
            if stream:
                return iter_json_array_sse(
                    iter_single_json_item(translated),
                    model=model,
                )
            content = json.dumps([translated], ensure_ascii=False)
            return build_chat_completion(content=content, model=model)

        if not has_image_input:
            raise ValueError(
                f"Model '{OCR_UX_MODEL}' requires image input and supports only OCR/OCR+translate."
            )

        if should_translate:
            if stream:
                upstream_stream = await self._create_upstream_stream(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                )
                ocr_results = await self._collect_ocr_results_from_stream(upstream_stream)
                translated = await self._translate_ocr_results(
                    ocr_results=ocr_results,
                    user_text=user_text,
                    system_text=system_text,
                )
                return iter_json_array_sse(iter_ocr_item_jsons(translated), model=model)

            upstream_response = await self._create_upstream_non_stream(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
            )
            raw_text = extract_completion_text(upstream_response)
            ocr_results = parse_raw_str(raw_text)
            translated = await self._translate_ocr_results(
                ocr_results=ocr_results,
                user_text=user_text,
                system_text=system_text,
            )
            content = serialize_ocr_results(translated)
            return build_chat_completion(content=content, model=model)

        if stream:
            upstream_stream = await self._create_upstream_stream(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
            )
            return iter_json_array_sse(
                self._iter_ocr_item_jsons_from_stream(upstream_stream),
                model=model,
            )

        upstream_response = await self._create_upstream_non_stream(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
        )
        raw_text = extract_completion_text(upstream_response)
        ocr_results = parse_raw_str(raw_text)
        content = serialize_ocr_results(ocr_results)
        return build_chat_completion(content=content, model=model)

    async def ocr_stream(self, image_bytes: bytes) -> AsyncIterator[str]:
        """Run OCR with image bytes and yield streamed text chunks."""
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")

        image_url = bytes_to_data_uri(image_bytes)
        response = await self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            model=self.model,
            stream=True,
        )

        async for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def translate_stream(
        self,
        text: str,
        target_language: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        prompt = system_prompt or build_translate_system_prompt(
            target_language=target_language
        )
        response = await self.client.chat.completions.create(
            model=model if model else self.translate_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def resolve_target_language(
        self,
        user_prompt: str,
        default_language: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        del model
        parsed = parse_translate_command(system_prompt or "")
        if not parsed:
            parsed = parse_translate_command(user_prompt)
        if not parsed:
            return default_language
        language, _ = parsed
        return language if language else default_language

    async def resolve_chat_task(
        self,
        user_prompt: str,
        has_image: bool,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Literal["ocr", "ocr_translate", "translate", "invalid"]:
        del model
        prompt = user_prompt.strip()
        if not prompt and not (system_prompt or "").strip():
            return "ocr" if has_image else "invalid"

        is_translate = parse_translate_command(system_prompt or "") is not None
        if not is_translate:
            is_translate = parse_translate_command(prompt) is not None
        if has_image:
            return "ocr_translate" if is_translate else "ocr"
        return "translate" if is_translate else "invalid"

    async def summarize_translation_context(
        self,
        text: str,
        target_language: str,
        model: Optional[str] = None,
    ) -> str:
        if not text.strip():
            return build_translate_system_prompt(target_language=target_language)

        system_prompt = (
            "你是翻译任务提示词压缩器。"
            "请根据文档全文生成一段很短的翻译系统提示词，"
            "用于指导逐句翻译到目标语言。"
            "要求：不超过120字，强调术语一致、语气一致、格式保留。"
            "只输出提示词本身。"
        )
        response = await self.client.chat.completions.create(
            model=model if model else self.translate_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"目标语言：{target_language}\n文档全文：\n{text}\n/no_think",
                },
            ],
            temperature=0.2,
            max_tokens=200,
            stream=False,
        )
        content = response.choices[0].message.content if response.choices else None
        summary = content.strip() if content else ""
        if not summary:
            return build_translate_system_prompt(target_language=target_language)
        return summary

    async def translate_text(
        self,
        text: str,
        target_language: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        if not text.strip():
            return text
        prompt = system_prompt or build_translate_system_prompt(
            target_language=target_language
        )
        response = await self.client.chat.completions.create(
            model=model if model else self.translate_model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            return text
        return content.strip()

    async def _create_upstream_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        max_tokens: Optional[int] = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": map_upstream_model(
                model,
                ocr_ux_model=OCR_UX_MODEL,
                internal_ocr_model=INTERNAL_OCR_MODEL,
            ),
            "stream": True,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return await self.client.chat.completions.create(**kwargs)

    async def _create_upstream_non_stream(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        max_tokens: Optional[int] = None,
    ) -> ChatCompletion:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "model": map_upstream_model(
                model,
                ocr_ux_model=OCR_UX_MODEL,
                internal_ocr_model=INTERNAL_OCR_MODEL,
            ),
            "stream": False,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return await self.client.chat.completions.create(**kwargs)

    def _should_translate(self, system_text: str, user_text: str) -> bool:
        if parse_translate_command(system_text):
            return True
        return parse_translate_command(user_text) is not None

    async def _run_translate_pipeline(
        self,
        user_text: str,
        system_text: str,
        translate_model: str | None = None,
    ) -> str:
        parsed = parse_translate_command(user_text)
        source_text = user_text
        if parsed:
            source_text = parsed[1]
        if not source_text.strip():
            source_text = user_text

        target_language = await self.resolve_target_language(
            user_prompt=user_text,
            default_language=DEFAULT_TARGET_LANGUAGE,
            system_prompt=system_text,
            model=DEFAULT_PROMPT_PARSE_MODEL,
        )
        return await self.translate_text(
            text=source_text,
            target_language=target_language,
            model=translate_model,
        )

    async def _collect_ocr_results_from_stream(
        self,
        chunks: AsyncIterator[ChatCompletionChunk],
    ) -> OCRResults:
        results: OCRResults = []
        async for item in parse_raw_str_stream(self._iter_ocr_text(chunks)):
            results.append(item)
        return results

    async def _translate_ocr_results(
        self,
        ocr_results: OCRResults,
        user_text: str,
        system_text: str,
    ) -> OCRResults:
        target_language = await self.resolve_target_language(
            user_prompt=user_text,
            default_language=DEFAULT_TARGET_LANGUAGE,
            system_prompt=system_text,
            model=DEFAULT_PROMPT_PARSE_MODEL,
        )
        full_context = "\n".join(item.ref for item in ocr_results)
        system_prompt = await self.summarize_translation_context(
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
                    target_text = await self.translate_text(
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

    async def _iter_ocr_text(
        self,
        chunks: AsyncIterator[ChatCompletionChunk],
    ) -> AsyncIterator[str]:
        async for chunk in chunks:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

    async def _iter_ocr_item_jsons_from_stream(
        self,
        chunks: AsyncIterator[ChatCompletionChunk],
    ) -> AsyncIterator[str]:
        async for item in parse_raw_str_stream(self._iter_ocr_text(chunks)):
            yield json.dumps(item.model_dump(mode="json"), ensure_ascii=False)
