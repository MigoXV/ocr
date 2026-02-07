import base64
import re
from typing import AsyncIterator, Literal, Optional

from openai import AsyncOpenAI

from .prompts import build_translate_system_prompt


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

    async def ocr_stream(self, image_bytes: bytes) -> AsyncIterator[str]:
        """Run OCR with image bytes and yield streamed text chunks."""
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")

        image_url = self._bytes_to_data_uri(image_bytes)
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
        model: Optional[str] = None,
    ) -> str:
        del model
        parsed = self.parse_translate_command(user_prompt)
        if not parsed:
            return default_language
        language, _ = parsed
        return language if language else default_language

    async def resolve_chat_task(
        self,
        user_prompt: str,
        has_image: bool,
        model: Optional[str] = None,
    ) -> Literal["ocr", "ocr_translate", "translate", "invalid"]:
        del model
        prompt = user_prompt.strip()
        if not prompt:
            return "ocr" if has_image else "invalid"

        is_translate = self.parse_translate_command(prompt) is not None
        if has_image:
            return "ocr_translate" if is_translate else "ocr"
        return "translate" if is_translate else "invalid"

    @staticmethod
    def parse_translate_command(user_prompt: str) -> Optional[tuple[str, str]]:
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

    @staticmethod
    def _bytes_to_data_uri(image_bytes: bytes) -> str:
        mime_type = OpenAIOCRInferencer._detect_mime_type(image_bytes)
        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{base64_data}"

    @staticmethod
    def _detect_mime_type(image_bytes: bytes) -> str:
        if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if image_bytes.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if image_bytes.startswith((b"GIF87a", b"GIF89a")):
            return "image/gif"
        if image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
            return "image/webp"
        return "image/png"
