import base64
from typing import Iterator, Optional

from openai import OpenAI

from .prompts import build_translate_system_prompt


class OpenAIOCRInferencer:
    """OpenAI-compatible OCR inferencer."""

    def __init__(
        self,
        model: str = "deepseek-ocr2",
        prompt: str = "OCR this image.",
        translate_model: str = "tencent/Hunyuan-MT-7B",
        client: Optional[OpenAI] = None,
    ) -> None:
        self.model = model
        self.prompt = prompt
        self.translate_model = translate_model
        self.client = client or OpenAI()

    def ocr_stream(self, image_bytes: bytes) -> Iterator[str]:
        """Run OCR with image bytes and yield streamed text chunks."""
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty")

        image_url = self._bytes_to_data_uri(image_bytes)
        response = self.client.chat.completions.create(
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

        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def translate_stream(
        self,
        text: str,
        target_language: str,
        full_context: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """Translate text and yield streamed translation chunks."""
        if not text or len(text.strip()) == 0:
            return

        translate_model = model or self.translate_model
        system_prompt = build_translate_system_prompt(
            target_language=target_language,
            full_context=full_context,
        )

        response = self.client.chat.completions.create(
            model=translate_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in response:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content

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
