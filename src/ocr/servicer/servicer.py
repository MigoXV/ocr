import base64
import io
import time
from typing import Any

from openai.types.images_response import ImagesResponse
from PIL import Image

from ocr.image.render import render_ocr_results
from ocr.inferencers import OpenAIOCRInferencer
from ocr.inferencers.ocr_parser import get_plain_text, parse_plain_text, parse_raw_str

DEFAULT_OCR_MODEL = "deepseek-ocr2"
DEFAULT_TRANSLATE_MODEL = "MedAIBase/Tencent-HY-MT1.5:7b"
DEFAULT_PROMPT_PARSE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_TARGET_LANGUAGE = "法语"


class ImageEditService:
    def __init__(
        self,
        ocr_model: str = DEFAULT_OCR_MODEL,
        translate_model: str = DEFAULT_TRANSLATE_MODEL,
        prompt_parse_model: str = DEFAULT_PROMPT_PARSE_MODEL,
    ) -> None:
        self.inferencer = OpenAIOCRInferencer(
            model=ocr_model,
            translate_model=translate_model,
        )
        self.prompt_parse_model = prompt_parse_model

    async def edit_image(
        self,
        image_bytes: bytes,
        prompt: str = "",
        model: str = "gpt-image-1",
        response_format: str = "b64_json",
    ):
        del model

        if response_format != "b64_json":
            raise ValueError("Only response_format='b64_json' is supported.")
        if not image_bytes:
            raise ValueError("image_bytes cannot be empty.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ocr_raw_text = await self._collect_stream_text(
            self.inferencer.ocr_stream(image_bytes)
        )
        ocr_results = parse_raw_str(ocr_raw_text)
        plain_text = get_plain_text(ocr_results)
        target_language = await self.inferencer.resolve_target_language(
            user_prompt=prompt,
            default_language=DEFAULT_TARGET_LANGUAGE,
            model=self.prompt_parse_model,
        )
        translated_text = await self._collect_stream_text(
            self.inferencer.translate_stream(
                text=plain_text,
                target_language=target_language,
            )
        )
        translated_results = parse_plain_text(translated_text, ocr_results)
        rendered_image = render_ocr_results(image, translated_results)
        rendered_b64 = self._to_png_b64(rendered_image)
        return self._build_images_response(rendered_b64)

    async def _collect_stream_text(self, text_stream: Any) -> str:
        chunks: list[str] = []
        async for content in text_stream:
            chunks.append(content)
        return "".join(chunks)

    @staticmethod
    def _to_png_b64(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    @staticmethod
    def _build_images_response(image_b64: str):
        payload = {
            "created": int(time.time()),
            "data": [{"b64_json": image_b64}],
        }
        if hasattr(ImagesResponse, "model_validate"):
            return ImagesResponse.model_validate(payload)
        if hasattr(ImagesResponse, "parse_obj"):
            return ImagesResponse.parse_obj(payload)
        return ImagesResponse(**payload)

