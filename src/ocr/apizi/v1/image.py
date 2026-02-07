from fastapi import APIRouter, File, Form, UploadFile

from ocr.servicer.servicer import ImageEditService

router = APIRouter()
image_edit_service = ImageEditService()


@router.post("/v1/images/edits")
async def create_image_edit(
    image: UploadFile = File(...),
    prompt: str = Form(""),
    model: str = Form("gpt-image-1"),
    response_format: str = Form("b64_json"),
):
    image_bytes = await image.read()
    return await image_edit_service.edit_image(
        image_bytes=image_bytes,
        prompt=prompt,
        model=model,
        response_format=response_format,
    )

