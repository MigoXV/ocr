from fastapi import APIRouter

from ocr.api.v1.chat import router as chat_router
from ocr.api.v1.image import router as image_router

router = APIRouter()
router.include_router(image_router)
router.include_router(chat_router)
