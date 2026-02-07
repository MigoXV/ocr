from fastapi import APIRouter

from ocr.apizi.v1.chat import router as chat_router
from ocr.apizi.v1.image import router as image_router

router = APIRouter()
router.include_router(image_router)
router.include_router(chat_router)
