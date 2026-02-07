from typing import Annotated, Any, Optional

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import APIStatusError, AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ocr.inferencers import OpenAIOCRInferencer

router = APIRouter()
client = AsyncOpenAI()
_default_inferencer = OpenAIOCRInferencer(client=client)

Inferencer = OpenAIOCRInferencer


def get_inferencer() -> Inferencer:
    return _default_inferencer


@router.post("/v1/chat/completions", response_model=None)
async def create_chat_completion(
    inferencer: Annotated[Inferencer, Depends(get_inferencer)],
    messages: Annotated[list[ChatCompletionMessageParam], Body(..., embed=True)],
    model: Annotated[str, Body(..., embed=True)],
    max_tokens: Annotated[Optional[int], Body(embed=True)] = None,
    stream: Annotated[bool, Body(embed=True)] = False,
) -> Any:
    try:
        response = await inferencer.create_chat_completion_response(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            stream=stream,
        )
        if stream:
            return StreamingResponse(response, media_type="text/event-stream")
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
