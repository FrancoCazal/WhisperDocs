from fastapi import APIRouter, Response

from app.metrics import render_metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> Response:
    body, content_type = render_metrics()
    return Response(content=body, media_type=content_type)
