from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="Metrics placeholder")
async def metrics() -> dict:
    """Placeholder for Prometheus or custom metrics."""
    return {"message": "metrics endpoint (to integrate with Prometheus later)"}
