from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="Health check")
async def health_check() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}
