from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from fcpo_trading.api.dependencies import get_db
from fcpo_trading.schemas.signal import SignalResponse
from fcpo_trading.services.repositories import SignalRepository

router = APIRouter()


@router.get("/history", response_model=List[SignalResponse])
async def list_signals(
    db: AsyncSession = Depends(get_db),
    signal_type: Optional[str] = Query(None),
    start: Optional[datetime] = Query(None),
    end: Optional[datetime] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> List[SignalResponse]:
    """Retrieve historical signals with optional filters."""
    repo = SignalRepository(db)
    records = await repo.list(
        signal_type=signal_type,
        start=start,
        end=end,
        limit=limit,
        offset=offset,
    )
    responses: List[SignalResponse] = []
    for s in records:
        tp_levels = [v for v in [s.tp1, s.tp2, s.tp3] if v is not None]
        responses.append(
            SignalResponse(
                signal_id=s.id,
                signal=s.signal_type,
                confidence=float(s.confidence),
                entry_price=float(s.entry_price),
                tp_levels=[float(v) for v in tp_levels],
                sl_level=float(s.sl),
                timestamp=s.timestamp,
                model_version=s.model_version,
            )
        )
    return responses
