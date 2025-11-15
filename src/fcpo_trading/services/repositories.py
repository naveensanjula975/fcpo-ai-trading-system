from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from fcpo_trading.models.signal import Signal
from fcpo_trading.schemas.signal import SignalResponse


class SignalRepository:
    """Repository for reading/writing signal records."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        response: SignalResponse,
    ) -> Signal:
        signal = Signal(
            id=response.signal_id,
            timestamp=response.timestamp,
            ticker="FCPO",
            signal_type=response.signal,
            entry_price=response.entry_price,
            tp1=response.tp_levels[0] if response.tp_levels else None,
            tp2=response.tp_levels[1] if len(response.tp_levels) > 1 else None,
            tp3=response.tp_levels[2] if len(response.tp_levels) > 2 else None,
            sl=response.sl_level,
            confidence=response.confidence,
            model_version=response.model_version,
            status="ACTIVE",
            created_at=datetime.utcnow(),
        )
        self._session.add(signal)
        await self._session.commit()
        await self._session.refresh(signal)
        return signal

    async def get_by_id(self, signal_id: UUID) -> Optional[Signal]:
        stmt = select(Signal).where(Signal.id == signal_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def list(
        self,
        *,
        signal_type: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Iterable[Signal]:
        stmt = select(Signal).order_by(Signal.created_at.desc())
        if signal_type:
            stmt = stmt.where(Signal.signal_type == signal_type)
        if start:
            stmt = stmt.where(Signal.created_at >= start)
        if end:
            stmt = stmt.where(Signal.created_at <= end)
        stmt = stmt.offset(offset).limit(limit)
        result = await self._session.execute(stmt)
        return result.scalars().all()
