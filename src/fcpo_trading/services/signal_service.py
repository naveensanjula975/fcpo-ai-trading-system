from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from fcpo_trading.schemas.signal import SignalResponse, TradingViewWebhookPayload
from fcpo_trading.services.repositories import SignalRepository
from fcpo_trading.services.strategy import AIMLSignalStrategy, RuleBasedSignalStrategy
from fcpo_trading.services.ml_service import MLService
from fcpo_trading.core.config import settings


class SignalService:
    """Service orchestrating signal generation and persistence."""

    def __init__(
        self,
        session: AsyncSession,
        ml_service: MLService,
    ) -> None:
        self._repo = SignalRepository(session)
        # Choose strategy via config or fallback
        self._strategy = AIMLSignalStrategy(ml_service)
        self._fallback_strategy = RuleBasedSignalStrategy()

    async def process_webhook(
        self,
        payload: TradingViewWebhookPayload,
    ) -> SignalResponse:
        data: Dict[str, Any] = payload.model_dump()
        try:
            result = self._strategy.generate(data)
        except Exception as exc:  # noqa: BLE001
            # Fallback if ML fails
            result = self._fallback_strategy.generate(data)

        signal_id = uuid4()
        now = datetime.utcnow()
        response = SignalResponse(
            signal_id=signal_id,
            signal=result["signal"],
            confidence=result["confidence"],
            entry_price=result["entry_price"],
            tp_levels=result["tp_levels"],
            sl_level=result["sl_level"],
            timestamp=now,
            model_version=settings.model_version,
        )

        await self._repo.create(response)
        return response
