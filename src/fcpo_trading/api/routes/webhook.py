from __future__ import annotations

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from fcpo_trading.api.dependencies import get_db, get_ml_service
from fcpo_trading.core.config import settings
from fcpo_trading.core.exceptions import TradingViewAuthError, ModelInferenceError
from fcpo_trading.schemas.signal import SignalResponse, TradingViewWebhookPayload
from fcpo_trading.services.ml_service import MLService
from fcpo_trading.services.signal_service import SignalService

router = APIRouter()


@router.post("/tradingview", response_model=SignalResponse)
async def tradingview_webhook(
    payload: TradingViewWebhookPayload,
    request: Request,
    x_api_key: str = Header(..., alias="X-API-KEY"),
    db: AsyncSession = Depends(get_db),
    ml_service: MLService = Depends(get_ml_service),
) -> SignalResponse:
    """Handle webhooks from TradingView strategy."""
    if x_api_key != settings.api_key_tradingview:
        raise TradingViewAuthError()

    try:
        service = SignalService(session=db, ml_service=ml_service)
        response = await service.process_webhook(payload)
    except Exception as exc:  # noqa: BLE001
        raise ModelInferenceError(details=str(exc)) from exc

    return response
