from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class Bar(BaseModel):
    open: float
    high: float
    low: float
    close: float


class TradingViewWebhookPayload(BaseModel):
    ticker: str = "FCPO"
    action: Literal["buy", "sell", "none"] = "none"
    price: float
    volume: float
    time: datetime
    interval: str
    bar: Bar


class SignalResponse(BaseModel):
    signal_id: UUID
    signal: Literal["BUY", "SELL", "HOLD"]
    confidence: float = Field(ge=0, le=100)
    entry_price: float
    tp_levels: list[float]
    sl_level: float
    timestamp: datetime
    model_version: str
