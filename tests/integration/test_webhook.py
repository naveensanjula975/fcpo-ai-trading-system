from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from fcpo_trading.core.config import settings


@pytest.mark.asyncio
async def test_tradingview_webhook_ok(client: AsyncClient) -> None:
    payload = {
        "ticker": "FCPO",
        "action": "buy",
        "price": 3850.0,
        "volume": 1250,
        "time": datetime.now(timezone.utc).isoformat(),
        "interval": "5",
        "bar": {
            "open": 3845.0,
            "high": 3852.0,
            "low": 3843.0,
            "close": 3850.0,
        },
    }

    headers = {"X-API-KEY": settings.api_key_tradingview}
    response = await client.post("/webhook/tradingview", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["signal"] in ("BUY", "SELL", "HOLD")
