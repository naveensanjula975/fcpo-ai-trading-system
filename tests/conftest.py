from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from fcpo_trading.main import create_app


@pytest.fixture(scope="session")
def app() -> FastAPI:
    return create_app()


@pytest.fixture
async def client(app: FastAPI) -> AsyncGenerator[AsyncClient, Any]:
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac
