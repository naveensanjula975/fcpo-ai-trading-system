from __future__ import annotations

from collections.abc import AsyncGenerator

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from fcpo_trading.core.config import settings
from fcpo_trading.models.database import get_db_session
from fcpo_trading.services.ml_service import MLService


async def get_redis_client() -> AsyncGenerator[Redis, None]:
    redis = Redis.from_url(settings.redis_url, encoding="utf-8", decode_responses=True)
    try:
        yield redis
    finally:
        await redis.aclose()


def get_ml_service() -> MLService:
    # Stateless wrapper
    return MLService()


# Re-export DB session dependency
get_db = get_db_session
