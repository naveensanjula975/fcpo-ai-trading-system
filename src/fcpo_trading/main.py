from __future__ import annotations

from fastapi import FastAPI

from fcpo_trading.api.routes import health, metrics, signals, webhook
from fcpo_trading.api.middleware.logging_middleware import LoggingMiddleware
from fcpo_trading.api.middleware.error_middleware import ErrorHandlingMiddleware
from fcpo_trading.core.config import settings
from fcpo_trading.core.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
    )

    # Middlewares (Observer for requests/errors)
    app.add_middleware(LoggingMiddleware)
    app.add_middleware(ErrorHandlingMiddleware)

    # Routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(metrics.router, prefix="/metrics", tags=["metrics"])
    app.include_router(signals.router, prefix="/signals", tags=["signals"])
    app.include_router(webhook.router, prefix="/webhook", tags=["webhook"])

    return app


app = create_app()
