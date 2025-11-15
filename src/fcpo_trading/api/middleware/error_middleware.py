from __future__ import annotations

from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from fcpo_trading.core.exceptions import ModelInferenceError, TradingViewAuthError
from fcpo_trading.core.logging import logger


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling middleware."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            return await call_next(request)
        except TradingViewAuthError as exc:
            logger.warning("TradingView auth error", extra={"path": request.url.path})
            return JSONResponse(
                status_code=401,
                content={"detail": exc.message},
            )
        except ModelInferenceError as exc:
            logger.error(
                "Model inference error",
                extra={"path": request.url.path, "details": str(exc.details)},
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Model inference failed"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unhandled exception", extra={"path": request.url.path})
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )
