from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from fcpo_trading.core.logging import logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Log each request with latency."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration = (time.perf_counter() - start) * 1000

        logger.info(
            "request completed",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration, 2),
            },
        )

        return response
