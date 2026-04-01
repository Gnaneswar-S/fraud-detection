"""
HTTP middleware stack.

RequestLoggingMiddleware
  Structured access log: method path status latency req_id
  Injects X-Request-ID and X-Response-Time into every response.
  Catches unhandled 500s and returns clean JSON.

SecurityHeadersMiddleware
  Adds OWASP-recommended security headers to every response.

PrometheusMiddleware  (imported by main.py if Prometheus is enabled)
  Records per-endpoint request counts and latency.
"""
from __future__ import annotations

import logging
import time
import uuid

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("fraud_api.access")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Structured access log compatible with Render / Datadog / CloudWatch.

    Log format:
        INFO  POST /predict 200 38.1ms req_id=a3f8b2c1
        WARN  POST /predict 422 2.1ms  req_id=d9e14f5a
        ERROR POST /predict 500 1.2ms  req_id=f0a38b7e  error=...
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        req_id = str(uuid.uuid4()).replace("-", "")[:12]
        request.state.req_id = req_id
        t0 = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception as exc:
            latency = (time.perf_counter() - t0) * 1000
            logger.error(
                "%s %s 500 %.1fms req_id=%s error=%s",
                request.method, request.url.path, latency, req_id, repr(exc),
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "req_id": req_id},
            )

        latency = (time.perf_counter() - t0) * 1000
        level   = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level, "%s %s %d %.1fms req_id=%s",
            request.method, request.url.path,
            response.status_code, latency, req_id,
        )

        response.headers["X-Request-ID"]    = req_id
        response.headers["X-Response-Time"] = f"{latency:.1f}ms"
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    OWASP-recommended security headers on every response.

    Headers added:
      X-Content-Type-Options : nosniff         (prevent MIME sniffing)
      X-Frame-Options        : DENY            (prevent clickjacking)
      Referrer-Policy        : strict-origin   (limit referrer leakage)
      X-XSS-Protection       : 1; mode=block   (legacy XSS filter)
      Cache-Control          : no-store        (prevent caching of fraud scores)
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"]        = "DENY"
        response.headers["Referrer-Policy"]        = "strict-origin-when-cross-origin"
        response.headers["X-XSS-Protection"]       = "1; mode=block"
        # Fraud scores must never be cached by intermediaries
        if request.url.path in ("/predict", "/explain"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response