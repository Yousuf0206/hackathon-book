"""
Structured logging middleware with key metrics.
"""
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from typing import Optional
import time
import logging
import json
from datetime import datetime
import uuid
from fastapi.responses import JSONResponse


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoggingMiddleware:
    """
    Middleware to implement structured logging for API requests and responses with key metrics.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope, receive)

        # Log request
        start_time = time.time()
        request_id = str(uuid.uuid4())

        # Get request details
        request_details = {
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
            "content_length": request.headers.get("content-length", 0),
            "api_version": "v1"  # Add API version for metrics tracking
        }

        # Log the incoming request
        logger.info(json.dumps({
            "event": "request_received",
            "level": "info",
            "details": request_details
        }))

        try:
            # Process the request
            response = await self.app(scope, receive, send)

            # Calculate processing time
            process_time = time.time() - start_time

            # Log response with metrics
            process_time_ms = round(process_time * 1000, 2)
            response_details = {
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id,
                "status_code": response.status,
                "process_time_ms": process_time_ms,
                "path": request.url.path,
                "method": request.method
            }

            # Log metrics for monitoring
            logger.info(json.dumps({
                "event": "request_completed",
                "level": "info",
                "details": response_details,
                "metrics": {
                    "response_time_ms": process_time_ms,
                    "status_code": response_details["status_code"],
                    "endpoint": response_details["path"],
                    "method": request.method
                }
            }))

            return response

        except Exception as e:
            # Calculate processing time for error case
            process_time = time.time() - start_time

            # Log error with metrics
            error_details = {
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id,
                "status_code": 500,
                "process_time_ms": round(process_time * 1000, 2),
                "error": str(e),
                "path": request.url.path,
                "method": request.method
            }

            logger.error(json.dumps({
                "event": "request_error",
                "level": "error",
                "details": error_details,
                "metrics": {
                    "response_time_ms": error_details["process_time_ms"],
                    "status_code": 500,
                    "endpoint": error_details["path"],
                    "method": request.method,
                    "error_type": type(e).__name__
                }
            }))

            # Create error response
            response_body = json.dumps({
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An internal server error occurred"
                }
            }).encode("utf-8")

            # Create ASGI response
            response_headers = [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(response_body)).encode()),
            ]

            async def send_error_response():
                await send({
                    "type": "http.response.start",
                    "status": 500,
                    "headers": response_headers,
                })
                await send({
                    "type": "http.response.body",
                    "body": response_body,
                })

            await send_error_response()