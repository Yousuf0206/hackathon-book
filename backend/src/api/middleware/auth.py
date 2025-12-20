"""
Authentication and rate limiting middleware.
"""
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, Optional
import time
import os
from collections import defaultdict, deque


class RateLimitMiddleware:
    """
    Middleware to implement rate limiting based on IP address.
    """
    def __init__(self, app):
        self.app = app
        # Get rate limit from environment or use default
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # 1 minute in seconds

        # Store request timestamps by IP
        self.requests_by_ip: Dict[str, deque] = defaultdict(deque)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope, receive)

        # Get client IP address
        client_ip = request.client.host if request.client else "unknown"

        # Get current time
        current_time = time.time()

        # Clean old requests outside the time window
        while (self.requests_by_ip[client_ip] and
               current_time - self.requests_by_ip[client_ip][0] > self.rate_limit_window):
            self.requests_by_ip[client_ip].popleft()

        # Check if rate limit exceeded
        if len(self.requests_by_ip[client_ip]) >= self.rate_limit_requests:
            response = JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Rate limit exceeded. Maximum {self.rate_limit_requests} requests per {self.rate_limit_window} seconds."
                    }
                }
            )
            await response(scope, receive, send)
            return

        # Add current request timestamp
        self.requests_by_ip[client_ip].append(current_time)

        # Continue with the request
        response = self.app(scope, receive, send)
        return await response


# Simple authentication middleware (placeholder for now)
class AuthMiddleware:
    """
    Placeholder for authentication middleware.
    In a real implementation, you might validate API keys, JWT tokens, etc.
    """
    async def __call__(self, request: Request, call_next):
        # For now, just continue with the request
        # In a real implementation, you would validate authentication here
        response = await call_next(request)
        return response