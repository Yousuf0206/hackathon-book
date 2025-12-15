"""
Security middleware for adding security headers.
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
import re
import html


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware to implement security headers.
    """
    async def dispatch(self, request: Request, call_next):
        # Process the request
        response = await call_next(request)

        # Add security headers to response
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"  # or "SAMEORIGIN" if needed
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:;"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"

        return response


def validate_user_input(input_text: str) -> str:
    """
    Function to validate and sanitize user input.
    """
    if not input_text:
        return input_text

    # Sanitize HTML to prevent XSS
    sanitized = html.escape(input_text)

    # Additional validation can be implemented here based on requirements
    # For now, we'll just return the sanitized input
    return sanitized