"""
Main FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import health, query, chat, embed
from ..api.middleware.auth import RateLimitMiddleware
from ..api.middleware.logging import LoggingMiddleware
from ..api.middleware.security import SecurityMiddleware


# Create FastAPI app instance
app = FastAPI(
    title="RAG Chatbot API",
    description="API for RAG-based question answering system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(SecurityMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(LoggingMiddleware)

# Include API routes
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(embed.router, prefix="/api", tags=["embed"])

@app.get("/")
async def root():
    """
    Root endpoint for basic health check.
    """
    return {"message": "RAG Chatbot API is running!"}

@app.get("/api")
async def api_root():
    """
    API root endpoint.
    """
    return {"message": "Welcome to the RAG Chatbot API", "version": "1.0.0"}