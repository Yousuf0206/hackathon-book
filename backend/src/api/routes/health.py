"""
Health check API route.
"""
from fastapi import APIRouter
from typing import Dict, Any
import os
from datetime import datetime
import openai
from ...services.embedding_service import EmbeddingService
from ...services.retrieval_service import RetrievalService
from ...services.chat_service import ChatService


router = APIRouter()


@router.get("/health", response_model=Dict[str, Any])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint that verifies the status of the API and its dependencies.
    """
    # Check environment variables
    required_env_vars = ["COHERE_API_KEY", "QDRANT_URL", "OPENAI_API_KEY"]
    missing_env_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_env_vars.append(var)

    # Test services
    dependencies_status = {
        "qdrant": "unknown",
        "cohere": "unknown",
        "openai": "unknown"
    }

    # Test Cohere connection
    try:
        embedding_service = EmbeddingService()
        # Perform a simple test embedding
        test_embedding = embedding_service.generate_embeddings(["test"])
        dependencies_status["cohere"] = "available" if test_embedding else "unavailable"
    except Exception:
        dependencies_status["cohere"] = "unavailable"

    # Test Qdrant connection
    try:
        retrieval_service = RetrievalService()
        # Perform a simple test - try to get collection info
        retrieval_service.client.get_collection(retrieval_service.collection_name)
        dependencies_status["qdrant"] = "connected"
    except Exception:
        dependencies_status["qdrant"] = "unavailable"

    # Test OpenAI connection
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            # Perform a simple test - check if API key is valid
            # We'll use a simple models list call as a test
            client = openai.OpenAI(api_key=api_key)
            client.models.list()
            dependencies_status["openai"] = "available"
        else:
            dependencies_status["openai"] = "unavailable"
    except Exception:
        dependencies_status["openai"] = "unavailable"

    # Determine overall status
    all_healthy = all(status == "available" or status == "connected"
                      for status in dependencies_status.values()) and not missing_env_vars

    return {
        "status": "healthy" if all_healthy else "unhealthy",
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": dependencies_status,
        "missing_env_vars": missing_env_vars,
        "details": {
            "qdrant_status": dependencies_status["qdrant"],
            "cohere_status": dependencies_status["cohere"],
            "openai_status": dependencies_status["openai"],
            "env_vars_check": "passed" if not missing_env_vars else "failed"
        }
    }