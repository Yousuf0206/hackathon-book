"""
Health Check API Endpoints
Provides endpoints for system health validation including Neon and Qdrant connections.
"""
from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
import asyncio
import logging
import os

from ...utils.db_validator import validate_neon_connection, validate_neon_schema
from ...utils.vector_validator import (
    validate_qdrant_connection,
    validate_qdrant_collections,
    validate_qdrant_vector_dimensions
)
from ...services.embedding_service import EmbeddingService
from ...services.retrieval_service import RetrievalService
from ...services.chat_service import ChatService

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health", summary="System Health Check")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint that validates all system components:
    - Neon PostgreSQL connection
    - Qdrant vector database connection
    - Basic API functionality
    """
    start_time = datetime.utcnow()

    # Initialize results
    health_result = {
        "status": "OK",
        "timestamp": start_time.isoformat(),
        "uptime": None,
        "components": {
            "api": {"status": "OK", "message": "API is responding"},
            "neon_db": {"status": "UNKNOWN", "message": "Checking..."},
            "qdrant_db": {"status": "UNKNOWN", "message": "Checking..."},
            "overall": "UNKNOWN"
        },
        "details": {}
    }

    try:
        # Validate Neon connection
        try:
            neon_result = await validate_neon_connection()
            health_result["components"]["neon_db"]["status"] = neon_result["status"]
            health_result["components"]["neon_db"]["message"] = neon_result.get("error_message", "Neon connection OK")
            health_result["details"]["neon"] = neon_result
        except Exception as e:
            health_result["components"]["neon_db"]["status"] = "FAIL"
            health_result["components"]["neon_db"]["message"] = f"Neon validation error: {str(e)}"
            health_result["status"] = "FAIL"
            logger.error(f"Neon validation failed: {str(e)}")

        # Validate Qdrant connection
        try:
            qdrant_result = await validate_qdrant_connection()
            health_result["components"]["qdrant_db"]["status"] = qdrant_result["status"]
            health_result["components"]["qdrant_db"]["message"] = qdrant_result.get("error_message", "Qdrant connection OK")
            health_result["details"]["qdrant"] = qdrant_result
        except Exception as e:
            health_result["components"]["qdrant_db"]["status"] = "FAIL"
            health_result["components"]["qdrant_db"]["message"] = f"Qdrant validation error: {str(e)}"
            health_result["status"] = "FAIL"
            logger.error(f"Qdrant validation failed: {str(e)}")

        # Determine overall status
        if (health_result["components"]["neon_db"]["status"] == "OK" and
            health_result["components"]["qdrant_db"]["status"] == "OK"):
            health_result["components"]["overall"] = "OK"
        elif (health_result["components"]["neon_db"]["status"] == "FAIL" or
              health_result["components"]["qdrant_db"]["status"] == "FAIL"):
            health_result["components"]["overall"] = "FAIL"
            health_result["status"] = "FAIL"
        else:
            health_result["components"]["overall"] = "DEGRADED"
            health_result["status"] = "DEGRADED"

        # Calculate response time
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        health_result["response_time_ms"] = round(response_time_ms, 2)

        logger.info(f"Health check completed with status: {health_result['status']}")

        return health_result

    except Exception as e:
        logger.error(f"Unexpected error in health check: {str(e)}")
        return {
            "status": "FAIL",
            "timestamp": datetime.utcnow().isoformat(),
            "error": f"Health check failed: {str(e)}",
            "components": {
                "api": {"status": "OK", "message": "API is responding"},
                "neon_db": {"status": "UNKNOWN", "message": "Not checked due to error"},
                "qdrant_db": {"status": "UNKNOWN", "message": "Not checked due to error"},
                "overall": "FAIL"
            }
        }


@router.get("/health/full", summary="Full System Health Check")
async def full_health_check() -> Dict[str, Any]:
    """
    Full health check that includes detailed validation of all system components:
    - Neon connection and schema
    - Qdrant connection, collections, and vector dimensions
    - Additional system checks
    """
    start_time = datetime.utcnow()

    full_health_result = {
        "status": "OK",
        "timestamp": start_time.isoformat(),
        "components": {
            "api": {"status": "OK", "message": "API is responding"},
            "neon_db": {"status": "UNKNOWN", "message": "Checking..."},
            "neon_schema": {"status": "UNKNOWN", "message": "Checking..."},
            "qdrant_db": {"status": "UNKNOWN", "message": "Checking..."},
            "qdrant_collections": {"status": "UNKNOWN", "message": "Checking..."},
            "qdrant_vectors": {"status": "UNKNOWN", "message": "Checking..."},
            "overall": "UNKNOWN"
        },
        "details": {}
    }

    try:
        # Run all validations concurrently for efficiency
        validation_tasks = [
            validate_neon_connection(),
            validate_neon_schema(),
            validate_qdrant_connection(),
            validate_qdrant_collections(["book_content_chunks", "user_queries", "ai_responses"]),
            validate_qdrant_vector_dimensions("book_content_chunks")  # Default expected collection
        ]

        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        # Process Neon connection result
        if not isinstance(results[0], Exception):
            neon_conn_result = results[0]
            full_health_result["components"]["neon_db"]["status"] = neon_conn_result["status"]
            full_health_result["components"]["neon_db"]["message"] = neon_conn_result.get("error_message", "Neon connection OK")
            full_health_result["details"]["neon_connection"] = neon_conn_result
        else:
            full_health_result["components"]["neon_db"]["status"] = "FAIL"
            full_health_result["components"]["neon_db"]["message"] = f"Neon connection validation error: {str(results[0])}"
            full_health_result["status"] = "FAIL"
            logger.error(f"Neon connection validation failed: {str(results[0])}")

        # Process Neon schema result
        if not isinstance(results[1], Exception):
            neon_schema_result = results[1]
            full_health_result["components"]["neon_schema"]["status"] = neon_schema_result["status"]
            full_health_result["components"]["neon_schema"]["message"] = neon_schema_result.get("error_message", "Neon schema OK")
            full_health_result["details"]["neon_schema"] = neon_schema_result
        else:
            full_health_result["components"]["neon_schema"]["status"] = "FAIL"
            full_health_result["components"]["neon_schema"]["message"] = f"Neon schema validation error: {str(results[1])}"
            full_health_result["status"] = "FAIL"
            logger.error(f"Neon schema validation failed: {str(results[1])}")

        # Process Qdrant connection result
        if not isinstance(results[2], Exception):
            qdrant_conn_result = results[2]
            full_health_result["components"]["qdrant_db"]["status"] = qdrant_conn_result["status"]
            full_health_result["components"]["qdrant_db"]["message"] = qdrant_conn_result.get("error_message", "Qdrant connection OK")
            full_health_result["details"]["qdrant_connection"] = qdrant_conn_result
        else:
            full_health_result["components"]["qdrant_db"]["status"] = "FAIL"
            full_health_result["components"]["qdrant_db"]["message"] = f"Qdrant connection validation error: {str(results[2])}"
            full_health_result["status"] = "FAIL"
            logger.error(f"Qdrant connection validation failed: {str(results[2])}")

        # Process Qdrant collections result
        if not isinstance(results[3], Exception):
            qdrant_collections_result = results[3]
            full_health_result["components"]["qdrant_collections"]["status"] = qdrant_collections_result["status"]
            full_health_result["components"]["qdrant_collections"]["message"] = qdrant_collections_result.get("error_message", "Qdrant collections OK")
            full_health_result["details"]["qdrant_collections"] = qdrant_collections_result
        else:
            full_health_result["components"]["qdrant_collections"]["status"] = "FAIL"
            full_health_result["components"]["qdrant_collections"]["message"] = f"Qdrant collections validation error: {str(results[3])}"
            full_health_result["status"] = "FAIL"
            logger.error(f"Qdrant collections validation failed: {str(results[3])}")

        # Process Qdrant vector dimensions result
        if not isinstance(results[4], Exception):
            qdrant_vectors_result = results[4]
            full_health_result["components"]["qdrant_vectors"]["status"] = qdrant_vectors_result["status"]
            full_health_result["components"]["qdrant_vectors"]["message"] = qdrant_vectors_result.get("error_message", "Qdrant vectors OK")
            full_health_result["details"]["qdrant_vectors"] = qdrant_vectors_result
        else:
            full_health_result["components"]["qdrant_vectors"]["status"] = "FAIL"
            full_health_result["components"]["qdrant_vectors"]["message"] = f"Qdrant vectors validation error: {str(results[4])}"
            full_health_result["status"] = "FAIL"
            logger.error(f"Qdrant vectors validation failed: {str(results[4])}")

        # Determine overall status
        statuses = [
            full_health_result["components"]["neon_db"]["status"],
            full_health_result["components"]["neon_schema"]["status"],
            full_health_result["components"]["qdrant_db"]["status"],
            full_health_result["components"]["qdrant_collections"]["status"],
            full_health_result["components"]["qdrant_vectors"]["status"]
        ]

        if all(status == "OK" for status in statuses):
            full_health_result["components"]["overall"] = "OK"
        elif any(status == "FAIL" for status in statuses):
            full_health_result["components"]["overall"] = "FAIL"
            full_health_result["status"] = "FAIL"
        else:
            full_health_result["components"]["overall"] = "DEGRADED"
            full_health_result["status"] = "DEGRADED"

        # Calculate response time
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        full_health_result["response_time_ms"] = round(response_time_ms, 2)

        logger.info(f"Full health check completed with status: {full_health_result['status']}")

        return full_health_result

    except Exception as e:
        logger.error(f"Unexpected error in full health check: {str(e)}")
        return {
            "status": "FAIL",
            "timestamp": datetime.utcnow().isoformat(),
            "error": f"Full health check failed: {str(e)}",
            "components": {
                "api": {"status": "OK", "message": "API is responding"},
                "neon_db": {"status": "UNKNOWN", "message": "Not checked due to error"},
                "neon_schema": {"status": "UNKNOWN", "message": "Not checked due to error"},
                "qdrant_db": {"status": "UNKNOWN", "message": "Not checked due to error"},
                "qdrant_collections": {"status": "UNKNOWN", "message": "Not checked due to error"},
                "qdrant_vectors": {"status": "UNKNOWN", "message": "Not checked due to error"},
                "overall": "FAIL"
            }
        }


@router.get("/health/neon", summary="Neon-specific Health Check")
async def neon_health_check() -> Dict[str, Any]:
    """
    Health check specifically for Neon PostgreSQL connection and schema
    """
    start_time = datetime.utcnow()

    neon_health_result = {
        "service": "Neon PostgreSQL",
        "status": "UNKNOWN",
        "timestamp": start_time.isoformat(),
        "details": {}
    }

    try:
        # Validate Neon connection
        conn_result = await validate_neon_connection()
        neon_health_result["status"] = conn_result["status"]
        neon_health_result["details"]["connection"] = conn_result

        # Validate Neon schema
        schema_result = await validate_neon_schema()
        neon_health_result["details"]["schema"] = schema_result

        # If any validation failed, mark overall as fail
        if conn_result["status"] != "OK" or schema_result["status"] not in ["OK", "WARNING"]:
            neon_health_result["status"] = "FAIL"

        # Calculate response time
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        neon_health_result["response_time_ms"] = round(response_time_ms, 2)

        logger.info(f"Neon health check completed with status: {neon_health_result['status']}")

        return neon_health_result

    except Exception as e:
        logger.error(f"Neon health check failed: {str(e)}")
        return {
            "service": "Neon PostgreSQL",
            "status": "FAIL",
            "timestamp": datetime.utcnow().isoformat(),
            "error": f"Neon health check failed: {str(e)}",
            "details": {}
        }


@router.get("/health/qdrant", summary="Qdrant-specific Health Check")
async def qdrant_health_check() -> Dict[str, Any]:
    """
    Health check specifically for Qdrant vector database
    """
    start_time = datetime.utcnow()

    qdrant_health_result = {
        "service": "Qdrant Vector Database",
        "status": "UNKNOWN",
        "timestamp": start_time.isoformat(),
        "details": {}
    }

    try:
        # Validate Qdrant connection
        conn_result = await validate_qdrant_connection()
        qdrant_health_result["status"] = conn_result["status"]
        qdrant_health_result["details"]["connection"] = conn_result

        # Validate Qdrant collections
        collections_result = await validate_qdrant_collections(["book_content_chunks", "user_queries", "ai_responses"])
        qdrant_health_result["details"]["collections"] = collections_result

        # Validate Qdrant vector dimensions
        vectors_result = await validate_qdrant_vector_dimensions("book_content_chunks")
        qdrant_health_result["details"]["vectors"] = vectors_result

        # If any validation failed, mark overall as fail
        if (conn_result["status"] != "OK" or
            collections_result["status"] not in ["OK", "WARNING"] or
            vectors_result["status"] not in ["OK", "WARNING"]):
            qdrant_health_result["status"] = "FAIL"

        # Calculate response time
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        qdrant_health_result["response_time_ms"] = round(response_time_ms, 2)

        logger.info(f"Qdrant health check completed with status: {qdrant_health_result['status']}")

        return qdrant_health_result

    except Exception as e:
        logger.error(f"Qdrant health check failed: {str(e)}")
        return {
            "service": "Qdrant Vector Database",
            "status": "FAIL",
            "timestamp": datetime.utcnow().isoformat(),
            "error": f"Qdrant health check failed: {str(e)}",
            "details": {}
        }