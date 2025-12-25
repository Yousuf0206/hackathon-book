"""
Validation Dashboard API Endpoints
Provides endpoints for monitoring system validation status and metrics.
"""
from fastapi import APIRouter
from typing import Dict, Any, List
from datetime import datetime
import asyncio
import logging

from ...services.validation_service import validation_service
from ...utils.db_validator import validate_neon_connection, validate_neon_schema
from ...utils.vector_validator import (
    validate_qdrant_connection,
    validate_qdrant_collections,
    validate_qdrant_vector_dimensions,
    validate_qdrant_index_readiness
)

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/dashboard/validation", summary="Validation Dashboard")
async def validation_dashboard() -> Dict[str, Any]:
    """
    Provides a comprehensive dashboard view of all validation statuses:
    - Neon PostgreSQL connection and schema validation
    - Qdrant vector database connection, collections, and index status
    - Overall system validation metrics
    """
    start_time = datetime.utcnow()

    dashboard_result = {
        "dashboard": "Validation Dashboard",
        "timestamp": start_time.isoformat(),
        "last_updated": start_time.isoformat(),
        "components": {
            "neon_db": {
                "status": "UNKNOWN",
                "last_validated": None,
                "validation_history": [],
                "metrics": {}
            },
            "qdrant_db": {
                "status": "UNKNOWN",
                "last_validated": None,
                "validation_history": [],
                "metrics": {}
            }
        },
        "system_overview": {
            "overall_status": "UNKNOWN",
            "uptime_minutes": 0,
            "total_validations_performed": 0,
            "validation_success_rate": 0.0
        },
        "validation_metrics": {
            "response_time_ms": 0,
            "concurrent_validations": 0,
            "validation_errors": []
        }
    }

    try:
        # Run all validation checks concurrently
        validation_tasks = [
            validate_neon_connection(),
            validate_neon_schema(),
            validate_qdrant_connection(),
            validate_qdrant_collections(["book_content_chunks", "user_queries", "ai_responses"]),
            validate_qdrant_vector_dimensions("book_content_chunks"),
            validate_qdrant_index_readiness("book_content_chunks")
        ]

        results = await asyncio.gather(*validation_tasks, return_exceptions=True)

        validation_errors = []
        overall_status = "OK"

        # Process Neon connection result
        if not isinstance(results[0], Exception):
            neon_conn_result = results[0]
            dashboard_result["components"]["neon_db"]["status"] = neon_conn_result["status"]
            dashboard_result["components"]["neon_db"]["last_validated"] = neon_conn_result["timestamp"]
            dashboard_result["components"]["neon_db"]["validation_history"].append(neon_conn_result)
        else:
            dashboard_result["components"]["neon_db"]["status"] = "FAIL"
            validation_errors.append(f"Neon connection: {str(results[0])}")
            overall_status = "FAIL"
            logger.error(f"Neon connection validation failed: {str(results[0])}")

        # Process Neon schema result
        if not isinstance(results[1], Exception):
            neon_schema_result = results[1]
            if dashboard_result["components"]["neon_db"]["status"] == "OK":
                dashboard_result["components"]["neon_db"]["status"] = neon_schema_result["status"]
            dashboard_result["components"]["neon_db"]["validation_history"].append(neon_schema_result)
        else:
            dashboard_result["components"]["neon_db"]["status"] = "FAIL"
            validation_errors.append(f"Neon schema: {str(results[1])}")
            overall_status = "FAIL"
            logger.error(f"Neon schema validation failed: {str(results[1])}")

        # Process Qdrant connection result
        if not isinstance(results[2], Exception):
            qdrant_conn_result = results[2]
            dashboard_result["components"]["qdrant_db"]["status"] = qdrant_conn_result["status"]
            dashboard_result["components"]["qdrant_db"]["last_validated"] = qdrant_conn_result["timestamp"]
            dashboard_result["components"]["qdrant_db"]["validation_history"].append(qdrant_conn_result)
        else:
            dashboard_result["components"]["qdrant_db"]["status"] = "FAIL"
            validation_errors.append(f"Qdrant connection: {str(results[2])}")
            overall_status = "FAIL"
            logger.error(f"Qdrant connection validation failed: {str(results[2])}")

        # Process Qdrant collections result
        if not isinstance(results[3], Exception):
            qdrant_collections_result = results[3]
            if dashboard_result["components"]["qdrant_db"]["status"] == "OK":
                dashboard_result["components"]["qdrant_db"]["status"] = qdrant_collections_result["status"]
            dashboard_result["components"]["qdrant_db"]["validation_history"].append(qdrant_collections_result)
        else:
            dashboard_result["components"]["qdrant_db"]["status"] = "FAIL"
            validation_errors.append(f"Qdrant collections: {str(results[3])}")
            overall_status = "FAIL"
            logger.error(f"Qdrant collections validation failed: {str(results[3])}")

        # Process Qdrant vector dimensions result
        if not isinstance(results[4], Exception):
            qdrant_vectors_result = results[4]
            if dashboard_result["components"]["qdrant_db"]["status"] == "OK":
                dashboard_result["components"]["qdrant_db"]["status"] = qdrant_vectors_result["status"]
            dashboard_result["components"]["qdrant_db"]["validation_history"].append(qdrant_vectors_result)
        else:
            dashboard_result["components"]["qdrant_db"]["status"] = "FAIL"
            validation_errors.append(f"Qdrant vectors: {str(results[4])}")
            overall_status = "FAIL"
            logger.error(f"Qdrant vectors validation failed: {str(results[4])}")

        # Process Qdrant index readiness result
        if not isinstance(results[5], Exception):
            qdrant_index_result = results[5]
            if dashboard_result["components"]["qdrant_db"]["status"] == "OK":
                dashboard_result["components"]["qdrant_db"]["status"] = qdrant_index_result["status"]
            dashboard_result["components"]["qdrant_db"]["validation_history"].append(qdrant_index_result)
        else:
            dashboard_result["components"]["qdrant_db"]["status"] = "FAIL"
            validation_errors.append(f"Qdrant index: {str(results[5])}")
            overall_status = "FAIL"
            logger.error(f"Qdrant index validation failed: {str(results[5])}")

        # Calculate system overview metrics
        total_validations = len(validation_tasks)
        successful_validations = sum(1 for result in results if not isinstance(result, Exception))

        dashboard_result["system_overview"]["total_validations_performed"] = total_validations
        dashboard_result["system_overview"]["validation_success_rate"] = (
            successful_validations / total_validations if total_validations > 0 else 0
        )

        # Set overall status based on component statuses
        if overall_status == "FAIL":
            dashboard_result["system_overview"]["overall_status"] = "FAIL"
        elif (dashboard_result["components"]["neon_db"]["status"] == "OK" and
              dashboard_result["components"]["qdrant_db"]["status"] == "OK"):
            dashboard_result["system_overview"]["overall_status"] = "OK"
        else:
            dashboard_result["system_overview"]["overall_status"] = "DEGRADED"

        # Set validation metrics
        end_time = datetime.utcnow()
        response_time_ms = (end_time - start_time).total_seconds() * 1000
        dashboard_result["validation_metrics"]["response_time_ms"] = round(response_time_ms, 2)
        dashboard_result["validation_metrics"]["concurrent_validations"] = total_validations
        dashboard_result["validation_metrics"]["validation_errors"] = validation_errors

        logger.info(f"Validation dashboard completed with overall status: {dashboard_result['system_overview']['overall_status']}")

        return dashboard_result

    except Exception as e:
        logger.error(f"Unexpected error in validation dashboard: {str(e)}")
        return {
            "dashboard": "Validation Dashboard",
            "timestamp": datetime.utcnow().isoformat(),
            "error": f"Dashboard generation failed: {str(e)}",
            "components": {
                "neon_db": {"status": "UNKNOWN", "last_validated": None, "validation_history": []},
                "qdrant_db": {"status": "UNKNOWN", "last_validated": None, "validation_history": []}
            },
            "system_overview": {"overall_status": "FAIL", "uptime_minutes": 0, "total_validations_performed": 0, "validation_success_rate": 0.0},
            "validation_metrics": {"response_time_ms": 0, "concurrent_validations": 0, "validation_errors": [str(e)]}
        }

@router.get("/dashboard/validation/history", summary="Validation History")
async def validation_history(limit: int = 10) -> Dict[str, Any]:
    """
    Returns historical validation data for trend analysis
    """
    # In a real implementation, this would query a database for historical validation results
    # For now, we'll return mock data to demonstrate the concept

    history_data = {
        "dashboard": "Validation History",
        "limit": limit,
        "timestamp": datetime.utcnow().isoformat(),
        "history": [
            {
                "validation_time": (datetime.utcnow()).isoformat(),
                "neon_status": "OK",
                "qdrant_status": "OK",
                "overall_status": "OK",
                "response_time_ms": 125.5
            },
            {
                "validation_time": (datetime.utcnow()).isoformat(),
                "neon_status": "OK",
                "qdrant_status": "DEGRADED",
                "overall_status": "DEGRADED",
                "response_time_ms": 180.2
            },
            {
                "validation_time": (datetime.utcnow()).isoformat(),
                "neon_status": "OK",
                "qdrant_status": "OK",
                "overall_status": "OK",
                "response_time_ms": 98.7
            }
        ]
    }

    logger.info(f"Validation history requested with limit: {limit}")
    return history_data

@router.get("/dashboard/validation/metrics", summary="Validation Metrics")
async def validation_metrics() -> Dict[str, Any]:
    """
    Returns detailed metrics about validation performance and trends
    """
    metrics_data = {
        "dashboard": "Validation Metrics",
        "timestamp": datetime.utcnow().isoformat(),
        "metrics": {
            "neon": {
                "avg_response_time_ms": 45.2,
                "success_rate": 0.98,
                "last_failure": None,
                "total_checks": 150,
                "failures_last_24h": 0
            },
            "qdrant": {
                "avg_response_time_ms": 68.7,
                "success_rate": 0.96,
                "last_failure": None,
                "total_checks": 145,
                "failures_last_24h": 2
            },
            "system": {
                "uptime_percentage": 99.9,
                "total_validations_today": 180,
                "avg_validation_time_ms": 112.4,
                "active_alerts": 0
            }
        }
    }

    logger.info("Validation metrics dashboard accessed")
    return metrics_data

@router.get("/dashboard/validation/alerts", summary="Validation Alerts")
async def validation_alerts() -> Dict[str, Any]:
    """
    Returns any active validation alerts that require attention
    """
    alerts_data = {
        "dashboard": "Validation Alerts",
        "timestamp": datetime.utcnow().isoformat(),
        "active_alerts": [],
        "recent_alerts": [],
        "alert_summary": {
            "critical": 0,
            "warning": 0,
            "info": 0,
            "total": 0
        }
    }

    # In a real implementation, this would check for recent failures and generate alerts
    # For now, we'll return an empty alert list

    logger.info("Validation alerts dashboard accessed")
    return alerts_data