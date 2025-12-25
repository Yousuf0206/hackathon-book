"""
Infrastructure Validation Service
This module provides centralized validation services for Neon and Qdrant infrastructure.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..utils.db_validator import validate_neon_connection, validate_neon_schema
from ..utils.vector_validator import (
    validate_qdrant_connection,
    validate_qdrant_collections,
    validate_qdrant_vector_dimensions,
    validate_qdrant_index_readiness
)

logger = logging.getLogger(__name__)

class ValidationService:
    """
    Service class for validating infrastructure components including:
    - Neon PostgreSQL connection and schema
    - Qdrant vector database connection, collections, and indexes
    - System-wide validation checks
    """

    def __init__(self):
        self.validation_results = {}

    async def validate_infrastructure(self,
                                   validate_neon: bool = True,
                                   validate_qdrant: bool = True,
                                   collections_to_check: List[str] = None,
                                   expected_dimension: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform comprehensive infrastructure validation

        Args:
            validate_neon: Whether to validate Neon connection and schema
            validate_qdrant: Whether to validate Qdrant connection and collections
            collections_to_check: List of Qdrant collections to validate
            expected_dimension: Expected vector dimension for validation

        Returns:
            Dictionary with validation results
        """
        if collections_to_check is None:
            collections_to_check = ["book_content_chunks", "user_queries", "ai_responses"]

        start_time = datetime.utcnow()

        validation_result = {
            "status": "UNKNOWN",
            "timestamp": start_time.isoformat(),
            "components": {
                "neon": {"status": "UNKNOWN", "message": "Not validated"},
                "qdrant": {"status": "UNKNOWN", "message": "Not validated"}
            },
            "details": {},
            "validation_summary": {
                "neon_validated": validate_neon,
                "qdrant_validated": validate_qdrant,
                "collections_checked": collections_to_check
            }
        }

        try:
            validation_tasks = []

            # Add Neon validation tasks if requested
            if validate_neon:
                validation_tasks.extend([
                    validate_neon_connection(),
                    validate_neon_schema()
                ])

            # Add Qdrant validation tasks if requested
            if validate_qdrant:
                validation_tasks.extend([
                    validate_qdrant_connection(),
                    validate_qdrant_collections(collections_to_check),
                    validate_qdrant_vector_dimensions("book_content_chunks", expected_dimension)
                ])

            # Execute all validation tasks concurrently
            results = await asyncio.gather(*validation_tasks, return_exceptions=True)

            # Process results
            result_index = 0

            if validate_neon:
                # Process Neon connection result
                if not isinstance(results[result_index], Exception):
                    neon_conn_result = results[result_index]
                    validation_result["components"]["neon"]["status"] = neon_conn_result["status"]
                    validation_result["components"]["neon"]["message"] = neon_conn_result.get("error_message", "Neon connection OK")
                    validation_result["details"]["neon_connection"] = neon_conn_result
                else:
                    validation_result["components"]["neon"]["status"] = "FAIL"
                    validation_result["components"]["neon"]["message"] = f"Neon connection validation error: {str(results[result_index])}"
                    logger.error(f"Neon connection validation failed: {str(results[result_index])}")
                result_index += 1

                # Process Neon schema result
                if not isinstance(results[result_index], Exception):
                    neon_schema_result = results[result_index]
                    if validation_result["components"]["neon"]["status"] != "FAIL":  # Only update if connection was OK
                        validation_result["components"]["neon"]["status"] = neon_schema_result["status"]
                        validation_result["components"]["neon"]["message"] = neon_schema_result.get("error_message", "Neon schema OK")
                    validation_result["details"]["neon_schema"] = neon_schema_result
                else:
                    validation_result["components"]["neon"]["status"] = "FAIL"
                    validation_result["components"]["neon"]["message"] = f"Neon schema validation error: {str(results[result_index])}"
                    logger.error(f"Neon schema validation failed: {str(results[result_index])}")
                result_index += 1

            if validate_qdrant:
                # Process Qdrant connection result
                if not isinstance(results[result_index], Exception):
                    qdrant_conn_result = results[result_index]
                    validation_result["components"]["qdrant"]["status"] = qdrant_conn_result["status"]
                    validation_result["components"]["qdrant"]["message"] = qdrant_conn_result.get("error_message", "Qdrant connection OK")
                    validation_result["details"]["qdrant_connection"] = qdrant_conn_result
                else:
                    validation_result["components"]["qdrant"]["status"] = "FAIL"
                    validation_result["components"]["qdrant"]["message"] = f"Qdrant connection validation error: {str(results[result_index])}"
                    logger.error(f"Qdrant connection validation failed: {str(results[result_index])}")
                result_index += 1

                # Process Qdrant collections result
                if not isinstance(results[result_index], Exception):
                    qdrant_collections_result = results[result_index]
                    if validation_result["components"]["qdrant"]["status"] != "FAIL":  # Only update if connection was OK
                        validation_result["components"]["qdrant"]["status"] = qdrant_collections_result["status"]
                        validation_result["components"]["qdrant"]["message"] = qdrant_collections_result.get("error_message", "Qdrant collections OK")
                    validation_result["details"]["qdrant_collections"] = qdrant_collections_result
                else:
                    validation_result["components"]["qdrant"]["status"] = "FAIL"
                    validation_result["components"]["qdrant"]["message"] = f"Qdrant collections validation error: {str(results[result_index])}"
                    logger.error(f"Qdrant collections validation failed: {str(results[result_index])}")
                result_index += 1

                # Process Qdrant vector dimensions result
                if not isinstance(results[result_index], Exception):
                    qdrant_vectors_result = results[result_index]
                    if validation_result["components"]["qdrant"]["status"] != "FAIL":  # Only update if connection was OK
                        validation_result["components"]["qdrant"]["status"] = qdrant_vectors_result["status"]
                        validation_result["components"]["qdrant"]["message"] = qdrant_vectors_result.get("error_message", "Qdrant vectors OK")
                    validation_result["details"]["qdrant_vectors"] = qdrant_vectors_result
                else:
                    validation_result["components"]["qdrant"]["status"] = "FAIL"
                    validation_result["components"]["qdrant"]["message"] = f"Qdrant vectors validation error: {str(results[result_index])}"
                    logger.error(f"Qdrant vectors validation failed: {str(results[result_index])}")

            # Determine overall status
            statuses = []
            if validate_neon:
                statuses.append(validation_result["components"]["neon"]["status"])
            if validate_qdrant:
                statuses.append(validation_result["components"]["qdrant"]["status"])

            if all(status == "OK" for status in statuses):
                validation_result["status"] = "OK"
            elif any(status == "FAIL" for status in statuses):
                validation_result["status"] = "FAIL"
            else:
                validation_result["status"] = "DEGRADED"

            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            validation_result["response_time_ms"] = round(response_time_ms, 2)

            logger.info(f"Infrastructure validation completed with status: {validation_result['status']}")

            return validation_result

        except Exception as e:
            logger.error(f"Unexpected error in infrastructure validation: {str(e)}")
            return {
                "status": "FAIL",
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Infrastructure validation failed: {str(e)}",
                "components": {
                    "neon": {"status": "UNKNOWN", "message": "Not validated due to error"},
                    "qdrant": {"status": "UNKNOWN", "message": "Not validated due to error"}
                },
                "validation_summary": {
                    "neon_validated": validate_neon,
                    "qdrant_validated": validate_qdrant,
                    "collections_checked": collections_to_check
                }
            }

    async def validate_qdrant_collection_existence(self, collection_names: List[str]) -> Dict[str, Any]:
        """
        Validate that specific Qdrant collections exist

        Args:
            collection_names: List of collection names to validate

        Returns:
            Dictionary with collection validation results
        """
        start_time = datetime.utcnow()

        validation_result = {
            "status": "UNKNOWN",
            "timestamp": start_time.isoformat(),
            "collections_validated": [],
            "summary": {
                "total": len(collection_names),
                "existing": 0,
                "missing": 0
            }
        }

        try:
            # Validate each collection individually
            collection_results = await asyncio.gather(
                *[validate_qdrant_collections([name]) for name in collection_names],
                return_exceptions=True
            )

            existing_count = 0
            missing_count = 0

            for i, collection_name in enumerate(collection_names):
                if not isinstance(collection_results[i], Exception):
                    collection_result = collection_results[i]
                    # Extract the specific collection result
                    collection_detail = collection_result.get("collections_validated", [{}])[0] if collection_result.get("collections_validated") else {}

                    collection_validation = {
                        "name": collection_name,
                        "status": collection_detail.get("status", "UNKNOWN"),
                        "exists": collection_detail.get("status") == "OK",
                        "details": collection_detail
                    }

                    validation_result["collections_validated"].append(collection_validation)

                    if collection_detail.get("status") == "OK":
                        existing_count += 1
                    else:
                        missing_count += 1
                else:
                    collection_validation = {
                        "name": collection_name,
                        "status": "FAIL",
                        "exists": False,
                        "error": str(collection_results[i]),
                        "details": {}
                    }
                    validation_result["collections_validated"].append(collection_validation)
                    missing_count += 1

            validation_result["summary"]["existing"] = existing_count
            validation_result["summary"]["missing"] = missing_count

            # Determine overall status
            if missing_count == 0:
                validation_result["status"] = "OK"
            elif existing_count == 0:
                validation_result["status"] = "FAIL"
            else:
                validation_result["status"] = "DEGRADED"

            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            validation_result["response_time_ms"] = round(response_time_ms, 2)

            logger.info(f"Qdrant collection existence validation completed: {existing_count}/{len(collection_names)} collections exist")

            return validation_result

        except Exception as e:
            logger.error(f"Unexpected error in Qdrant collection existence validation: {str(e)}")
            return {
                "status": "FAIL",
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Qdrant collection existence validation failed: {str(e)}",
                "collections_validated": [],
                "summary": {
                    "total": len(collection_names),
                    "existing": 0,
                    "missing": 0
                }
            }

    async def validate_vector_dimension_compatibility(self, collection_name: str, expected_dimension: int) -> Dict[str, Any]:
        """
        Validate that the vector dimensions in a Qdrant collection match expected values

        Args:
            collection_name: Name of the collection to validate
            expected_dimension: Expected vector dimension

        Returns:
            Dictionary with dimension validation results
        """
        start_time = datetime.utcnow()

        validation_result = {
            "status": "UNKNOWN",
            "timestamp": start_time.isoformat(),
            "collection": collection_name,
            "expected_dimension": expected_dimension,
            "actual_dimension": None,
            "compatible": False
        }

        try:
            dimension_result = await validate_qdrant_vector_dimensions(collection_name, expected_dimension)

            validation_result["status"] = dimension_result["status"]
            validation_result["actual_dimension"] = dimension_result.get("actual_dimension")
            validation_result["compatible"] = dimension_result["status"] == "OK"
            validation_result["details"] = dimension_result

            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            validation_result["response_time_ms"] = round(response_time_ms, 2)

            logger.info(f"Vector dimension validation for '{collection_name}': expected={expected_dimension}, actual={validation_result['actual_dimension']}, compatible={validation_result['compatible']}")

            return validation_result

        except Exception as e:
            logger.error(f"Unexpected error in vector dimension validation: {str(e)}")
            return {
                "status": "FAIL",
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Vector dimension validation failed: {str(e)}",
                "collection": collection_name,
                "expected_dimension": expected_dimension,
                "actual_dimension": None,
                "compatible": False
            }

    async def validate_index_readiness(self, collection_name: str) -> Dict[str, Any]:
        """
        Validate that a Qdrant collection index is ready for search operations

        Args:
            collection_name: Name of the collection to validate

        Returns:
            Dictionary with index readiness validation results
        """
        start_time = datetime.utcnow()

        validation_result = {
            "status": "UNKNOWN",
            "timestamp": start_time.isoformat(),
            "collection": collection_name,
            "points_count": 0,
            "index_status": "unknown",
            "ready_for_search": False
        }

        try:
            from ..utils.vector_validator import validate_qdrant_index_readiness
            readiness_result = await validate_qdrant_index_readiness(collection_name)

            validation_result["status"] = readiness_result["status"]
            validation_result["points_count"] = readiness_result.get("points_count", 0)
            validation_result["index_status"] = readiness_result.get("index_status", "unknown")
            validation_result["ready_for_search"] = validation_result["status"] == "OK"
            validation_result["details"] = readiness_result

            # Calculate response time
            end_time = datetime.utcnow()
            response_time_ms = (end_time - start_time).total_seconds() * 1000
            validation_result["response_time_ms"] = round(response_time_ms, 2)

            logger.info(f"Index readiness validation for '{collection_name}': status={validation_result['status']}, points={validation_result['points_count']}")

            return validation_result

        except Exception as e:
            logger.error(f"Unexpected error in index readiness validation: {str(e)}")
            return {
                "status": "FAIL",
                "timestamp": datetime.utcnow().isoformat(),
                "error": f"Index readiness validation failed: {str(e)}",
                "collection": collection_name,
                "points_count": 0,
                "index_status": "unknown",
                "ready_for_search": False
            }

# Create a singleton instance
validation_service = ValidationService()