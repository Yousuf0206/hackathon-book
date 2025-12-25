"""
Qdrant Vector Database Validator
This module provides utilities to validate Qdrant vector database connections and collections.
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponseException
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

class QdrantValidator:
    """Utility class for validating Qdrant vector database connections and collections"""

    def __init__(self):
        # Get Qdrant configuration from environment variables
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_port = os.getenv("QDRANT_PORT", "6333")

        # Determine if we're using local or cloud instance
        if self.qdrant_url:
            # Cloud instance
            self.client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=10
            )
        else:
            # Local instance
            self.client = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(self.qdrant_port),
                timeout=10
            )

    async def validate_connection(self) -> Dict[str, Any]:
        """
        Validate the Qdrant connection
        Returns a dictionary with validation results
        """
        result = {
            "service": "Qdrant Vector Database",
            "status": "FAIL",
            "timestamp": None,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "error_message": None
        }

        try:
            # Try to get the list of collections to verify connection
            collections = self.client.get_collections()

            # If we get here, the connection is successful
            result["status"] = "OK"
            result["timestamp"] = asyncio.get_event_loop().time()

            logger.info("Qdrant connection validated successfully")

        except UnexpectedResponseException as e:
            result["error_message"] = f"Qdrant API error: {str(e)}"
            logger.error(f"Qdrant connection validation failed: {str(e)}")

        except Exception as e:
            result["error_message"] = f"Connection error: {str(e)}"
            logger.error(f"Qdrant connection validation failed: {str(e)}")

        return result

    async def validate_collection_exists(self, collection_name: str) -> Dict[str, Any]:
        """
        Validate that a specific Qdrant collection exists
        Returns a dictionary with validation results
        """
        result = {
            "service": f"Qdrant Collection: {collection_name}",
            "status": "FAIL",
            "timestamp": None,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "error_message": None,
            "collection_name": collection_name
        }

        try:
            # Try to get collection info to verify it exists
            collection_info = self.client.get_collection(collection_name)

            # If we get here, the collection exists
            result["status"] = "OK"
            result["timestamp"] = asyncio.get_event_loop().time()
            result["collection_info"] = {
                "vector_size": collection_info.config.params.vectors.size,
                "distance": collection_info.config.params.vectors.distance.value,
                "count": collection_info.points_count
            }

            logger.info(f"Qdrant collection '{collection_name}' validated successfully")

        except UnexpectedResponseException as e:
            if e.status_code == 404:
                result["error_message"] = f"Collection '{collection_name}' does not exist"
            else:
                result["error_message"] = f"Qdrant API error: {str(e)}"
            logger.error(f"Qdrant collection validation failed: {str(e)}")

        except Exception as e:
            result["error_message"] = f"Collection validation error: {str(e)}"
            logger.error(f"Qdrant collection validation failed: {str(e)}")

        return result

    async def validate_collections_exist(self, collection_names: List[str]) -> Dict[str, Any]:
        """
        Validate that multiple Qdrant collections exist
        Returns a dictionary with validation results
        """
        result = {
            "service": "Qdrant Collections Validation",
            "status": "FAIL",
            "timestamp": None,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "error_message": None,
            "collections_validated": []
        }

        try:
            all_valid = True
            error_messages = []

            for collection_name in collection_names:
                collection_result = await self.validate_collection_exists(collection_name)
                result["collections_validated"].append(collection_result)

                if collection_result["status"] != "OK":
                    all_valid = False
                    error_messages.append(f"{collection_name}: {collection_result['error_message']}")

            if all_valid:
                result["status"] = "OK"
            else:
                result["error_message"] = "; ".join(error_messages)

            result["timestamp"] = asyncio.get_event_loop().time()

            logger.info(f"Qdrant collections validation completed: {len([c for c in result['collections_validated'] if c['status'] == 'OK'])}/{len(collection_names)} collections exist")

        except Exception as e:
            result["error_message"] = f"Batch validation error: {str(e)}"
            logger.error(f"Qdrant batch collection validation failed: {str(e)}")

        return result

    async def validate_vector_dimensions(self, collection_name: str, expected_dimension: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate that the vector dimensions match expected values
        Returns a dictionary with validation results
        """
        result = {
            "service": f"Qdrant Vector Dimensions: {collection_name}",
            "status": "FAIL",
            "timestamp": None,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "error_message": None,
            "collection_name": collection_name,
            "expected_dimension": expected_dimension
        }

        try:
            # Get collection info to check vector dimensions
            collection_info = self.client.get_collection(collection_name)
            actual_dimension = collection_info.config.params.vectors.size

            result["actual_dimension"] = actual_dimension

            if expected_dimension is None or actual_dimension == expected_dimension:
                result["status"] = "OK"
            else:
                result["error_message"] = f"Dimension mismatch: expected {expected_dimension}, got {actual_dimension}"

            result["timestamp"] = asyncio.get_event_loop().time()

            logger.info(f"Qdrant vector dimensions validated for '{collection_name}': {actual_dimension}")

        except UnexpectedResponseException as e:
            result["error_message"] = f"Collection not found: {str(e)}"
            logger.error(f"Qdrant vector dimensions validation failed: {str(e)}")

        except Exception as e:
            result["error_message"] = f"Vector dimensions validation error: {str(e)}"
            logger.error(f"Qdrant vector dimensions validation failed: {str(e)}")

        return result

    async def validate_index_readiness(self, collection_name: str) -> Dict[str, Any]:
        """
        Validate that the collection index is ready for search
        Returns a dictionary with validation results
        """
        result = {
            "service": f"Qdrant Index Readiness: {collection_name}",
            "status": "FAIL",
            "timestamp": None,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "error_message": None,
            "collection_name": collection_name
        }

        try:
            # Get collection info to check index status
            collection_info = self.client.get_collection(collection_name)

            # Check if the collection has data
            points_count = collection_info.points_count
            result["points_count"] = points_count

            # Check if index is properly configured
            index_status = "ready" if points_count > 0 else "empty"
            result["index_status"] = index_status

            if points_count > 0:
                result["status"] = "OK"
            else:
                result["status"] = "WARNING"
                result["error_message"] = f"Collection is empty with {points_count} points"

            result["timestamp"] = asyncio.get_event_loop().time()

            logger.info(f"Qdrant index readiness validated for '{collection_name}': {points_count} points")

        except UnexpectedResponseException as e:
            result["error_message"] = f"Collection not found: {str(e)}"
            logger.error(f"Qdrant index readiness validation failed: {str(e)}")

        except Exception as e:
            result["error_message"] = f"Index readiness validation error: {str(e)}"
            logger.error(f"Qdrant index readiness validation failed: {str(e)}")

        return result

# Singleton instance
qdrant_validator = QdrantValidator()

async def validate_qdrant_connection() -> Dict[str, Any]:
    """Convenience function to validate Qdrant connection"""
    return await qdrant_validator.validate_connection()

async def validate_qdrant_collection(collection_name: str) -> Dict[str, Any]:
    """Convenience function to validate a Qdrant collection"""
    return await qdrant_validator.validate_collection_exists(collection_name)

async def validate_qdrant_collections(collection_names: List[str]) -> Dict[str, Any]:
    """Convenience function to validate multiple Qdrant collections"""
    return await qdrant_validator.validate_collections_exist(collection_names)

async def validate_qdrant_vector_dimensions(collection_name: str, expected_dimension: Optional[int] = None) -> Dict[str, Any]:
    """Convenience function to validate Qdrant vector dimensions"""
    return await qdrant_validator.validate_vector_dimensions(collection_name, expected_dimension)

async def validate_qdrant_index_readiness(collection_name: str) -> Dict[str, Any]:
    """Convenience function to validate Qdrant index readiness"""
    return await qdrant_validator.validate_index_readiness(collection_name)

# For testing purposes
if __name__ == "__main__":
    import asyncio

    async def main():
        print("Testing Qdrant connection validation...")
        result = await validate_qdrant_connection()
        print(f"Connection result: {result}")

        print("\nTesting Qdrant collection validation...")
        collection_result = await validate_qdrant_collection("book_content_chunks")
        print(f"Collection result: {collection_result}")

        print("\nTesting Qdrant collections batch validation...")
        batch_result = await validate_qdrant_collections(["book_content_chunks", "user_queries"])
        print(f"Batch result: {batch_result}")

    asyncio.run(main())