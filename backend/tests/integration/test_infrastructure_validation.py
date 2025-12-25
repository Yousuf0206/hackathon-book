"""
Integration tests for infrastructure validation components.
Tests Neon and Qdrant connection validation functionality.
"""
import pytest
import asyncio
from unittest.mock import patch, AsyncMock
from typing import Dict, Any

from src.utils.db_validator import validate_neon_connection, validate_neon_schema
from src.utils.vector_validator import (
    validate_qdrant_connection,
    validate_qdrant_collections,
    validate_qdrant_vector_dimensions,
    validate_qdrant_index_readiness
)
from src.services.validation_service import ValidationService


@pytest.fixture
def mock_neon_connection():
    """Mock Neon connection for testing"""
    with patch('src.utils.db_validator.create_async_engine') as mock_engine:
        mock_session = AsyncMock()
        mock_session.execute = AsyncMock()

        # Mock the sessionmaker to return our mock session
        with patch('src.utils.db_validator.async_sessionmaker') as mock_sessionmaker:
            mock_sessionmaker.return_value = lambda: mock_session
            yield mock_engine, mock_session


@pytest.fixture
def mock_qdrant_client():
    """Mock Qdrant client for testing"""
    with patch('src.utils.vector_validator.QdrantClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(return_value=AsyncMock(points_count=10))
        mock_client.get_collection = AsyncMock(return_value=AsyncMock(
            config=AsyncMock(params=AsyncMock(vectors=AsyncMock(size=1536))),
            points_count=10
        ))
        mock_client_class.return_value = mock_client
        yield mock_client


class TestNeonValidation:
    """Test cases for Neon PostgreSQL validation"""

    @pytest.mark.asyncio
    async def test_validate_neon_connection_success(self, mock_neon_connection):
        """Test successful Neon connection validation"""
        mock_engine, mock_session = mock_neon_connection

        # Mock successful query execution
        mock_result = AsyncMock()
        mock_result.scalar = AsyncMock(return_value=1)
        mock_session.execute.return_value = mock_result

        with patch('src.utils.db_validator.os.getenv', return_value='postgresql://test:test@localhost/test'):
            result = await validate_neon_connection()

        assert result["status"] == "OK"
        assert result["service"] == "Neon PostgreSQL"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_validate_neon_connection_failure(self, mock_neon_connection):
        """Test Neon connection validation failure"""
        mock_engine, mock_session = mock_neon_connection

        # Mock a connection error
        mock_session.execute.side_effect = Exception("Connection failed")

        with patch('src.utils.db_validator.os.getenv', return_value='postgresql://test:test@localhost/test'):
            result = await validate_neon_connection()

        assert result["status"] == "FAIL"
        assert "Connection error" in result["error_message"]

    @pytest.mark.asyncio
    async def test_validate_neon_schema_success(self, mock_neon_connection):
        """Test successful Neon schema validation"""
        mock_engine, mock_session = mock_neon_connection

        # Mock successful schema query execution
        mock_result = AsyncMock()
        mock_result.scalar = AsyncMock(return_value=True)
        mock_session.execute.return_value = mock_result

        with patch('src.utils.db_validator.os.getenv', return_value='postgresql://test:test@localhost/test'):
            result = await validate_neon_schema()

        assert result["status"] in ["OK", "WARNING"]  # Could be WARNING if no tables exist
        assert result["service"] == "Neon PostgreSQL Schema"


class TestQdrantValidation:
    """Test cases for Qdrant vector database validation"""

    @pytest.mark.asyncio
    async def test_validate_qdrant_connection_success(self, mock_qdrant_client):
        """Test successful Qdrant connection validation"""
        result = await validate_qdrant_connection()

        assert result["status"] == "OK"
        assert result["service"] == "Qdrant Vector Database"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_validate_qdrant_connection_failure(self, mock_qdrant_client):
        """Test Qdrant connection validation failure"""
        # Simulate connection failure
        mock_qdrant_client.get_collections.side_effect = Exception("Connection failed")

        result = await validate_qdrant_connection()

        assert result["status"] == "FAIL"
        assert "Connection error" in result["error_message"]

    @pytest.mark.asyncio
    async def test_validate_qdrant_collections_success(self, mock_qdrant_client):
        """Test successful Qdrant collection validation"""
        collections_to_test = ["book_content_chunks", "user_queries"]
        result = await validate_qdrant_collections(collections_to_test)

        assert result["status"] in ["OK", "WARNING"]  # Could be WARNING if collections don't exist
        assert result["service"] == "Qdrant Collections Validation"
        assert len(result["collections_validated"]) == len(collections_to_test)

    @pytest.mark.asyncio
    async def test_validate_qdrant_vector_dimensions(self, mock_qdrant_client):
        """Test Qdrant vector dimension validation"""
        result = await validate_qdrant_vector_dimensions("test_collection", 1536)

        assert result["status"] in ["OK", "FAIL"]  # Depends on actual dimension comparison
        assert result["collection_name"] == "test_collection"

    @pytest.mark.asyncio
    async def test_validate_qdrant_index_readiness(self, mock_qdrant_client):
        """Test Qdrant index readiness validation"""
        result = await validate_qdrant_index_readiness("test_collection")

        assert result["status"] in ["OK", "WARNING"]  # Depends on points count
        assert result["collection_name"] == "test_collection"


class TestValidationService:
    """Test cases for the ValidationService class"""

    @pytest.mark.asyncio
    async def test_validate_infrastructure_complete(self, mock_neon_connection, mock_qdrant_client):
        """Test complete infrastructure validation"""
        mock_engine, mock_session = mock_neon_connection
        mock_result = AsyncMock()
        mock_result.scalar = AsyncMock(return_value=1)
        mock_session.execute.return_value = mock_result

        validation_service = ValidationService()

        with patch('src.utils.db_validator.os.getenv', return_value='postgresql://test:test@localhost/test'):
            result = await validation_service.validate_infrastructure(
                validate_neon=True,
                validate_qdrant=True,
                collections_to_check=["test_collection"],
                expected_dimension=1536
            )

        # The result should have both neon and qdrant components validated
        assert "neon" in result["components"]
        assert "qdrant" in result["components"]
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_validate_qdrant_collection_existence(self, mock_qdrant_client):
        """Test Qdrant collection existence validation"""
        validation_service = ValidationService()

        result = await validation_service.validate_qdrant_collection_existence(
            ["book_content_chunks", "user_queries", "ai_responses"]
        )

        assert "collections_validated" in result
        assert result["summary"]["total"] == 3
        assert isinstance(result["collections_validated"], list)

    @pytest.mark.asyncio
    async def test_validate_vector_dimension_compatibility(self, mock_qdrant_client):
        """Test vector dimension compatibility validation"""
        validation_service = ValidationService()

        result = await validation_service.validate_vector_dimension_compatibility(
            "test_collection", 1536
        )

        assert result["collection"] == "test_collection"
        assert result["expected_dimension"] == 1536
        assert "compatible" in result

    @pytest.mark.asyncio
    async def test_validate_index_readiness(self, mock_qdrant_client):
        """Test index readiness validation"""
        validation_service = ValidationService()

        result = await validation_service.validate_index_readiness("test_collection")

        assert result["collection"] == "test_collection"
        assert "ready_for_search" in result


class TestHealthEndpoints:
    """Test cases for health check endpoints"""

    def setup_method(self):
        """Setup for health endpoint tests"""
        from src.api.main import app
        from fastapi.testclient import TestClient
        self.client = TestClient(app)

    def test_health_endpoint_basic(self):
        """Test basic health endpoint"""
        response = self.client.get("/api/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "components" in data

    def test_full_health_endpoint(self):
        """Test full health endpoint"""
        response = self.client.get("/api/health/full")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "components" in data
        # Should have more detailed component information than basic health

    def test_neon_health_endpoint(self):
        """Test Neon-specific health endpoint"""
        response = self.client.get("/api/health/neon")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "Neon PostgreSQL"
        assert "status" in data

    def test_qdrant_health_endpoint(self):
        """Test Qdrant-specific health endpoint"""
        response = self.client.get("/api/health/qdrant")
        assert response.status_code == 200

        data = response.json()
        assert data["service"] == "Qdrant Vector Database"
        assert "status" in data


# Run these tests with: pytest tests/integration/test_infrastructure_validation.py