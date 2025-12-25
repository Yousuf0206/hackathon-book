"""
Neon PostgreSQL Connection Validator
This module provides utilities to validate Neon database connections.
"""
import asyncio
import logging
from typing import Dict, Any
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

class NeonDBValidator:
    """Utility class for validating Neon PostgreSQL connections"""

    def __init__(self):
        self.db_url = os.getenv("NEON_DATABASE_URL")
        if not self.db_url:
            raise ValueError("NEON_DATABASE_URL environment variable is not set")

    async def validate_connection(self) -> Dict[str, Any]:
        """
        Validate the Neon PostgreSQL connection
        Returns a dictionary with validation results
        """
        result = {
            "service": "Neon PostgreSQL",
            "status": "FAIL",
            "timestamp": None,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "error_message": None
        }

        try:
            # Create an async engine to test the connection
            engine = create_async_engine(self.db_url)
            async_session = async_sessionmaker(engine, expire_on_commit=False)

            # Try to execute a simple query
            async with async_session() as session:
                result_db = await session.execute(text("SELECT 1"))
                _ = result_db.scalar()

            # If we get here, the connection is successful
            result["status"] = "OK"
            result["timestamp"] = asyncio.get_event_loop().time()

            # Close the engine
            await engine.dispose()

            logger.info("Neon PostgreSQL connection validated successfully")

        except SQLAlchemyError as e:
            result["error_message"] = f"SQLAlchemy error: {str(e)}"
            logger.error(f"Neon connection validation failed: {str(e)}")

        except Exception as e:
            result["error_message"] = f"Connection error: {str(e)}"
            logger.error(f"Neon connection validation failed: {str(e)}")

        return result

    async def validate_schema(self) -> Dict[str, Any]:
        """
        Validate that required Neon schema/tables exist
        Returns a dictionary with schema validation results
        """
        result = {
            "service": "Neon PostgreSQL Schema",
            "status": "FAIL",
            "timestamp": None,
            "environment": os.getenv("ENVIRONMENT", "development"),
            "error_message": None,
            "tables_validated": []
        }

        try:
            engine = create_async_engine(self.db_url)
            async_session = async_sessionmaker(engine, expire_on_commit=False)

            # Check for common tables that would be expected in a RAG system
            tables_to_check = ["users", "chat_sessions", "user_queries", "ai_responses"]

            async with async_session() as session:
                for table in tables_to_check:
                    try:
                        # Check if table exists by querying information schema
                        query = text(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.tables
                                WHERE table_name = '{table}'
                            );
                        """)
                        result_db = await session.execute(query)
                        table_exists = result_db.scalar()

                        if table_exists:
                            result["tables_validated"].append({"table": table, "exists": True})
                        else:
                            result["tables_validated"].append({"table": table, "exists": False})

                    except Exception:
                        # If table doesn't exist, that's ok, we'll mark it as doesn't exist
                        result["tables_validated"].append({"table": table, "exists": False})

            # Check if at least some tables exist (or create basic ones if needed)
            existing_tables = [t for t in result["tables_validated"] if t["exists"]]
            if len(existing_tables) > 0:
                result["status"] = "OK"
            else:
                result["status"] = "WARNING"  # Tables don't exist but connection is ok
                result["error_message"] = "No expected tables found in database"

            result["timestamp"] = asyncio.get_event_loop().time()
            await engine.dispose()

            logger.info(f"Neon schema validation completed: {len(existing_tables)} tables found")

        except Exception as e:
            result["error_message"] = f"Schema validation error: {str(e)}"
            logger.error(f"Neon schema validation failed: {str(e)}")

        return result

# Singleton instance
neon_validator = NeonDBValidator()

async def validate_neon_connection() -> Dict[str, Any]:
    """Convenience function to validate Neon connection"""
    return await neon_validator.validate_connection()

async def validate_neon_schema() -> Dict[str, Any]:
    """Convenience function to validate Neon schema"""
    return await neon_validator.validate_schema()

# For testing purposes
if __name__ == "__main__":
    import asyncio

    async def main():
        print("Testing Neon connection validation...")
        result = await validate_neon_connection()
        print(f"Connection result: {result}")

        print("\nTesting Neon schema validation...")
        schema_result = await validate_neon_schema()
        print(f"Schema result: {schema_result}")

    asyncio.run(main())