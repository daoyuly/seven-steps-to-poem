"""
Database connection and session management for Seven Steps to Poem system.

This module provides async database connections, session management,
and connection pooling for PostgreSQL.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import structlog
from sqlalchemy import MetaData, event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool, QueuePool

from .config import get_settings
from .exceptions import DatabaseError

logger = structlog.get_logger(__name__)

# Global engine and session maker
_engine: Optional[AsyncEngine] = None
_session_maker: Optional[async_sessionmaker[AsyncSession]] = None


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    
    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key constraints for SQLite."""
    if "sqlite" in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_engine() -> AsyncEngine:
    """Create async database engine with connection pooling."""
    settings = get_settings()
    
    # Configure connection pool based on environment
    if settings.debug:
        pool_class = NullPool
        pool_kwargs = {}
    else:
        pool_class = QueuePool
        pool_kwargs = {
            "pool_size": settings.database.pool_size,
            "max_overflow": settings.database.max_overflow,
            "pool_timeout": settings.database.pool_timeout,
            "pool_pre_ping": True,
        }
    
    engine = create_async_engine(
        settings.database.url,
        echo=settings.database.echo,
        poolclass=pool_class,
        **pool_kwargs,
    )
    
    logger.info(
        "Created database engine",
        url=settings.database.url,
        pool_size=settings.database.pool_size,
        echo=settings.database.echo,
    )
    
    return engine


def get_engine() -> AsyncEngine:
    """Get the global database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine()
    return _engine


def get_session_maker() -> async_sessionmaker[AsyncSession]:
    """Get the global session maker."""
    global _session_maker
    if _session_maker is None:
        engine = get_engine()
        _session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_maker


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session with automatic cleanup.
    
    This context manager ensures proper session cleanup even if
    exceptions occur during database operations.
    
    Yields:
        AsyncSession: Database session for operations
        
    Raises:
        DatabaseError: If database connection fails
    """
    session_maker = get_session_maker()
    
    async with session_maker() as session:
        try:
            logger.debug("Created database session")
            yield session
            await session.commit()
            logger.debug("Committed database session")
        except Exception as e:
            logger.error("Database session error, rolling back", error=str(e))
            await session.rollback()
            raise DatabaseError(
                message=f"Database operation failed: {str(e)}",
                details={"original_error": str(e)}
            ) from e
        finally:
            await session.close()
            logger.debug("Closed database session")


async def init_database() -> None:
    """Initialize database connection and verify connectivity."""
    try:
        engine = get_engine()
        
        # Test connection
        async with engine.begin() as conn:
            await conn.run_sync(lambda sync_conn: sync_conn.execute("SELECT 1"))
        
        logger.info("Database connection established successfully")
        
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise DatabaseError(
            message="Database initialization failed",
            details={"error": str(e)}
        ) from e


async def close_database() -> None:
    """Close database connections and cleanup resources."""
    global _engine, _session_maker
    
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine disposed")
    
    _session_maker = None
    logger.info("Database cleanup completed")