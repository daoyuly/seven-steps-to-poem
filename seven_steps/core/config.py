"""
Core configuration management for Seven Steps to Poem system.

This module provides centralized configuration management with environment
variable support, validation, and type safety.
"""

import secrets
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseSettings, Field, validator
from pydantic_settings import BaseSettings as PydanticBaseSettings


class DatabaseSettings(BaseSettings):
    """Database connection settings."""
    
    url: str = Field(
        default="postgresql://postgres:password@localhost:5432/sevensteps",
        env="DATABASE_URL",
        description="PostgreSQL connection URL"
    )
    pool_size: int = Field(default=10, env="DATABASE_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DATABASE_MAX_OVERFLOW")
    pool_timeout: int = Field(default=30, env="DATABASE_POOL_TIMEOUT")
    echo: bool = Field(default=False, env="DATABASE_ECHO")


class RedisSettings(BaseSettings):
    """Redis connection settings."""
    
    url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis connection URL"
    )
    max_connections: int = Field(default=100, env="REDIS_MAX_CONNECTIONS")
    socket_timeout: int = Field(default=5, env="REDIS_SOCKET_TIMEOUT")
    socket_connect_timeout: int = Field(default=5, env="REDIS_SOCKET_CONNECT_TIMEOUT")


class Neo4jSettings(BaseSettings):
    """Neo4j connection settings."""
    
    url: str = Field(
        default="bolt://localhost:7687",
        env="NEO4J_URL",
        description="Neo4j connection URL"
    )
    username: str = Field(default="neo4j", env="NEO4J_USERNAME")
    password: str = Field(default="password", env="NEO4J_PASSWORD")
    database: str = Field(default="neo4j", env="NEO4J_DATABASE")


class LLMSettings(BaseSettings):
    """Large Language Model settings."""
    
    provider: str = Field(default="openai", env="LLM_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(default=None, env="ANTHROPIC_API_KEY")
    model: str = Field(default="gpt-4-turbo", env="LLM_MODEL")
    temperature: float = Field(default=0.1, env="LLM_TEMPERATURE")
    max_tokens: int = Field(default=2000, env="LLM_MAX_TOKENS")
    timeout: int = Field(default=60, env="LLM_TIMEOUT")
    
    @validator("provider")
    def validate_provider(cls, v: str) -> str:
        if v not in ["openai", "anthropic"]:
            raise ValueError("LLM provider must be 'openai' or 'anthropic'")
        return v


class VectorDBSettings(BaseSettings):
    """Vector database settings."""
    
    provider: str = Field(default="pinecone", env="VECTOR_DB_PROVIDER")
    pinecone_api_key: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    pinecone_environment: str = Field(default="us-west1-gcp", env="PINECONE_ENVIRONMENT")
    index_name: str = Field(default="seven-steps-problems", env="VECTOR_DB_INDEX_NAME")
    dimension: int = Field(default=1536, env="VECTOR_DB_DIMENSION")


class ObjectStorageSettings(BaseSettings):
    """Object storage settings."""
    
    provider: str = Field(default="minio", env="OBJECT_STORAGE_PROVIDER")
    endpoint: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    access_key: str = Field(default="minioadmin", env="MINIO_ACCESS_KEY")
    secret_key: str = Field(default="minioadmin", env="MINIO_SECRET_KEY")
    bucket_name: str = Field(default="seven-steps-bucket", env="MINIO_BUCKET_NAME")
    secure: bool = Field(default=False, env="MINIO_SECURE")


class MessageQueueSettings(BaseSettings):
    """Message queue settings."""
    
    provider: str = Field(default="redis", env="MESSAGE_QUEUE_PROVIDER")
    redis_url: str = Field(
        default="redis://localhost:6379/1",
        env="MESSAGE_QUEUE_REDIS_URL"
    )
    kafka_bootstrap_servers: List[str] = Field(
        default=["localhost:9092"],
        env="KAFKA_BOOTSTRAP_SERVERS"
    )


class SecuritySettings(BaseSettings):
    """Security and authentication settings."""
    
    secret_key: str = Field(
        default_factory=lambda: secrets.token_urlsafe(32),
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        env="REFRESH_TOKEN_EXPIRE_DAYS"
    )


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings."""
    
    prometheus_enabled: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(default=8090, env="PROMETHEUS_PORT")
    jaeger_enabled: bool = Field(default=True, env="JAEGER_ENABLED")
    jaeger_agent_host: str = Field(default="localhost", env="JAEGER_AGENT_HOST")
    jaeger_agent_port: int = Field(default=6831, env="JAEGER_AGENT_PORT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")


class Settings(PydanticBaseSettings):
    """Main application settings."""
    
    # Application settings
    app_name: str = Field(default="Seven Steps to Poem", env="APP_NAME")
    version: str = Field(default="1.0.0", env="APP_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    environment: str = Field(default="development", env="ENVIRONMENT")
    
    # Service settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    redis: RedisSettings = RedisSettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    llm: LLMSettings = LLMSettings()
    vector_db: VectorDBSettings = VectorDBSettings()
    object_storage: ObjectStorageSettings = ObjectStorageSettings()
    message_queue: MessageQueueSettings = MessageQueueSettings()
    security: SecuritySettings = SecuritySettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # Workflow settings
    max_execution_time: int = Field(default=3600, env="MAX_EXECUTION_TIME")  # 1 hour
    retry_attempts: int = Field(default=3, env="RETRY_ATTEMPTS")
    retry_delay: int = Field(default=5, env="RETRY_DELAY")  # seconds
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        if v not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get global settings instance."""
    return settings