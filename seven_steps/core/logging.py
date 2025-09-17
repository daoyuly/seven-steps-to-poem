"""
Structured logging configuration for Seven Steps to Poem system.

This module provides centralized logging with structured output,
correlation IDs, and integration with monitoring systems.
"""

import logging
import logging.config
import sys
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor

from .config import get_settings

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)


def add_correlation_id(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add correlation ID to log entries."""
    cid = correlation_id.get()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def add_service_info(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add service information to log entries."""
    settings = get_settings()
    event_dict.update({
        "service": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
    })
    return event_dict


def setup_logging() -> None:
    """Configure structured logging for the application."""
    settings = get_settings()
    
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.monitoring.log_level.upper()),
    )
    
    # Configure structlog
    processors: list[Processor] = [
        # Add correlation ID and service info
        add_correlation_id,
        add_service_info,
        # Add timestamp
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        # Add stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]
    
    if settings.environment == "production":
        # JSON output for production
        processors.extend([
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer()
        ])
    else:
        # Human-readable output for development
        processors.extend([
            structlog.processors.ExceptionPrettyPrinter(),
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


def set_correlation_id(cid: Optional[str] = None) -> str:
    """Set correlation ID for current context."""
    if cid is None:
        cid = str(uuid.uuid4())
    correlation_id.set(cid)
    return cid


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID."""
    return correlation_id.get()


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls.logger = get_logger(cls.__name__)
    
    @property
    def logger(self) -> structlog.BoundLogger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


# Initialize logging on import
setup_logging()