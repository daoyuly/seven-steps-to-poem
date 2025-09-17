"""
Core exception classes for Seven Steps to Poem system.

This module defines the exception hierarchy used throughout the system
for consistent error handling and reporting.
"""

from typing import Any, Dict, Optional


class SevenStepsError(Exception):
    """Base exception class for Seven Steps system."""
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}


class ConfigurationError(SevenStepsError):
    """Raised when there's a configuration error."""
    pass


class ValidationError(SevenStepsError):
    """Raised when data validation fails."""
    pass


class DatabaseError(SevenStepsError):
    """Raised when database operations fail."""
    pass


class ExternalServiceError(SevenStepsError):
    """Raised when external service calls fail."""
    pass


class LLMError(ExternalServiceError):
    """Raised when LLM API calls fail."""
    pass


class VectorDBError(ExternalServiceError):
    """Raised when vector database operations fail."""
    pass


class ObjectStorageError(ExternalServiceError):
    """Raised when object storage operations fail."""
    pass


class WorkflowError(SevenStepsError):
    """Raised when workflow execution fails."""
    pass


class AgentError(SevenStepsError):
    """Raised when agent execution fails."""
    pass


class ProblemFramerError(AgentError):
    """Raised when Problem Framer agent fails."""
    pass


class IssueTreeError(AgentError):
    """Raised when Issue Tree agent fails."""
    pass


class PrioritizationError(AgentError):
    """Raised when Prioritization agent fails."""
    pass


class PlannerError(AgentError):
    """Raised when Planner agent fails."""
    pass


class AnalysisError(AgentError):
    """Raised when Analysis agent fails."""
    pass


class SynthesizerError(AgentError):
    """Raised when Synthesizer agent fails."""
    pass


class PresentationError(AgentError):
    """Raised when Presentation agent fails."""
    pass


class AuthenticationError(SevenStepsError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(SevenStepsError):
    """Raised when authorization fails."""
    pass


class RateLimitError(SevenStepsError):
    """Raised when rate limits are exceeded."""
    pass


class ResourceNotFoundError(SevenStepsError):
    """Raised when a requested resource is not found."""
    pass


class ResourceConflictError(SevenStepsError):
    """Raised when a resource conflict occurs."""
    pass


class TimeoutError(SevenStepsError):
    """Raised when operations timeout."""
    pass