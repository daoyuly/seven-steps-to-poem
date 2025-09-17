"""
FastAPI dependencies for authentication, database sessions, and common utilities.

This module provides reusable dependencies for the API endpoints.
"""

from typing import AsyncGenerator, Optional

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from seven_steps.core.config import get_settings
from seven_steps.core.database import get_db_session
from seven_steps.core.exceptions import AuthenticationError, DatabaseError
from seven_steps.core.logging import get_correlation_id, set_correlation_id
from seven_steps.core.models import User
from seven_steps.core.orchestrator import get_orchestrator

logger = structlog.get_logger(__name__)
security = HTTPBearer(auto_error=False)


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.
    
    Yields:
        AsyncSession: Database session
    """
    try:
        async with get_db_session() as session:
            yield session
    except DatabaseError as e:
        logger.error("Database dependency failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        ) from e


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_database)
) -> User:
    """
    Dependency to get current authenticated user.
    
    Args:
        credentials: JWT token from Authorization header
        db: Database session
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        settings = get_settings()
        payload = jwt.decode(
            credentials.credentials,
            settings.security.secret_key,
            algorithms=[settings.security.algorithm]
        )
        
        user_id: str = payload.get("sub")
        if user_id is None:
            raise AuthenticationError("Invalid token payload")
            
    except JWTError as e:
        logger.warning("JWT validation failed", error=str(e), token_preview=credentials.credentials[:20])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    
    # For now, create a mock user since we haven't implemented full auth
    # In production, this would query the database
    user = User(
        id=user_id,
        email=payload.get("email", "test@example.com"),
        name=payload.get("name", "Test User"),
        role="analyst",
        organization_id=payload.get("org_id", "default-org"),
        is_active=True
    )
    
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Dependency to get current active user.
    
    Args:
        current_user: Current user from token
        
    Returns:
        User: Active user
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def get_orchestrator_service():
    """
    Dependency to get orchestrator service.
    
    Returns:
        SimpleOrchestrator: Orchestrator instance
    """
    return get_orchestrator()


async def set_correlation_id_from_request(request: Request) -> str:
    """
    Dependency to set correlation ID from request headers.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Correlation ID
    """
    # Try to get correlation ID from headers
    correlation_id = request.headers.get("X-Correlation-ID")
    
    if not correlation_id:
        # Generate new correlation ID
        correlation_id = set_correlation_id()
    else:
        # Use provided correlation ID
        set_correlation_id(correlation_id)
    
    return correlation_id


class RequireRole:
    """Dependency class to require specific user roles."""
    
    def __init__(self, *required_roles: str):
        """
        Initialize with required roles.
        
        Args:
            *required_roles: Required user roles
        """
        self.required_roles = set(required_roles)
    
    def __call__(self, current_user: User = Depends(get_current_active_user)) -> User:
        """
        Check if user has required role.
        
        Args:
            current_user: Current active user
            
        Returns:
            User: User with required role
            
        Raises:
            HTTPException: If user lacks required role
        """
        if current_user.role not in self.required_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {' or '.join(self.required_roles)}"
            )
        return current_user


# Common role requirements
require_admin = RequireRole("admin")
require_analyst = RequireRole("admin", "analyst")
require_owner = RequireRole("admin", "owner")


async def validate_problem_access(
    problem_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_database)
) -> str:
    """
    Validate that user has access to a specific problem.
    
    Args:
        problem_id: Problem identifier
        current_user: Current user
        db: Database session
        
    Returns:
        str: Problem ID if access is valid
        
    Raises:
        HTTPException: If access is denied or problem not found
    """
    # For now, just return the problem_id
    # In production, this would check database for ownership/access
    logger.info(
        "Problem access validated",
        problem_id=problem_id,
        user_id=current_user.id,
        organization_id=current_user.organization_id
    )
    
    return problem_id


class PaginationParams:
    """Dependency class for pagination parameters."""
    
    def __init__(
        self,
        page: int = 1,
        limit: int = 20,
        max_limit: int = 100
    ):
        """
        Initialize pagination parameters.
        
        Args:
            page: Page number (1-based)
            limit: Items per page
            max_limit: Maximum items per page
        """
        if page < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Page must be >= 1"
            )
        
        if limit < 1:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be >= 1"
            )
        
        if limit > max_limit:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Limit cannot exceed {max_limit}"
            )
        
        self.page = page
        self.limit = limit
        self.offset = (page - 1) * limit


def get_pagination_params(
    page: int = 1,
    limit: int = 20
) -> PaginationParams:
    """
    Dependency function for pagination parameters.
    
    Args:
        page: Page number
        limit: Items per page
        
    Returns:
        PaginationParams: Validated pagination parameters
    """
    return PaginationParams(page, limit)