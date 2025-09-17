"""
Main FastAPI application setup and configuration.

This module creates and configures the FastAPI application with
all routes, middleware, and startup/shutdown handlers.
"""

import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from seven_steps.api.dependencies import set_correlation_id_from_request
from seven_steps.api.routes import executions, health, problems
from seven_steps.core.config import get_settings
from seven_steps.core.database import close_database, init_database
from seven_steps.core.exceptions import SevenStepsError
from seven_steps.core.logging import get_correlation_id, get_logger, setup_logging
from seven_steps.core.schemas import ErrorResponse

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Args:
        app: FastAPI application instance
    """
    # Startup
    logger.info("Starting Seven Steps to Poem API")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")
        
        # Initialize other services here (Redis, Neo4j, etc.)
        # For now, we'll just log that we're ready
        logger.info("All services initialized")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down Seven Steps to Poem API")
    
    try:
        # Cleanup database connections
        await close_database()
        logger.info("Database connections closed")
        
        # Cleanup other services here
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))
    
    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured application instance
    """
    settings = get_settings()
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="AI Agent system for automated business problem solving using McKinsey's 7-step methodology",
        version=settings.version,
        debug=settings.debug,
        lifespan=lifespan,
        # OpenAPI customization
        openapi_tags=[
            {
                "name": "problems",
                "description": "Problem management and lifecycle operations"
            },
            {
                "name": "executions", 
                "description": "Workflow execution monitoring and control"
            },
            {
                "name": "system",
                "description": "System health, metrics, and status"
            }
        ]
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["https://app.sevenstepstopoem.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        """Add correlation ID to all requests."""
        # Set correlation ID from request
        await set_correlation_id_from_request(request)
        correlation_id = get_correlation_id()
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add correlation ID to response headers
        if correlation_id:
            response.headers["X-Correlation-ID"] = correlation_id
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log request
        logger.info(
            "Request processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time=process_time,
            correlation_id=correlation_id
        )
        
        return response
    
    @app.middleware("http")
    async def error_handling_middleware(request: Request, call_next):
        """Global error handling middleware."""
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(
                "Unhandled exception",
                error=str(e),
                path=request.url.path,
                method=request.method,
                correlation_id=get_correlation_id()
            )
            
            # Return generic error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error_code="INTERNAL_ERROR",
                    error_message="An internal error occurred",
                    request_id=get_correlation_id()
                ).dict()
            )
    
    # Custom exception handlers
    @app.exception_handler(SevenStepsError)
    async def seven_steps_exception_handler(request: Request, exc: SevenStepsError):
        """Handle custom Seven Steps exceptions."""
        logger.error(
            "Seven Steps error",
            error_code=exc.error_code,
            error_message=exc.message,
            details=exc.details,
            path=request.url.path,
            correlation_id=get_correlation_id()
        )
        
        # Map error types to HTTP status codes
        status_code_mapping = {
            "ValidationError": status.HTTP_400_BAD_REQUEST,
            "AuthenticationError": status.HTTP_401_UNAUTHORIZED,
            "AuthorizationError": status.HTTP_403_FORBIDDEN,
            "ResourceNotFoundError": status.HTTP_404_NOT_FOUND,
            "ResourceConflictError": status.HTTP_409_CONFLICT,
            "RateLimitError": status.HTTP_429_TOO_MANY_REQUESTS,
            "TimeoutError": status.HTTP_408_REQUEST_TIMEOUT,
        }
        
        status_code = status_code_mapping.get(
            exc.error_code,
            status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        
        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error_code=exc.error_code,
                error_message=exc.message,
                details=exc.details,
                request_id=get_correlation_id()
            ).dict()
        )
    
    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(request: Request, exc: HTTPException):
        """Enhanced HTTP exception handler with correlation ID."""
        correlation_id = get_correlation_id()
        
        # Log the error
        logger.warning(
            "HTTP exception",
            status_code=exc.status_code,
            detail=exc.detail,
            path=request.url.path,
            correlation_id=correlation_id
        )
        
        # Call default handler but add correlation ID
        response = await http_exception_handler(request, exc)
        if correlation_id:
            response.headers["X-Correlation-ID"] = correlation_id
        
        return response
    
    # Include routers
    app.include_router(health.router, prefix="/v1")
    app.include_router(problems.router, prefix="/v1")
    app.include_router(executions.router, prefix="/v1")
    
    # Root endpoint
    @app.get("/", include_in_schema=False)
    async def root() -> Dict[str, Any]:
        """Root endpoint with API information."""
        return {
            "name": settings.app_name,
            "version": settings.version,
            "environment": settings.environment,
            "docs_url": "/docs",
            "health_url": "/v1/health",
            "status": "running"
        }
    
    logger.info(
        "FastAPI application created",
        app_name=settings.app_name,
        version=settings.version,
        debug=settings.debug,
        environment=settings.environment
    )
    
    return app


# Create the application instance
app = create_app()


def main() -> None:
    """Main entry point for running the application."""
    settings = get_settings()
    
    # Run with uvicorn
    uvicorn.run(
        "seven_steps.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_config=None,  # Disable uvicorn's logging config
        access_log=False,  # We handle access logging in middleware
    )


if __name__ == "__main__":
    main()