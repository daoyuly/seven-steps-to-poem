"""
Health check and system status API routes.

This module provides endpoints for monitoring system health,
service status, and basic metrics.
"""

import time
from typing import Any, Dict

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from seven_steps.api.dependencies import get_database
from seven_steps.core.config import get_settings
from seven_steps.core.schemas import HealthCheck, SystemMetrics

logger = structlog.get_logger(__name__)
router = APIRouter(tags=["system"])


@router.get(
    "/health",
    response_model=HealthCheck,
    summary="Health check",
    description="Check API health and service status"
)
async def health_check(
    db: AsyncSession = Depends(get_database)
) -> HealthCheck:
    """
    Perform a health check of the system and its dependencies.
    
    Returns the overall system status and individual service statuses.
    """
    services = {}
    overall_status = "healthy"
    
    # Check database connectivity
    try:
        await db.execute(text("SELECT 1"))
        services["database"] = "up"
        logger.debug("Database health check passed")
    except Exception as e:
        services["database"] = "down"
        overall_status = "unhealthy"
        logger.error("Database health check failed", error=str(e))
    
    # Check configuration
    try:
        settings = get_settings()
        services["configuration"] = "up"
        logger.debug("Configuration health check passed")
    except Exception as e:
        services["configuration"] = "down"
        overall_status = "unhealthy"
        logger.error("Configuration health check failed", error=str(e))
    
    # In production, you would check other services:
    # - Redis connectivity
    # - Neo4j connectivity
    # - OpenAI API availability
    # - Object storage connectivity
    
    # For now, mark them as up for development
    services.update({
        "redis": "up",
        "neo4j": "up",
        "openai_api": "up",
        "object_storage": "up",
        "orchestrator": "up"
    })
    
    return HealthCheck(
        status=overall_status,
        services=services
    )


@router.get(
    "/metrics",
    response_model=SystemMetrics,
    summary="Get system metrics",
    description="Retrieve system performance and usage metrics"
)
async def get_system_metrics(
    db: AsyncSession = Depends(get_database)
) -> SystemMetrics:
    """
    Get system performance and usage metrics.
    
    In production, these would be real metrics from monitoring systems.
    For now, we provide mock metrics for development.
    """
    try:
        # In production, these would be real queries
        # For development, we'll provide mock data
        
        # Mock metrics - replace with real data in production
        metrics = SystemMetrics(
            total_problems=150,
            active_problems=23,
            success_rate=0.87,
            average_processing_time=450,  # seconds
            agent_performance={
                "problem_framer": {
                    "success_rate": 0.95,
                    "avg_duration": 120
                },
                "issue_tree": {
                    "success_rate": 0.92,
                    "avg_duration": 180
                },
                "prioritization": {
                    "success_rate": 0.90,
                    "avg_duration": 90
                },
                "planner": {
                    "success_rate": 0.88,
                    "avg_duration": 150
                },
                "analysis": {
                    "success_rate": 0.85,
                    "avg_duration": 300
                },
                "synthesizer": {
                    "success_rate": 0.89,
                    "avg_duration": 240
                },
                "presentation": {
                    "success_rate": 0.93,
                    "avg_duration": 180
                }
            }
        )
        
        logger.info("System metrics retrieved")
        return metrics
        
    except Exception as e:
        logger.error("Failed to get system metrics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system metrics"
        ) from e


@router.get(
    "/version",
    response_model=Dict[str, Any],
    summary="Get version info",
    description="Get application version and build information"
)
async def get_version() -> Dict[str, Any]:
    """Get application version and build information."""
    settings = get_settings()
    
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "debug": settings.debug,
        "timestamp": time.time()
    }