"""
Execution monitoring API routes.

This module provides endpoints for monitoring workflow execution
status, progress, and logs.
"""

from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from seven_steps.api.dependencies import (
    get_current_active_user,
    get_orchestrator_service,
    validate_problem_access,
)
from seven_steps.core.models import User
from seven_steps.core.orchestrator import SimpleOrchestrator
from seven_steps.core.schemas import ExecutionStatus

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/problems", tags=["executions"])


@router.get(
    "/{problem_id}/executions/{execution_id}",
    response_model=ExecutionStatus,
    summary="Get execution status",
    description="Check the status of a running analysis step"
)
async def get_execution_status(
    execution_id: str,
    problem_id: str = Depends(validate_problem_access),
    current_user: User = Depends(get_current_active_user),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_service)
) -> ExecutionStatus:
    """Get the status of a specific execution."""
    try:
        execution_status = await orchestrator.get_execution_status(
            problem_id=problem_id,
            execution_id=execution_id
        )
        
        if not execution_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found"
            )
        
        return execution_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get execution status",
            problem_id=problem_id,
            execution_id=execution_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve execution status"
        ) from e


@router.post(
    "/{problem_id}/executions/{execution_id}/cancel",
    response_model=Dict[str, Any],
    summary="Cancel execution",
    description="Cancel a running execution"
)
async def cancel_execution(
    execution_id: str,
    problem_id: str = Depends(validate_problem_access),
    current_user: User = Depends(get_current_active_user),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_service)
) -> Dict[str, Any]:
    """Cancel a running execution."""
    try:
        success = await orchestrator.cancel_execution(
            problem_id=problem_id,
            execution_id=execution_id
        )
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found or cannot be cancelled"
            )
        
        logger.info(
            "Execution cancelled",
            problem_id=problem_id,
            execution_id=execution_id,
            cancelled_by=current_user.id
        )
        
        return {
            "message": "Execution cancelled successfully",
            "execution_id": execution_id,
            "cancelled_by": current_user.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to cancel execution",
            problem_id=problem_id,
            execution_id=execution_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel execution"
        ) from e


@router.get(
    "/{problem_id}/progress",
    response_model=Dict[str, Any],
    summary="Get workflow progress",
    description="Get overall workflow progress for a problem"
)
async def get_workflow_progress(
    problem_id: str = Depends(validate_problem_access),
    current_user: User = Depends(get_current_active_user),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_service)
) -> Dict[str, Any]:
    """Get overall workflow progress for a problem."""
    try:
        progress = await orchestrator.get_workflow_progress(problem_id)
        
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow not found"
            )
        
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to get workflow progress",
            problem_id=problem_id,
            error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve workflow progress"
        ) from e