"""
Problem management API routes.

This module provides REST endpoints for creating, managing, and
tracking business problems through the Seven Steps methodology.
"""

from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from seven_steps.agents.problem_framer import ProblemFramer
from seven_steps.api.dependencies import (
    PaginationParams,
    get_current_active_user,
    get_database,
    get_orchestrator_service,
    get_pagination_params,
    validate_problem_access,
)
from seven_steps.core.exceptions import ValidationError, WorkflowError
from seven_steps.core.models import Problem, User
from seven_steps.core.orchestrator import SimpleOrchestrator, WorkflowStep
from seven_steps.core.schemas import (
    ErrorResponse,
    PaginationInfo,
    ProblemCreateRequest,
    ProblemDetail,
    ProblemResponse,
    ProblemSummary,
)

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/problems", tags=["problems"])


@router.post(
    "",
    response_model=ProblemResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new problem",
    description="Submit a business problem for AI agent analysis"
)
async def create_problem(
    problem_data: ProblemCreateRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_database),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_service)
) -> ProblemResponse:
    """
    Create a new business problem for analysis.
    
    This endpoint creates a new problem record and initializes the
    workflow for processing through the Seven Steps methodology.
    """
    try:
        logger.info(
            "Creating new problem",
            submitter=problem_data.submitter,
            urgency=problem_data.urgency,
            user_id=current_user.id
        )
        
        # Create problem record
        problem = Problem(
            raw_text=problem_data.raw_text,
            submitter=problem_data.submitter,
            urgency=problem_data.urgency,
            metadata=problem_data.metadata,
            created_by=current_user.id,
            organization_id=current_user.organization_id,
            status="created"
        )
        
        db.add(problem)
        await db.commit()
        await db.refresh(problem)
        
        # Initialize workflow
        await orchestrator.create_workflow(
            problem_id=problem.id,
            user_id=current_user.id,
            organization_id=current_user.organization_id
        )
        
        logger.info(
            "Problem created successfully",
            problem_id=problem.id,
            user_id=current_user.id
        )
        
        return ProblemResponse(
            id=problem.id,
            status=problem.status,
            created_at=problem.created_at
        )
        
    except Exception as e:
        logger.error("Failed to create problem", error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create problem"
        ) from e


@router.get(
    "",
    response_model=Dict[str, Any],
    summary="List problems",
    description="Retrieve a paginated list of problems for the current user"
)
async def list_problems(
    pagination: PaginationParams = Depends(get_pagination_params),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    urgency_filter: Optional[str] = Query(None, description="Filter by urgency"),
    sort: str = Query("created_at", description="Sort field"),
    order: str = Query("desc", description="Sort order (asc/desc)"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_database)
) -> Dict[str, Any]:
    """
    List problems for the current user's organization.
    
    Returns a paginated list of problems with filtering and sorting options.
    """
    try:
        # Build query
        query = select(Problem).where(
            Problem.organization_id == current_user.organization_id
        )
        
        # Apply filters
        if status_filter:
            query = query.where(Problem.status == status_filter)
        
        if urgency_filter:
            query = query.where(Problem.urgency == urgency_filter)
        
        # Apply sorting
        sort_column = getattr(Problem, sort, Problem.created_at)
        if order.lower() == "desc":
            query = query.order_by(sort_column.desc())
        else:
            query = query.order_by(sort_column.asc())
        
        # Get total count
        count_query = select(func.count()).select_from(
            query.subquery()
        )
        total_count = await db.scalar(count_query)
        
        # Apply pagination
        query = query.offset(pagination.offset).limit(pagination.limit)
        
        # Execute query
        result = await db.execute(query)
        problems = result.scalars().all()
        
        # Convert to summary format
        problem_summaries = []
        for problem in problems:
            summary = ProblemSummary(
                id=problem.id,
                title=problem.title,
                status=problem.status,
                created_at=problem.created_at,
                updated_at=problem.updated_at,
                urgency=problem.urgency,
                progress_percentage=0  # TODO: Calculate from workflow
            )
            problem_summaries.append(summary)
        
        # Build pagination info
        total_pages = (total_count + pagination.limit - 1) // pagination.limit
        pagination_info = PaginationInfo(
            page=pagination.page,
            limit=pagination.limit,
            total_items=total_count,
            total_pages=total_pages,
            has_next=pagination.page < total_pages,
            has_previous=pagination.page > 1
        )
        
        return {
            "problems": [p.dict() for p in problem_summaries],
            "pagination": pagination_info.dict()
        }
        
    except Exception as e:
        logger.error("Failed to list problems", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve problems"
        ) from e


@router.get(
    "/{problem_id}",
    response_model=ProblemDetail,
    summary="Get problem details",
    description="Retrieve detailed information about a specific problem"
)
async def get_problem(
    problem_id: str = Depends(validate_problem_access),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_database),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_service)
) -> ProblemDetail:
    """Get detailed information about a specific problem."""
    try:
        # Get problem from database
        query = select(Problem).where(Problem.id == problem_id)
        result = await db.execute(query)
        problem = result.scalar_one_or_none()
        
        if not problem:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Problem not found"
            )
        
        # Get workflow progress
        workflow_progress = await orchestrator.get_workflow_progress(problem_id)
        
        return ProblemDetail(
            id=problem.id,
            raw_text=problem.raw_text,
            submitter=problem.submitter,
            status=problem.status,
            created_at=problem.created_at,
            updated_at=problem.updated_at,
            metadata=problem.metadata,
            progress=workflow_progress
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get problem", problem_id=problem_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve problem"
        ) from e


@router.delete(
    "/{problem_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a problem",
    description="Delete a problem and all associated data"
)
async def delete_problem(
    problem_id: str = Depends(validate_problem_access),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_database)
) -> None:
    """Delete a problem and all associated data."""
    try:
        # Get problem from database
        query = select(Problem).where(Problem.id == problem_id)
        result = await db.execute(query)
        problem = result.scalar_one_or_none()
        
        if not problem:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Problem not found"
            )
        
        # Check permissions (only owner or admin can delete)
        if (problem.created_by != current_user.id and 
            current_user.role not in ["admin", "owner"]):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to delete problem"
            )
        
        # Delete problem (cascade will handle related records)
        await db.delete(problem)
        await db.commit()
        
        logger.info(
            "Problem deleted",
            problem_id=problem_id,
            deleted_by=current_user.id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete problem", problem_id=problem_id, error=str(e))
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete problem"
        ) from e


@router.post(
    "/{problem_id}/run-step",
    response_model=Dict[str, Any],
    status_code=status.HTTP_202_ACCEPTED,
    summary="Execute analysis step",
    description="Trigger execution of a specific analysis step"
)
async def run_step(
    step_request: Dict[str, Any],
    problem_id: str = Depends(validate_problem_access),
    current_user: User = Depends(get_current_active_user),
    orchestrator: SimpleOrchestrator = Depends(get_orchestrator_service)
) -> Dict[str, Any]:
    """Execute a specific analysis step for a problem."""
    try:
        step_name = step_request.get("step")
        options = step_request.get("options", {})
        
        if not step_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Step name is required"
            )
        
        # Validate step name
        try:
            step = WorkflowStep(step_name)
        except ValueError:
            valid_steps = [s.value for s in WorkflowStep]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid step. Valid steps: {valid_steps}"
            )
        
        # Register Problem Framer agent if not already registered
        # In production, this would be done at startup
        if step == WorkflowStep.FRAME:
            problem_framer = ProblemFramer()
            orchestrator.register_agent(WorkflowStep.FRAME, problem_framer)
        
        # Execute the step
        execution_context = await orchestrator.execute_step(
            problem_id=problem_id,
            step=step,
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            input_data=options,
            force=options.get("force_refresh", False)
        )
        
        logger.info(
            "Step execution started",
            problem_id=problem_id,
            step=step.value,
            execution_id=execution_context.execution_id,
            user_id=current_user.id
        )
        
        return {
            "execution_id": execution_context.execution_id,
            "status": execution_context.state.value,
            "estimated_duration": execution_context.estimated_duration,
            "step": step.value
        }
        
    except HTTPException:
        raise
    except WorkflowError as e:
        logger.warning("Workflow error", problem_id=problem_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        ) from e
    except Exception as e:
        logger.error("Failed to execute step", problem_id=problem_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute step"
        ) from e