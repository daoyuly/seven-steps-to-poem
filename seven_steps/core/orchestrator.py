"""
Simplified orchestrator for managing workflow execution.

This module provides in-memory workflow orchestration for the
Seven Steps methodology, with state management and step coordination.
"""

import asyncio
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import structlog
from pydantic import BaseModel, Field

from .exceptions import WorkflowError
from .logging import LoggerMixin
from .schemas import ExecutionStatus


class WorkflowStep(str, Enum):
    """Enumeration of workflow steps."""
    FRAME = "frame"
    TREE = "tree"
    PRIORITIZE = "prioritize"
    PLAN = "plan"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    PRESENT = "present"


class ExecutionState(str, Enum):
    """Enumeration of execution states."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepDependencies:
    """Define step dependencies and execution order."""
    
    DEPENDENCIES = {
        WorkflowStep.FRAME: set(),
        WorkflowStep.TREE: {WorkflowStep.FRAME},
        WorkflowStep.PRIORITIZE: {WorkflowStep.TREE},
        WorkflowStep.PLAN: {WorkflowStep.PRIORITIZE},
        WorkflowStep.ANALYZE: {WorkflowStep.PLAN},
        WorkflowStep.SYNTHESIZE: {WorkflowStep.ANALYZE},
        WorkflowStep.PRESENT: {WorkflowStep.SYNTHESIZE},
    }
    
    @classmethod
    def get_dependencies(cls, step: WorkflowStep) -> Set[WorkflowStep]:
        """Get dependencies for a given step."""
        return cls.DEPENDENCIES.get(step, set())
    
    @classmethod
    def can_execute(cls, step: WorkflowStep, completed_steps: Set[WorkflowStep]) -> bool:
        """Check if a step can be executed given completed steps."""
        dependencies = cls.get_dependencies(step)
        return dependencies.issubset(completed_steps)
    
    @classmethod
    def get_next_steps(cls, completed_steps: Set[WorkflowStep]) -> List[WorkflowStep]:
        """Get list of steps that can be executed next."""
        next_steps = []
        for step in WorkflowStep:
            if step not in completed_steps and cls.can_execute(step, completed_steps):
                next_steps.append(step)
        return next_steps


class ExecutionContext(BaseModel):
    """Context for workflow execution."""
    
    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    problem_id: str
    user_id: str
    organization_id: str
    step: WorkflowStep
    state: ExecutionState = ExecutionState.QUEUED
    progress_percentage: int = Field(default=0, ge=0, le=100)
    current_activity: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    estimated_duration: Optional[int] = None  # seconds
    actual_duration: Optional[int] = None
    error_details: Optional[Dict[str, Any]] = None
    result_data: Optional[Dict[str, Any]] = None
    
    class Config:
        use_enum_values = True


class WorkflowState(BaseModel):
    """Complete workflow state for a problem."""
    
    problem_id: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    completed_steps: Set[WorkflowStep] = Field(default_factory=set)
    failed_steps: Set[WorkflowStep] = Field(default_factory=set)
    current_executions: Dict[str, ExecutionContext] = Field(default_factory=dict)
    step_results: Dict[WorkflowStep, Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class SimpleOrchestrator(LoggerMixin):
    """
    Simplified in-memory workflow orchestrator.
    
    This orchestrator manages the execution of the Seven Steps methodology
    without requiring external workflow engines like Temporal.
    """
    
    def __init__(self):
        """Initialize the orchestrator."""
        self._workflows: Dict[str, WorkflowState] = {}
        self._agent_registry: Dict[WorkflowStep, Any] = {}
        self._lock = asyncio.Lock()
    
    def register_agent(self, step: WorkflowStep, agent: Any) -> None:
        """Register an agent for a specific workflow step."""
        self._agent_registry[step] = agent
        self.logger.info(f"Registered agent for step {step.value}", agent=agent.agent_name)
    
    async def create_workflow(
        self,
        problem_id: str,
        user_id: str,
        organization_id: str
    ) -> WorkflowState:
        """Create a new workflow for a problem."""
        async with self._lock:
            if problem_id in self._workflows:
                raise WorkflowError(f"Workflow already exists for problem {problem_id}")
            
            workflow_state = WorkflowState(problem_id=problem_id)
            self._workflows[problem_id] = workflow_state
            
            self.logger.info(
                "Created workflow",
                problem_id=problem_id,
                user_id=user_id,
                organization_id=organization_id
            )
            
            return workflow_state
    
    async def get_workflow(self, problem_id: str) -> Optional[WorkflowState]:
        """Get workflow state for a problem."""
        return self._workflows.get(problem_id)
    
    async def execute_step(
        self,
        problem_id: str,
        step: WorkflowStep,
        user_id: str,
        organization_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> ExecutionContext:
        """
        Execute a specific workflow step.
        
        Args:
            problem_id: Problem identifier
            step: Step to execute
            user_id: User executing the step
            organization_id: Organization identifier
            input_data: Optional input data for the step
            force: Force execution even if dependencies not met
            
        Returns:
            ExecutionContext for the running step
            
        Raises:
            WorkflowError: If step cannot be executed
        """
        async with self._lock:
            # Get or create workflow
            workflow = self._workflows.get(problem_id)
            if not workflow:
                workflow = await self.create_workflow(problem_id, user_id, organization_id)
            
            # Check dependencies unless forced
            if not force and not StepDependencies.can_execute(step, workflow.completed_steps):
                missing_deps = StepDependencies.get_dependencies(step) - workflow.completed_steps
                raise WorkflowError(
                    f"Cannot execute step {step.value}, missing dependencies: {[s.value for s in missing_deps]}"
                )
            
            # Check if agent is registered
            if step not in self._agent_registry:
                raise WorkflowError(f"No agent registered for step {step.value}")
            
            # Create execution context
            execution_context = ExecutionContext(
                problem_id=problem_id,
                step=step,
                state=ExecutionState.QUEUED,
                estimated_duration=self._estimate_duration(step)
            )
            
            # Add to current executions
            workflow.current_executions[execution_context.execution_id] = execution_context
            workflow.updated_at = time.time()
            
            self.logger.info(
                "Step execution queued",
                problem_id=problem_id,
                step=step.value,
                execution_id=execution_context.execution_id
            )
        
        # Start execution asynchronously
        asyncio.create_task(self._execute_step_async(execution_context, input_data))
        
        return execution_context
    
    async def _execute_step_async(
        self,
        execution_context: ExecutionContext,
        input_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Execute a step asynchronously."""
        try:
            # Update state to running
            async with self._lock:
                execution_context.state = ExecutionState.RUNNING
                execution_context.started_at = time.time()
                execution_context.current_activity = "Initializing"
                
                workflow = self._workflows[execution_context.problem_id]
                workflow.updated_at = time.time()
            
            self.logger.info(
                "Step execution started",
                problem_id=execution_context.problem_id,
                step=execution_context.step.value,
                execution_id=execution_context.execution_id
            )
            
            # Get the agent for this step
            agent = self._agent_registry[execution_context.step]
            
            # Prepare input data
            agent_input = self._prepare_agent_input(execution_context, input_data)
            
            # Update progress
            await self._update_progress(execution_context.execution_id, 10, "Preparing input data")
            
            # Execute the agent
            await self._update_progress(execution_context.execution_id, 30, "Executing agent")
            result = await agent.execute(agent_input)
            
            # Process result
            await self._update_progress(execution_context.execution_id, 90, "Processing results")
            
            if result.success:
                await self._complete_step(execution_context, result.data or {})
            else:
                await self._fail_step(execution_context, result.error_message or "Unknown error")
            
        except Exception as e:
            self.logger.error(
                "Step execution failed",
                problem_id=execution_context.problem_id,
                step=execution_context.step.value,
                execution_id=execution_context.execution_id,
                error=str(e)
            )
            await self._fail_step(execution_context, str(e))
    
    async def _complete_step(
        self,
        execution_context: ExecutionContext,
        result_data: Dict[str, Any]
    ) -> None:
        """Mark a step as completed."""
        async with self._lock:
            execution_context.state = ExecutionState.COMPLETED
            execution_context.completed_at = time.time()
            execution_context.actual_duration = int(
                execution_context.completed_at - (execution_context.started_at or 0)
            )
            execution_context.progress_percentage = 100
            execution_context.current_activity = "Completed"
            execution_context.result_data = result_data
            
            # Update workflow state
            workflow = self._workflows[execution_context.problem_id]
            workflow.completed_steps.add(execution_context.step)
            workflow.step_results[execution_context.step] = result_data
            workflow.updated_at = time.time()
            
            # Remove from failed steps if it was there
            workflow.failed_steps.discard(execution_context.step)
        
        self.logger.info(
            "Step execution completed",
            problem_id=execution_context.problem_id,
            step=execution_context.step.value,
            execution_id=execution_context.execution_id,
            duration=execution_context.actual_duration
        )
    
    async def _fail_step(self, execution_context: ExecutionContext, error_message: str) -> None:
        """Mark a step as failed."""
        async with self._lock:
            execution_context.state = ExecutionState.FAILED
            execution_context.completed_at = time.time()
            execution_context.actual_duration = int(
                execution_context.completed_at - (execution_context.started_at or 0)
            )
            execution_context.error_details = {"error": error_message}
            
            # Update workflow state
            workflow = self._workflows[execution_context.problem_id]
            workflow.failed_steps.add(execution_context.step)
            workflow.updated_at = time.time()
        
        self.logger.error(
            "Step execution failed",
            problem_id=execution_context.problem_id,
            step=execution_context.step.value,
            execution_id=execution_context.execution_id,
            error=error_message,
            duration=execution_context.actual_duration
        )
    
    async def _update_progress(
        self,
        execution_id: str,
        progress: int,
        activity: str
    ) -> None:
        """Update execution progress."""
        async with self._lock:
            for workflow in self._workflows.values():
                if execution_id in workflow.current_executions:
                    execution = workflow.current_executions[execution_id]
                    execution.progress_percentage = progress
                    execution.current_activity = activity
                    workflow.updated_at = time.time()
                    break
    
    def _prepare_agent_input(
        self,
        execution_context: ExecutionContext,
        input_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Prepare input data for agent execution."""
        base_input = {
            "context": {
                "user_id": execution_context.problem_id,  # Will be replaced with actual user_id
                "organization_id": execution_context.problem_id,  # Will be replaced with actual org_id
                "problem_id": execution_context.problem_id,
                "correlation_id": execution_context.execution_id,
            }
        }
        
        # Add step-specific input data
        if input_data:
            base_input.update(input_data)
        
        # Add results from previous steps
        workflow = self._workflows[execution_context.problem_id]
        if workflow.step_results:
            base_input["previous_results"] = workflow.step_results
        
        return base_input
    
    def _estimate_duration(self, step: WorkflowStep) -> int:
        """Estimate duration for a step in seconds."""
        duration_estimates = {
            WorkflowStep.FRAME: 120,      # 2 minutes
            WorkflowStep.TREE: 180,       # 3 minutes
            WorkflowStep.PRIORITIZE: 90,  # 1.5 minutes
            WorkflowStep.PLAN: 150,       # 2.5 minutes
            WorkflowStep.ANALYZE: 300,    # 5 minutes
            WorkflowStep.SYNTHESIZE: 240, # 4 minutes
            WorkflowStep.PRESENT: 180,    # 3 minutes
        }
        return duration_estimates.get(step, 180)
    
    async def get_execution_status(
        self,
        problem_id: str,
        execution_id: str
    ) -> Optional[ExecutionStatus]:
        """Get status of a specific execution."""
        workflow = self._workflows.get(problem_id)
        if not workflow:
            return None
        
        execution = workflow.current_executions.get(execution_id)
        if not execution:
            return None
        
        estimated_remaining = None
        if execution.state == ExecutionState.RUNNING and execution.started_at:
            elapsed = time.time() - execution.started_at
            if execution.estimated_duration:
                estimated_remaining = max(0, execution.estimated_duration - int(elapsed))
        
        return ExecutionStatus(
            execution_id=execution_id,
            status=execution.state.value,
            progress_percentage=execution.progress_percentage,
            current_activity=execution.current_activity,
            started_at=execution.started_at,
            completed_at=execution.completed_at,
            estimated_remaining=estimated_remaining,
            error_details=execution.error_details
        )
    
    async def cancel_execution(self, problem_id: str, execution_id: str) -> bool:
        """Cancel a running execution."""
        async with self._lock:
            workflow = self._workflows.get(problem_id)
            if not workflow:
                return False
            
            execution = workflow.current_executions.get(execution_id)
            if not execution or execution.state != ExecutionState.RUNNING:
                return False
            
            execution.state = ExecutionState.CANCELLED
            execution.completed_at = time.time()
            workflow.updated_at = time.time()
            
            self.logger.info(
                "Execution cancelled",
                problem_id=problem_id,
                execution_id=execution_id
            )
            
            return True
    
    async def get_workflow_progress(self, problem_id: str) -> Dict[str, Any]:
        """Get overall workflow progress for a problem."""
        workflow = self._workflows.get(problem_id)
        if not workflow:
            return {}
        
        total_steps = len(WorkflowStep)
        completed_steps = len(workflow.completed_steps)
        failed_steps = len(workflow.failed_steps)
        
        # Calculate overall progress percentage
        progress_percentage = int((completed_steps / total_steps) * 100)
        
        # Get current step information
        current_step = None
        for execution in workflow.current_executions.values():
            if execution.state == ExecutionState.RUNNING:
                current_step = execution.step.value
                break
        
        return {
            "problem_id": problem_id,
            "progress_percentage": progress_percentage,
            "completed_steps": [step.value for step in workflow.completed_steps],
            "failed_steps": [step.value for step in workflow.failed_steps],
            "current_step": current_step,
            "total_steps": total_steps,
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at,
        }


# Global orchestrator instance
_orchestrator: Optional[SimpleOrchestrator] = None


def get_orchestrator() -> SimpleOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SimpleOrchestrator()
    return _orchestrator