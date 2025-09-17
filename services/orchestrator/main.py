"""
Orchestrator Microservice - Main Application

This service coordinates the execution of all seven steps in the
problem-solving methodology through inter-service communication.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the parent directory to Python path to import seven_steps modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from seven_steps.core.config import get_settings
from seven_steps.core.logging import get_logger, setup_logging
from seven_steps.core.orchestrator import (
    SimpleOrchestrator,
    WorkflowStep,
    ExecutionState,
    WorkflowState
)

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global orchestrator instance
_orchestrator: SimpleOrchestrator = None


class ServiceRegistry:
    """Registry for microservice endpoints."""
    
    def __init__(self):
        self.services = {
            WorkflowStep.FRAME: os.getenv("PROBLEM_FRAMER_URL", "http://localhost:8001"),
            WorkflowStep.TREE: os.getenv("ISSUE_TREE_URL", "http://localhost:8002"),
            WorkflowStep.PRIORITIZE: os.getenv("PRIORITIZATION_URL", "http://localhost:8003"),
            WorkflowStep.PLAN: os.getenv("PLANNER_URL", "http://localhost:8004"),
            WorkflowStep.ANALYZE: os.getenv("ANALYSIS_URL", "http://localhost:8005"),
            WorkflowStep.SYNTHESIZE: os.getenv("SYNTHESIZER_URL", "http://localhost:8006"),
            WorkflowStep.PRESENT: os.getenv("PRESENTATION_URL", "http://localhost:8007"),
        }
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def call_service(
        self, 
        step: WorkflowStep, 
        endpoint: str, 
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a microservice endpoint."""
        service_url = self.services.get(step)
        if not service_url:
            raise ValueError(f"No service registered for step {step}")
        
        url = f"{service_url}{endpoint}"
        
        try:
            logger.info(f"Calling service", step=step.value, url=url)
            
            response = await self.client.post(url, json=data)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Service call successful", step=step.value, success=result.get("success"))
            
            return result
            
        except httpx.HTTPError as e:
            logger.error(f"Service call failed", step=step.value, url=url, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Service {step.value} unavailable"
            ) from e
    
    async def health_check_service(self, step: WorkflowStep) -> bool:
        """Check if a service is healthy."""
        try:
            service_url = self.services.get(step)
            if not service_url:
                return False
            
            response = await self.client.get(f"{service_url}/health", timeout=5.0)
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _orchestrator
    
    # Startup
    logger.info("Starting Orchestrator Service")
    
    try:
        _orchestrator = SimpleOrchestrator()
        
        # Register service-based agents
        service_registry = ServiceRegistry()
        app.state.service_registry = service_registry
        
        # Register a service agent adapter for each step
        for step in WorkflowStep:
            service_agent = ServiceAgent(step, service_registry)
            _orchestrator.register_agent(step, service_agent)
        
        logger.info("Orchestrator initialized with service agents")
        
    except Exception as e:
        logger.error("Failed to initialize orchestrator", error=str(e))
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down Orchestrator Service")
    if hasattr(app.state, 'service_registry'):
        await app.state.service_registry.close()


class ServiceAgent:
    """
    Adapter to make microservices work with the orchestrator.
    
    This class wraps microservice calls to match the agent interface
    expected by the orchestrator.
    """
    
    def __init__(self, step: WorkflowStep, service_registry: ServiceRegistry):
        self.step = step
        self.service_registry = service_registry
        self.agent_name = f"{step.value}-service"
    
    async def execute(self, raw_input: Dict[str, Any]) -> Any:
        """Execute the service call."""
        try:
            # Map step to service endpoint
            endpoint_map = {
                WorkflowStep.FRAME: "/frame",
                WorkflowStep.TREE: "/generate",
                WorkflowStep.PRIORITIZE: "/prioritize",
                WorkflowStep.PLAN: "/plan",
                WorkflowStep.ANALYZE: "/analyze",
                WorkflowStep.SYNTHESIZE: "/synthesize",
                WorkflowStep.PRESENT: "/present",
            }
            
            endpoint = endpoint_map.get(self.step, "/process")
            
            # Call the service
            result = await self.service_registry.call_service(
                self.step, 
                endpoint, 
                raw_input
            )
            
            # Return in agent format
            return type('Result', (), {
                'success': result.get('success', False),
                'data': result.get('data'),
                'error_message': result.get('error_message'),
                'confidence_score': result.get('confidence_score'),
                'execution_time': result.get('execution_time')
            })()
            
        except Exception as e:
            logger.error(f"Service agent execution failed", step=self.step.value, error=str(e))
            return type('Result', (), {
                'success': False,
                'error_message': str(e),
                'data': None,
                'confidence_score': None,
                'execution_time': None
            })()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Orchestrator Service",
        description="Workflow orchestration service for the Seven Steps methodology",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["http://localhost:8000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# Request/Response models
class CreateWorkflowRequest(BaseModel):
    """Request to create a new workflow."""
    
    problem_id: str
    user_id: str
    organization_id: str


class ExecuteStepRequest(BaseModel):
    """Request to execute a workflow step."""
    
    problem_id: str
    step: str
    user_id: str
    organization_id: str
    input_data: Dict[str, Any] = {}
    force: bool = False


class WorkflowStatusResponse(BaseModel):
    """Response with workflow status."""
    
    problem_id: str
    progress_percentage: int
    completed_steps: List[str]
    failed_steps: List[str]
    current_step: Optional[str]


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check dependent services
    service_health = {}
    
    if hasattr(app.state, 'service_registry'):
        for step in [WorkflowStep.FRAME, WorkflowStep.TREE]:  # Check key services
            is_healthy = await app.state.service_registry.health_check_service(step)
            service_health[step.value] = "up" if is_healthy else "down"
    
    return {
        "status": "healthy",
        "service": "orchestrator",
        "version": "1.0.0",
        "dependent_services": service_health
    }


@app.post("/workflows")
async def create_workflow(request: CreateWorkflowRequest):
    """Create a new workflow for a problem."""
    try:
        workflow_state = await _orchestrator.create_workflow(
            problem_id=request.problem_id,
            user_id=request.user_id,
            organization_id=request.organization_id
        )
        
        return {
            "success": True,
            "problem_id": request.problem_id,
            "created_at": workflow_state.created_at,
            "message": "Workflow created successfully"
        }
        
    except Exception as e:
        logger.error("Failed to create workflow", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )


@app.post("/workflows/execute-step")
async def execute_step(request: ExecuteStepRequest):
    """Execute a specific workflow step."""
    try:
        # Validate step
        try:
            step = WorkflowStep(request.step)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid step: {request.step}"
            )
        
        # Execute the step
        execution_context = await _orchestrator.execute_step(
            problem_id=request.problem_id,
            step=step,
            user_id=request.user_id,
            organization_id=request.organization_id,
            input_data=request.input_data,
            force=request.force
        )
        
        return {
            "success": True,
            "execution_id": execution_context.execution_id,
            "status": execution_context.state.value,
            "estimated_duration": execution_context.estimated_duration
        }
        
    except Exception as e:
        logger.error("Failed to execute step", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute step: {str(e)}"
        )


@app.get("/workflows/{problem_id}/status", response_model=WorkflowStatusResponse)
async def get_workflow_status(problem_id: str) -> WorkflowStatusResponse:
    """Get workflow status for a problem."""
    try:
        progress = await _orchestrator.get_workflow_progress(problem_id)
        
        if not progress:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow not found"
            )
        
        return WorkflowStatusResponse(
            problem_id=problem_id,
            progress_percentage=progress["progress_percentage"],
            completed_steps=progress["completed_steps"],
            failed_steps=progress["failed_steps"],
            current_step=progress["current_step"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}"
        )


@app.get("/workflows/{problem_id}/executions/{execution_id}")
async def get_execution_status(problem_id: str, execution_id: str):
    """Get execution status for a specific execution."""
    try:
        status = await _orchestrator.get_execution_status(problem_id, execution_id)
        
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found"
            )
        
        return status.dict()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get execution status", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get execution status: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get orchestrator metrics."""
    return {
        "service": "orchestrator",
        "status": "running",
        "active_workflows": len(_orchestrator._workflows) if _orchestrator else 0,
        "registered_agents": len(_orchestrator._agent_registry) if _orchestrator else 0
    }


def main():
    """Main entry point."""
    settings = get_settings()
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_config=None,
    )


if __name__ == "__main__":
    main()