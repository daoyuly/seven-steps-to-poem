"""
Problem Framer Microservice - Main Application

This service handles problem structuring and framing using AI.
It's designed to run as an independent microservice with its own API.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the parent directory to Python path to import seven_steps modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from seven_steps.agents.problem_framer import ProblemFramer, ProblemFramerInput
from seven_steps.core.config import get_settings
from seven_steps.core.logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global agent instance
_problem_framer: ProblemFramer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _problem_framer
    
    # Startup
    logger.info("Starting Problem Framer Service")
    try:
        _problem_framer = ProblemFramer()
        logger.info("Problem Framer agent initialized")
    except Exception as e:
        logger.error("Failed to initialize Problem Framer agent", error=str(e))
        raise e
    
    yield
    
    # Shutdown
    logger.info("Shutting down Problem Framer Service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Problem Framer Service",
        description="AI-powered business problem framing and structuring service",
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
class FrameProblemRequest(BaseModel):
    """Request model for problem framing."""
    
    raw_text: str
    submitter: str
    metadata: Dict[str, Any] = {}
    previous_clarifications: Dict[str, str] = {}
    context: Dict[str, Any]


class FrameProblemResponse(BaseModel):
    """Response model for problem framing."""
    
    success: bool
    data: Dict[str, Any] = None
    error_message: str = None
    confidence_score: float = None
    execution_time: float = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "problem-framer",
        "version": "1.0.0"
    }


@app.post("/frame", response_model=FrameProblemResponse)
async def frame_problem(request: FrameProblemRequest) -> FrameProblemResponse:
    """
    Frame a business problem using AI analysis.
    
    This endpoint takes a raw problem description and returns a structured
    problem frame with goals, scope, KPIs, and clarifying questions.
    """
    try:
        logger.info(
            "Processing problem framing request",
            submitter=request.submitter,
            text_length=len(request.raw_text)
        )
        
        # Create agent input
        agent_input = {
            "raw_text": request.raw_text,
            "submitter": request.submitter,
            "metadata": request.metadata,
            "previous_clarifications": request.previous_clarifications,
            "context": request.context
        }
        
        # Execute the agent
        result = await _problem_framer.execute(agent_input)
        
        # Convert result to response format
        response_data = None
        if result.success and result.data:
            response_data = result.data
        
        response = FrameProblemResponse(
            success=result.success,
            data=response_data,
            error_message=result.error_message,
            confidence_score=result.confidence_score,
            execution_time=result.execution_time
        )
        
        logger.info(
            "Problem framing completed",
            success=result.success,
            execution_time=result.execution_time,
            confidence_score=result.confidence_score
        )
        
        return response
        
    except Exception as e:
        logger.error("Problem framing failed", error=str(e))
        return FrameProblemResponse(
            success=False,
            error_message=f"Internal error: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    return {
        "service": "problem-framer",
        "status": "running",
        "requests_processed": 0,  # In production, this would be real metrics
        "average_processing_time": 0,
        "success_rate": 1.0
    }


def main():
    """Main entry point."""
    settings = get_settings()
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8001))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_config=None,
    )


if __name__ == "__main__":
    main()