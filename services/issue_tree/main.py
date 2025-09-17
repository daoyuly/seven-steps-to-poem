"""
Issue Tree Microservice - Main Application

This service handles MECE problem decomposition and issue tree generation.
It's designed to run as an independent microservice with its own API.
"""

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List

import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add the parent directory to Python path to import seven_steps modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from seven_steps.core.config import get_settings
from seven_steps.core.logging import get_logger, setup_logging
from seven_steps.core.schemas import ProblemFrame, TreeNode, IssueTree

# Setup logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Issue Tree Service")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Issue Tree Service")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Issue Tree Service",
        description="MECE problem decomposition and issue tree generation service",
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
class GenerateTreeRequest(BaseModel):
    """Request model for issue tree generation."""
    
    problem_frame: Dict[str, Any]
    context: Dict[str, Any]
    max_depth: int = 4
    max_branches: int = 5


class GenerateTreeResponse(BaseModel):
    """Response model for issue tree generation."""
    
    success: bool
    data: Dict[str, Any] = None
    error_message: str = None
    execution_time: float = None


class IssueTreeGenerator:
    """
    Issue Tree Generator using AI.
    
    This class handles the generation of MECE (Mutually Exclusive, 
    Collectively Exhaustive) issue trees from problem frames.
    """
    
    def __init__(self):
        self.settings = get_settings()
        # In production, this would initialize LLM client
        logger.info("Issue Tree Generator initialized")
    
    async def generate_tree(
        self, 
        problem_frame: Dict[str, Any], 
        context: Dict[str, Any],
        max_depth: int = 4,
        max_branches: int = 5
    ) -> IssueTree:
        """
        Generate an issue tree from a problem frame.
        
        Args:
            problem_frame: Structured problem frame
            context: Additional context information
            max_depth: Maximum tree depth
            max_branches: Maximum branches per node
            
        Returns:
            IssueTree: Generated issue tree structure
        """
        logger.info("Generating issue tree", goal=problem_frame.get("goal"))
        
        # For MVP, create a mock tree structure
        # In production, this would use LLM to generate the tree
        root_node = await self._create_mock_tree(problem_frame)
        
        # Create visualization data for D3.js
        viz_data = self._create_visualization_data(root_node)
        
        issue_tree = IssueTree(
            root=root_node,
            visualization_data=viz_data
        )
        
        return issue_tree
    
    async def _create_mock_tree(self, problem_frame: Dict[str, Any]) -> TreeNode:
        """Create a mock tree for demonstration purposes."""
        goal = problem_frame.get("goal", "Solve business problem")
        
        # Create root node
        root = TreeNode(
            id="root",
            title=f"Root Cause Analysis: {goal}",
            hypotheses=[
                "Problem may be caused by multiple factors",
                "Root causes can be categorized systematically"
            ],
            required_data=["historical_data", "current_metrics"],
            priority_hint="high"
        )
        
        # Create level 1 branches based on common business categories
        level1_categories = [
            {
                "id": "people",
                "title": "People & Organization",
                "hypotheses": ["Team capability issues", "Process execution problems"],
                "required_data": ["hr_data", "performance_metrics"]
            },
            {
                "id": "process",
                "title": "Process & Systems",
                "hypotheses": ["Inefficient workflows", "Technology limitations"],
                "required_data": ["process_data", "system_logs"]
            },
            {
                "id": "market",
                "title": "Market & External",
                "hypotheses": ["Market changes", "Competitive pressure"],
                "required_data": ["market_research", "competitor_analysis"]
            }
        ]
        
        for category in level1_categories:
            level1_node = TreeNode(
                id=category["id"],
                title=category["title"],
                hypotheses=category["hypotheses"],
                required_data=category["required_data"],
                priority_hint="medium"
            )
            
            # Add level 2 nodes
            for i in range(2):
                level2_node = TreeNode(
                    id=f"{category['id']}_sub_{i+1}",
                    title=f"{category['title']} - Factor {i+1}",
                    hypotheses=[f"Specific hypothesis {i+1}"],
                    required_data=[f"specific_data_{i+1}"],
                    priority_hint="low"
                )
                level1_node.children.append(level2_node)
            
            root.children.append(level1_node)
        
        return root
    
    def _create_visualization_data(self, root: TreeNode) -> Dict[str, Any]:
        """Create D3.js compatible visualization data."""
        def node_to_dict(node: TreeNode, parent_id: str = None) -> Dict[str, Any]:
            return {
                "id": node.id,
                "title": node.title,
                "parent": parent_id,
                "hypotheses_count": len(node.hypotheses),
                "data_requirements_count": len(node.required_data),
                "priority": node.priority_hint,
                "children": [node_to_dict(child, node.id) for child in node.children]
            }
        
        return {
            "tree": node_to_dict(root),
            "layout": "tree",
            "orientation": "top-to-bottom"
        }


# Global generator instance
_tree_generator = IssueTreeGenerator()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "issue-tree",
        "version": "1.0.0"
    }


@app.post("/generate", response_model=GenerateTreeResponse)
async def generate_issue_tree(request: GenerateTreeRequest) -> GenerateTreeResponse:
    """
    Generate an issue tree from a problem frame.
    
    This endpoint takes a structured problem frame and generates a MECE
    issue tree for systematic problem analysis.
    """
    try:
        start_time = asyncio.get_event_loop().time()
        
        logger.info(
            "Processing issue tree generation request",
            goal=request.problem_frame.get("goal"),
            max_depth=request.max_depth
        )
        
        # Generate the issue tree
        issue_tree = await _tree_generator.generate_tree(
            problem_frame=request.problem_frame,
            context=request.context,
            max_depth=request.max_depth,
            max_branches=request.max_branches
        )
        
        # Calculate execution time
        execution_time = asyncio.get_event_loop().time() - start_time
        
        response = GenerateTreeResponse(
            success=True,
            data=issue_tree.dict(),
            execution_time=execution_time
        )
        
        logger.info(
            "Issue tree generation completed",
            execution_time=execution_time,
            nodes_count=len(issue_tree.root.children)
        )
        
        return response
        
    except Exception as e:
        logger.error("Issue tree generation failed", error=str(e))
        return GenerateTreeResponse(
            success=False,
            error_message=f"Internal error: {str(e)}"
        )


@app.get("/metrics")
async def get_metrics():
    """Get service metrics."""
    return {
        "service": "issue-tree",
        "status": "running",
        "trees_generated": 0,  # In production, this would be real metrics
        "average_processing_time": 0,
        "success_rate": 1.0
    }


def main():
    """Main entry point."""
    settings = get_settings()
    
    # Get port from environment or use default
    port = int(os.getenv("PORT", 8002))
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
        log_config=None,
    )


if __name__ == "__main__":
    main()