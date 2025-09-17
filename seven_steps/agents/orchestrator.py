"""
Simple Agent Orchestrator implementation.

This orchestrator coordinates the execution of the seven-step agent workflow
with minimal complexity and maximum flexibility for task delegation.
"""

import asyncio
from typing import Any, Dict, List, Optional
from enum import Enum

from .base import AgentContext
from . import register_all_agents, AGENT_STEP_MAPPING


class ExecutionMode(str, Enum):
    """Execution modes for the orchestrator."""
    SEQUENTIAL = "sequential"  # Execute steps one by one
    PARALLEL = "parallel"     # Execute independent steps in parallel
    SELECTIVE = "selective"   # Execute only specified steps


class AgentOrchestrator:
    """
    Simple agent orchestrator for the seven-step methodology.
    
    This orchestrator focuses on core functionality:
    - Agent coordination and task delegation
    - Step dependency management
    - Result passing between agents
    - Sub-agent task distribution
    """
    
    def __init__(self):
        """Initialize the orchestrator with all agents."""
        self.agents = register_all_agents()
        self.step_results = {}
    
    async def execute_full_workflow(
        self,
        problem_id: str,
        user_id: str,
        organization_id: str,
        initial_input: Dict[str, Any],
        execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    ) -> Dict[str, Any]:
        """
        Execute the complete seven-step workflow.
        
        Args:
            problem_id: Unique problem identifier
            user_id: User executing the workflow
            organization_id: Organization identifier
            initial_input: Initial problem data
            execution_mode: How to execute the workflow
            
        Returns:
            Dict containing results from all steps
        """
        context = AgentContext(
            user_id=user_id,
            organization_id=organization_id,
            problem_id=problem_id
        )
        
        # Define step order and dependencies
        steps = [
            ("frame", "problem_framer", initial_input),
            ("tree", "issue_tree", lambda: {"problem_frame": self.step_results["frame"]["data"]}),
            ("prioritize", "prioritization", lambda: {"issue_tree": self.step_results["tree"]["data"]}),
            ("plan", "planner", lambda: {"priority_matrix": self.step_results["prioritize"]["data"]}),
            ("analyze", "data_analysis", lambda: {"analysis_plans": self.step_results["plan"]["data"]["analysis_plans"]}),
            ("synthesize", "synthesizer", lambda: {"analysis_results": self.step_results["analyze"]["data"]["analysis_results"]}),
            ("present", "presentation", lambda: {"synthesis_result": self.step_results["synthesize"]["data"]["synthesis_result"]})
        ]
        
        if execution_mode == ExecutionMode.SEQUENTIAL:
            return await self._execute_sequential(steps, context)
        elif execution_mode == ExecutionMode.PARALLEL:
            return await self._execute_parallel(steps, context)
        else:
            raise ValueError(f"Unsupported execution mode: {execution_mode}")
    
    async def execute_single_step(
        self,
        step_name: str,
        problem_id: str,
        user_id: str,
        organization_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a single step independently.
        
        Args:
            step_name: Name of the step to execute
            problem_id: Problem identifier
            user_id: User identifier
            organization_id: Organization identifier
            input_data: Input data for the step
            
        Returns:
            Result from the executed step
        """
        if step_name not in AGENT_STEP_MAPPING:
            raise ValueError(f"Unknown step: {step_name}")
        
        agent_name = AGENT_STEP_MAPPING[step_name]
        agent = self.agents[agent_name]
        
        context = AgentContext(
            user_id=user_id,
            organization_id=organization_id,
            problem_id=problem_id
        )
        
        # Prepare input with context
        agent_input = {
            "context": context.dict(),
            **input_data
        }
        
        # Execute the agent
        result = await agent.execute(agent_input)
        
        return {
            "step": step_name,
            "agent": agent_name,
            "success": result.success,
            "data": result.data,
            "error": result.error_message,
            "confidence": result.confidence_score,
            "execution_time": result.execution_time
        }
    
    async def delegate_to_subagent(
        self,
        parent_agent: str,
        subtask: str,
        input_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Delegate a subtask to a specialized sub-agent.
        
        This enables agents to break down complex tasks and delegate
        specific work to other agents or tools.
        
        Args:
            parent_agent: Name of the requesting agent
            subtask: Description of the subtask
            input_data: Data for the subtask
            context: Execution context
            
        Returns:
            Result from the sub-agent
        """
        # Route subtask to appropriate agent based on task type
        subtask_mapping = {
            "sql_query": "data_analysis",
            "data_visualization": "presentation", 
            "statistical_analysis": "data_analysis",
            "risk_assessment": "synthesizer",
            "stakeholder_analysis": "problem_framer",
            "hypothesis_testing": "data_analysis",
            "recommendation_formatting": "presentation"
        }
        
        # Determine which agent should handle the subtask
        target_agent_name = subtask_mapping.get(subtask.lower())
        
        if not target_agent_name:
            # Default to the data analysis agent for analytical tasks
            target_agent_name = "data_analysis"
        
        target_agent = self.agents[target_agent_name]
        
        # Prepare input for sub-agent
        subagent_input = {
            "context": context.dict(),
            "subtask": subtask,
            "parent_agent": parent_agent,
            **input_data
        }
        
        # Execute sub-agent
        result = await target_agent.execute(subagent_input)
        
        return {
            "subtask": subtask,
            "parent_agent": parent_agent,
            "executing_agent": target_agent_name,
            "success": result.success,
            "data": result.data,
            "error": result.error_message
        }
    
    async def _execute_sequential(
        self, 
        steps: List[tuple], 
        context: AgentContext
    ) -> Dict[str, Any]:
        """Execute steps sequentially with dependency management."""
        results = {}
        
        for step_name, agent_name, input_spec in steps:
            try:
                # Prepare input data
                if callable(input_spec):
                    input_data = input_spec()
                else:
                    input_data = input_spec
                
                # Add context
                agent_input = {
                    "context": context.dict(),
                    **input_data
                }
                
                # Execute agent
                agent = self.agents[agent_name]
                result = await agent.execute(agent_input)
                
                # Store result
                step_result = {
                    "step": step_name,
                    "agent": agent_name,
                    "success": result.success,
                    "data": result.data,
                    "error": result.error_message,
                    "confidence": result.confidence_score,
                    "execution_time": result.execution_time
                }
                
                results[step_name] = step_result
                self.step_results[step_name] = step_result
                
                # Stop if step failed
                if not result.success:
                    results["workflow_status"] = "failed"
                    results["failed_step"] = step_name
                    break
                    
            except Exception as e:
                results[step_name] = {
                    "step": step_name,
                    "agent": agent_name,
                    "success": False,
                    "error": str(e),
                    "execution_time": 0
                }
                results["workflow_status"] = "failed"
                results["failed_step"] = step_name
                break
        
        if "workflow_status" not in results:
            results["workflow_status"] = "completed"
        
        return results
    
    async def _execute_parallel(
        self, 
        steps: List[tuple], 
        context: AgentContext
    ) -> Dict[str, Any]:
        """Execute independent steps in parallel where possible."""
        # For now, implement as sequential since most steps have dependencies
        # In a full implementation, this would analyze dependencies and execute
        # independent steps concurrently
        return await self._execute_sequential(steps, context)
    
    def get_step_dependencies(self) -> Dict[str, List[str]]:
        """Get dependency mapping for all steps."""
        return {
            "frame": [],
            "tree": ["frame"],
            "prioritize": ["tree"],
            "plan": ["prioritize"],
            "analyze": ["plan"],
            "synthesize": ["analyze"],
            "present": ["synthesize"]
        }
    
    def get_available_agents(self) -> List[str]:
        """Get list of all available agents."""
        return list(self.agents.keys())
    
    def get_agent_capabilities(self, agent_name: str) -> Dict[str, Any]:
        """Get capabilities and metadata for a specific agent."""
        if agent_name not in self.agents:
            return {}
        
        agent = self.agents[agent_name]
        return {
            "agent_name": agent.agent_name,
            "input_model": agent.input_model.__name__,
            "output_model": agent.output_model.__name__,
            "description": agent.__class__.__doc__ or "No description available",
            "can_delegate": True,  # All agents can delegate to sub-agents
            "supported_subtasks": self._get_agent_subtasks(agent_name)
        }
    
    def _get_agent_subtasks(self, agent_name: str) -> List[str]:
        """Get supported subtasks for an agent."""
        subtask_map = {
            "problem_framer": ["stakeholder_analysis", "requirement_gathering"],
            "issue_tree": ["hypothesis_generation", "mece_validation"],
            "prioritization": ["impact_assessment", "feasibility_analysis"],
            "planner": ["resource_planning", "timeline_estimation"],
            "data_analysis": ["sql_query", "statistical_analysis", "hypothesis_testing"],
            "synthesizer": ["evidence_synthesis", "risk_assessment"],
            "presentation": ["data_visualization", "recommendation_formatting"]
        }
        
        return subtask_map.get(agent_name, [])


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator() -> AgentOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator