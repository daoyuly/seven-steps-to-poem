"""
Advanced Agent Orchestrator using OpenAI Agents SDK framework.

This orchestrator leverages OpenAI's multi-agent patterns, structured outputs,
and MCP tools for sophisticated seven-step workflow management.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

from openai import OpenAI
from openai.agents import Agent
from openai.agents.tools import HostedMCPTool

from .base_v2 import BaseSevenStepsAgent, AgentContext, get_agent_registry
from seven_steps.core.logging import LoggerMixin


class WorkflowMode(str, Enum):
    """Workflow execution modes."""
    SEQUENTIAL = "sequential"     # Execute steps one by one
    PARALLEL = "parallel"        # Execute independent steps in parallel  
    AGENT_AS_TOOL = "agent_as_tool"  # Agents call each other as tools
    HANDOFF = "handoff"         # Agents transfer control mid-execution


@dataclass
class WorkflowStep:
    """Represents a step in the seven-step workflow."""
    name: str
    agent_name: str
    dependencies: List[str]
    parallel_group: Optional[str] = None
    required: bool = True
    timeout_minutes: int = 10


class SevenStepsOrchestrator(Agent, LoggerMixin):
    """
    Advanced orchestrator for seven-step methodology using OpenAI Agents SDK.
    
    This orchestrator acts as a Portfolio Manager (PM) style agent that
    coordinates specialist agents using the "agent as a tool" pattern.
    """
    
    def __init__(self):
        """Initialize the orchestrator with all seven-step agents."""
        
        # Initialize OpenAI client
        from seven_steps.core.config import get_settings
        settings = get_settings()
        client = OpenAI(api_key=settings.llm.openai_api_key)
        
        # Define workflow steps
        self.workflow_steps = [
            WorkflowStep("frame", "problem_framer", []),
            WorkflowStep("tree", "issue_tree", ["frame"]),
            WorkflowStep("prioritize", "prioritization", ["tree"]),
            WorkflowStep("plan", "planner", ["prioritize"]),
            WorkflowStep("analyze", "data_analysis", ["plan"]),
            WorkflowStep("synthesize", "synthesizer", ["analyze"]),
            WorkflowStep("present", "presentation", ["synthesize"])
        ]
        
        # Get agent registry
        self.agent_registry = get_agent_registry()
        
        instructions = """You are the Portfolio Manager for the Seven Steps business analysis methodology. You coordinate specialist agents to deliver comprehensive business solutions.

RESPONSIBILITIES:
1. Orchestrate the seven-step workflow: Frame → Tree → Prioritize → Plan → Analyze → Synthesize → Present
2. Make intelligent decisions about execution flow and agent coordination
3. Handle errors gracefully and determine retry/skip logic
4. Synthesize results across multiple agents
5. Ensure quality and consistency throughout the workflow
6. Adapt execution based on intermediate results

COORDINATION PATTERNS:
- Agent as Tool: Call specialist agents for specific subtasks
- Handoff: Transfer control when appropriate
- Parallel Execution: Run independent analyses simultaneously
- Error Recovery: Handle failures and determine next steps

DECISION MAKING:
- Assess when clarification is needed before proceeding
- Determine which analyses are most critical
- Balance thoroughness with time constraints
- Escalate to human oversight when necessary

You have access to all seven specialist agents as tools. Use them strategically to deliver the best possible business analysis and recommendations."""

        # Create agent tools for each specialist agent
        agent_tools = self._create_agent_tools()
        
        # Add MCP tools for workflow management
        mcp_tools = [
            HostedMCPTool(server="workflow-tracker"),
            HostedMCPTool(server="quality-assessor"),
            HostedMCPTool(server="result-synthesizer")
        ]
        
        # Initialize as OpenAI Agent
        super().__init__(
            client=client,
            model="gpt-4o",
            name="SevenStepsOrchestrator",
            description="Portfolio Manager for Seven Steps business analysis methodology",
            instructions=instructions,
            tools=agent_tools + mcp_tools
        )
        
        # Workflow state
        self.workflow_state = {}
        self.step_results = {}
        self.execution_history = []
    
    def _create_agent_tools(self) -> List[Any]:
        """Create function tools that wrap specialist agents."""
        
        agent_tools = []
        
        # Define agent tool functions
        agent_tool_specs = [
            {
                "name": "frame_problem",
                "agent": "problem_framer",
                "description": "Structure raw business problems into actionable frameworks with goals, scope, KPIs, and stakeholder analysis",
                "parameters": {
                    "raw_text": "str - Raw problem description",
                    "submitter": "str - Problem submitter email", 
                    "metadata": "dict - Additional context metadata",
                    "previous_clarifications": "dict - Previous clarification answers"
                }
            },
            {
                "name": "create_issue_tree",
                "agent": "issue_tree",
                "description": "Decompose problems into MECE issue trees with testable hypotheses and data requirements",
                "parameters": {
                    "problem_frame": "dict - Structured problem frame from problem_framer",
                    "max_depth": "int - Maximum tree depth (default: 3)",
                    "focus_areas": "list - Specific areas to focus on"
                }
            },
            {
                "name": "prioritize_analyses",
                "agent": "prioritization", 
                "description": "Score issue tree nodes using impact × feasibility matrix and recommend execution sequence",
                "parameters": {
                    "issue_tree": "dict - Issue tree from issue_tree agent",
                    "scoring_criteria": "dict - Custom weights for scoring",
                    "resource_constraints": "dict - Available resources and constraints"
                }
            },
            {
                "name": "create_analysis_plans",
                "agent": "planner",
                "description": "Generate detailed, executable analysis plans with data requirements, methodologies, and timelines",
                "parameters": {
                    "priority_matrix": "dict - Priority matrix from prioritization agent",
                    "selected_nodes": "list - Specific nodes to plan for",
                    "planning_horizon": "str - Planning timeframe",
                    "available_resources": "dict - Available team and tools"
                }
            },
            {
                "name": "execute_data_analysis",
                "agent": "data_analysis",
                "description": "Execute analysis plans with data processing, statistical analysis, and artifact generation",
                "parameters": {
                    "analysis_plans": "list - Analysis plans from planner agent",
                    "execution_mode": "str - automated|supervised|manual",
                    "selected_plan_ids": "list - Specific plans to execute",
                    "data_access_config": "dict - Data source configuration"
                }
            },
            {
                "name": "synthesize_results",
                "agent": "synthesizer",
                "description": "Integrate analysis results using pyramid principle to generate business recommendations",
                "parameters": {
                    "analysis_results": "list - Results from data_analysis agent",
                    "synthesis_framework": "str - pyramid_principle|hypothesis_driven|impact_based",
                    "business_objectives": "dict - Original business objectives",
                    "stakeholder_priorities": "list - Stakeholder priorities"
                }
            },
            {
                "name": "create_presentation",
                "agent": "presentation",
                "description": "Generate professional presentations, reports, and multimedia content for stakeholder delivery",
                "parameters": {
                    "synthesis_result": "dict - Synthesis result from synthesizer agent",
                    "presentation_formats": "list - ppt|executive_summary|video_script|audio_script|one_pager",
                    "target_audience": "str - executives|stakeholders|technical_team|board",
                    "presentation_duration": "int - Duration in minutes",
                    "brand_guidelines": "dict - Brand colors, fonts, style preferences"
                }
            }
        ]
        
        # Create actual tool functions
        for spec in agent_tool_specs:
            tool_function = self._create_agent_tool_function(spec)
            agent_tools.append(tool_function)
        
        return agent_tools
    
    def _create_agent_tool_function(self, spec: Dict[str, Any]) -> Any:
        """Create a tool function that wraps an agent call."""
        
        async def agent_tool(**kwargs) -> Dict[str, Any]:
            """Execute the specified agent with given parameters."""
            
            agent_name = spec["agent"]
            agent = self.agent_registry.get(agent_name)
            
            if not agent:
                return {
                    "success": False,
                    "error": f"Agent '{agent_name}' not found in registry"
                }
            
            try:
                # Log the agent call
                self.logger.info(
                    "Orchestrator calling agent",
                    orchestrator="SevenStepsOrchestrator",
                    target_agent=agent_name,
                    tool_name=spec["name"],
                    parameters=list(kwargs.keys())
                )
                
                # Execute the agent
                result = await agent.execute(kwargs)
                
                # Store result in workflow state
                step_name = self._get_step_name_for_agent(agent_name)
                if step_name:
                    self.step_results[step_name] = result.data
                
                # Add to execution history
                self.execution_history.append({
                    "agent": agent_name,
                    "tool": spec["name"],
                    "success": result.success,
                    "confidence": result.confidence_score,
                    "execution_time": result.execution_time
                })
                
                return {
                    "success": result.success,
                    "data": result.data,
                    "error": result.error_message,
                    "confidence": result.confidence_score,
                    "execution_time": result.execution_time
                }
                
            except Exception as e:
                self.logger.error(
                    "Agent tool execution failed",
                    orchestrator="SevenStepsOrchestrator",
                    target_agent=agent_name,
                    tool_name=spec["name"],
                    error=str(e)
                )
                
                return {
                    "success": False,
                    "error": f"Agent execution failed: {str(e)}",
                    "agent": agent_name
                }
        
        # Set function metadata for OpenAI tool registration
        agent_tool.__name__ = spec["name"]
        agent_tool.__doc__ = spec["description"]
        
        return agent_tool
    
    def _get_step_name_for_agent(self, agent_name: str) -> Optional[str]:
        """Get workflow step name for an agent."""
        
        agent_to_step = {
            "problem_framer": "frame",
            "issue_tree": "tree", 
            "prioritization": "prioritize",
            "planner": "plan",
            "data_analysis": "analyze",
            "synthesizer": "synthesize",
            "presentation": "present"
        }
        
        return agent_to_step.get(agent_name)
    
    async def execute_workflow(
        self,
        problem_id: str,
        user_id: str,
        organization_id: str,
        initial_input: Dict[str, Any],
        workflow_mode: WorkflowMode = WorkflowMode.AGENT_AS_TOOL,
        selected_steps: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete seven-step workflow.
        
        Args:
            problem_id: Unique problem identifier
            user_id: User executing the workflow
            organization_id: Organization identifier
            initial_input: Initial problem data
            workflow_mode: How to execute the workflow
            selected_steps: Specific steps to execute (optional)
            
        Returns:
            Dict containing workflow results and metadata
        """
        
        # Initialize workflow context
        context = AgentContext(
            user_id=user_id,
            organization_id=organization_id,
            problem_id=problem_id,
            metadata={
                "workflow_mode": workflow_mode.value,
                "selected_steps": selected_steps or [],
                "orchestrator": "SevenStepsOrchestrator"
            }
        )
        
        self.workflow_state = {
            "context": context,
            "initial_input": initial_input,
            "workflow_mode": workflow_mode,
            "selected_steps": selected_steps,
            "status": "started",
            "current_step": None
        }
        
        try:
            self.logger.info(
                "Starting workflow execution",
                orchestrator="SevenStepsOrchestrator",
                problem_id=problem_id,
                workflow_mode=workflow_mode.value
            )
            
            if workflow_mode == WorkflowMode.AGENT_AS_TOOL:
                result = await self._execute_agent_as_tool_mode(context, initial_input)
            elif workflow_mode == WorkflowMode.SEQUENTIAL:
                result = await self._execute_sequential_mode(context, initial_input, selected_steps)
            elif workflow_mode == WorkflowMode.PARALLEL:
                result = await self._execute_parallel_mode(context, initial_input, selected_steps)
            elif workflow_mode == WorkflowMode.HANDOFF:
                result = await self._execute_handoff_mode(context, initial_input)
            else:
                raise ValueError(f"Unsupported workflow mode: {workflow_mode}")
            
            self.workflow_state["status"] = "completed"
            
            return {
                "workflow_status": "completed",
                "problem_id": problem_id,
                "results": result,
                "step_results": self.step_results,
                "execution_history": self.execution_history,
                "workflow_metadata": {
                    "mode": workflow_mode.value,
                    "total_steps": len(self.execution_history),
                    "total_time": sum(h.get("execution_time", 0) for h in self.execution_history),
                    "average_confidence": sum(h.get("confidence", 0) for h in self.execution_history if h.get("confidence")) / len([h for h in self.execution_history if h.get("confidence")]) if self.execution_history else 0
                }
            }
            
        except Exception as e:
            self.workflow_state["status"] = "failed"
            self.workflow_state["error"] = str(e)
            
            self.logger.error(
                "Workflow execution failed",
                orchestrator="SevenStepsOrchestrator",
                problem_id=problem_id,
                error=str(e)
            )
            
            return {
                "workflow_status": "failed",
                "problem_id": problem_id,
                "error": str(e),
                "step_results": self.step_results,
                "execution_history": self.execution_history
            }
    
    async def _execute_agent_as_tool_mode(
        self,
        context: AgentContext,
        initial_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow using agent-as-tool pattern with intelligent orchestration."""
        
        # Create comprehensive orchestration prompt
        orchestration_prompt = f"""Execute a comprehensive seven-step business analysis for the following problem:

PROBLEM: {initial_input.get('raw_text', 'No problem description provided')}
SUBMITTER: {initial_input.get('submitter', 'Unknown')}
CONTEXT: {initial_input.get('metadata', {})}

Use the available agent tools to execute the seven-step methodology:
1. Frame Problem → 2. Create Issue Tree → 3. Prioritize Analyses → 4. Create Plans → 5. Execute Analysis → 6. Synthesize Results → 7. Create Presentation

EXECUTION GUIDELINES:
1. Start by framing the problem to create a structured foundation
2. If clarification is needed, gather requirements before proceeding
3. Create a comprehensive issue tree with MECE decomposition
4. Prioritize analyses based on impact and feasibility
5. Generate detailed analysis plans for high-priority items
6. Execute analyses with appropriate depth and rigor
7. Synthesize results into actionable business recommendations
8. Create presentation materials appropriate for stakeholders

QUALITY STANDARDS:
- Ensure each step builds logically on the previous one
- Validate outputs for completeness and accuracy
- Balance thoroughness with practical constraints
- Focus on actionable, business-relevant insights
- Maintain consistent quality throughout the process

Execute the workflow intelligently, making decisions about execution flow, error handling, and quality assurance. Provide comprehensive results that enable data-driven business decisions."""

        # Execute the orchestration using the agent's completion
        result = await self.run(orchestration_prompt)
        
        return {
            "orchestration_result": result,
            "execution_mode": "agent_as_tool_intelligent"
        }
    
    async def _execute_sequential_mode(
        self,
        context: AgentContext,
        initial_input: Dict[str, Any],
        selected_steps: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Execute workflow steps sequentially with dependency management."""
        
        # Determine steps to execute
        steps_to_run = selected_steps or [step.name for step in self.workflow_steps]
        
        results = {}
        current_input = initial_input
        
        for step in self.workflow_steps:
            if step.name not in steps_to_run:
                continue
            
            # Check dependencies
            missing_deps = [dep for dep in step.dependencies if dep not in results]
            if missing_deps:
                self.logger.warning(
                    "Skipping step due to missing dependencies",
                    step=step.name,
                    missing_dependencies=missing_deps
                )
                continue
            
            try:
                self.workflow_state["current_step"] = step.name
                
                # Get agent and execute
                agent = self.agent_registry.get(step.agent_name)
                if not agent:
                    raise Exception(f"Agent '{step.agent_name}' not found")
                
                # Prepare input with context and previous results
                step_input = {
                    "context": context.__dict__,
                    **current_input
                }
                
                # Add previous results as input
                for dep in step.dependencies:
                    if dep in results:
                        step_input[f"{dep}_result"] = results[dep]
                
                # Execute step
                result = await agent.execute(step_input)
                results[step.name] = result.data
                
                # Update current input for next step
                if result.success and result.data:
                    current_input = result.data
                
            except Exception as e:
                self.logger.error(
                    "Step execution failed",
                    step=step.name,
                    agent=step.agent_name,
                    error=str(e)
                )
                results[step.name] = {"success": False, "error": str(e)}
                
                if step.required:
                    break  # Stop execution if required step fails
        
        return results
    
    async def _execute_parallel_mode(
        self,
        context: AgentContext,
        initial_input: Dict[str, Any],
        selected_steps: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Execute independent workflow steps in parallel."""
        
        # For now, implement as sequential since most steps have dependencies
        # In full implementation, would analyze dependency graph and execute
        # independent steps concurrently
        return await self._execute_sequential_mode(context, initial_input, selected_steps)
    
    async def _execute_handoff_mode(
        self,
        context: AgentContext,
        initial_input: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow using agent handoff pattern."""
        
        # Start with problem framer
        current_agent = self.agent_registry.get("problem_framer")
        current_input = {
            "context": context.__dict__,
            **initial_input
        }
        
        handoff_chain = []
        
        while current_agent and len(handoff_chain) < 7:  # Max 7 steps
            try:
                # Execute current agent
                result = await current_agent.execute(current_input)
                
                handoff_chain.append({
                    "agent": current_agent.agent_name,
                    "success": result.success,
                    "data": result.data,
                    "confidence": result.confidence_score
                })
                
                # Determine next agent based on result
                next_agent_name = self._determine_next_agent(current_agent.agent_name, result)
                
                if next_agent_name:
                    current_agent = self.agent_registry.get(next_agent_name)
                    current_input = {"context": context.__dict__, **result.data}
                else:
                    break  # Workflow complete
                    
            except Exception as e:
                handoff_chain.append({
                    "agent": current_agent.agent_name if current_agent else "unknown",
                    "success": False,
                    "error": str(e)
                })
                break
        
        return {
            "handoff_chain": handoff_chain,
            "execution_mode": "handoff"
        }
    
    def _determine_next_agent(self, current_agent: str, result: Any) -> Optional[str]:
        """Determine the next agent in handoff chain."""
        
        # Simple linear progression for now
        agent_sequence = [
            "problem_framer", "issue_tree", "prioritization", 
            "planner", "data_analysis", "synthesizer", "presentation"
        ]
        
        try:
            current_index = agent_sequence.index(current_agent)
            if current_index < len(agent_sequence) - 1:
                return agent_sequence[current_index + 1]
        except ValueError:
            pass
        
        return None


# Global orchestrator instance
_orchestrator: Optional[SevenStepsOrchestrator] = None


def get_orchestrator() -> SevenStepsOrchestrator:
    """Get the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = SevenStepsOrchestrator()
    return _orchestrator