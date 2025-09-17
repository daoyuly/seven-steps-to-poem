"""
Base classes for AI agents using OpenAI Agents SDK framework.

This module provides the foundational classes and interfaces
for all AI agents in the Seven Steps system, leveraging OpenAI's
structured outputs, tool integration, and MCP ecosystem.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union
from dataclasses import dataclass

import structlog
from pydantic import BaseModel
from openai import OpenAI
from openai.agents import Agent
from openai.agents.tools import HostedMCPTool, HTTPMCPTool

from seven_steps.core.config import get_settings
from seven_steps.core.exceptions import AgentError, ValidationError
from seven_steps.core.logging import LoggerMixin, get_correlation_id


@dataclass
class AgentContext:
    """Context information passed to agents."""
    user_id: str
    organization_id: str
    problem_id: str
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.correlation_id is None:
            self.correlation_id = get_correlation_id()


class AgentInput(BaseModel):
    """Base input model for agents."""
    context: AgentContext
    
    class Config:
        arbitrary_types_allowed = True


class AgentOutput(BaseModel):
    """Base output model for agents with structured validation."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    execution_time: Optional[float] = None
    artifacts: Optional[List[Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"


class BaseSevenStepsAgent(Agent, LoggerMixin):
    """
    Base class for all Seven Steps AI agents using OpenAI Agents SDK.
    
    This class provides common functionality for structured output,
    tool integration, MCP support, and agent orchestration.
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        instructions: str,
        model: str = "gpt-4o",
        output_type: Optional[Type[BaseModel]] = None,
        tools: Optional[List[Any]] = None,
        mcp_tools: Optional[List[str]] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Agent name
            description: Agent description
            instructions: System instructions for the agent
            model: OpenAI model to use
            output_type: Pydantic model for structured output
            tools: List of tools available to the agent
            mcp_tools: List of MCP server names to connect
        """
        self.settings = get_settings()
        self.agent_name = name
        self.agent_description = description
        
        # Initialize OpenAI client
        client = OpenAI(api_key=self.settings.llm.openai_api_key)
        
        # Prepare tools list
        agent_tools = []
        
        # Add custom tools
        if tools:
            agent_tools.extend(tools)
        
        # Add MCP tools
        if mcp_tools:
            for mcp_server in mcp_tools:
                if mcp_server.startswith("http"):
                    # HTTP MCP server
                    mcp_tool = HTTPMCPTool(server_url=mcp_server)
                else:
                    # Hosted MCP server
                    mcp_tool = HostedMCPTool(server=mcp_server)
                agent_tools.append(mcp_tool)
        
        # Initialize OpenAI Agent
        super().__init__(
            client=client,
            model=model,
            name=name,
            description=description,
            instructions=instructions,
            outputType=output_type,
            tools=agent_tools
        )
    
    @property
    @abstractmethod
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class for this agent."""
        pass
    
    @property
    @abstractmethod
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class for this agent."""
        pass
    
    @abstractmethod
    async def process_request(self, input_data: AgentInput) -> AgentOutput:
        """
        Process the input and return the result.
        
        Args:
            input_data: Validated input data for the agent
            
        Returns:
            AgentOutput: The processed result
        """
        pass
    
    async def execute(self, raw_input: Dict[str, Any]) -> AgentOutput:
        """
        Execute the agent with raw input data.
        
        This method handles input validation, processing, and output formatting
        using OpenAI Agents SDK structured approach.
        
        Args:
            raw_input: Raw input data dictionary
            
        Returns:
            AgentOutput: Processed result with success/failure information
        """
        start_time = asyncio.get_event_loop().time()
        correlation_id = raw_input.get("context", {}).get("correlation_id", get_correlation_id())
        
        try:
            self.logger.info(
                "Starting agent execution",
                agent=self.agent_name,
                correlation_id=correlation_id
            )
            
            # Validate input
            input_data = self.input_model(**raw_input)
            
            # Process the request
            result = await self.process_request(input_data)
            
            # Calculate execution time
            execution_time = asyncio.get_event_loop().time() - start_time
            result.execution_time = execution_time
            
            self.logger.info(
                "Agent execution completed successfully",
                agent=self.agent_name,
                correlation_id=correlation_id,
                execution_time=execution_time,
                confidence_score=result.confidence_score
            )
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            
            self.logger.error(
                "Agent execution failed",
                agent=self.agent_name,
                correlation_id=correlation_id,
                execution_time=execution_time,
                error=str(e)
            )
            
            return self.output_model(
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    async def run_structured_completion(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        output_type: Optional[Type[BaseModel]] = None
    ) -> Union[str, BaseModel]:
        """
        Run a structured completion using OpenAI Agents SDK.
        
        Args:
            prompt: The prompt to send
            context: Optional context data
            output_type: Optional Pydantic model for structured output
            
        Returns:
            Either a string response or structured BaseModel instance
        """
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            if context:
                context_msg = f"Context: {json.dumps(context, indent=2)}"
                messages.insert(0, {"role": "system", "content": context_msg})
            
            # Use the agent's completion method with structured output
            if output_type or self.outputType:
                structured_type = output_type or self.outputType
                response = await asyncio.to_thread(
                    self.client.beta.chat.completions.parse,
                    model=self.model,
                    messages=messages,
                    response_format=structured_type
                )
                return response.choices[0].message.parsed
            else:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=self.model,
                    messages=messages
                )
                return response.choices[0].message.content
                
        except Exception as e:
            self.logger.error(
                "Structured completion failed",
                agent=self.agent_name,
                error=str(e)
            )
            raise AgentError(
                message=f"Structured completion failed: {str(e)}",
                details={
                    "agent": self.agent_name,
                    "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt
                }
            ) from e
    
    async def delegate_to_subagent(
        self,
        subagent_name: str,
        task_description: str,
        input_data: Dict[str, Any],
        context: AgentContext
    ) -> Dict[str, Any]:
        """
        Delegate a subtask to another agent using function call pattern.
        
        Args:
            subagent_name: Name of the subagent to call
            task_description: Description of the task
            input_data: Data for the subtask
            context: Execution context
            
        Returns:
            Result from the subagent
        """
        from . import get_agent_registry
        
        registry = get_agent_registry()
        subagent = registry.get(subagent_name)
        
        if not subagent:
            raise AgentError(f"Subagent '{subagent_name}' not found in registry")
        
        self.logger.info(
            "Delegating task to subagent",
            parent_agent=self.agent_name,
            subagent=subagent_name,
            task=task_description,
            correlation_id=context.correlation_id
        )
        
        # Prepare input for subagent
        subagent_input = {
            "context": {
                "user_id": context.user_id,
                "organization_id": context.organization_id,
                "problem_id": context.problem_id,
                "correlation_id": context.correlation_id,
                "metadata": {
                    **context.metadata,
                    "parent_agent": self.agent_name,
                    "delegated_task": task_description
                }
            },
            **input_data
        }
        
        # Execute subagent
        result = await subagent.execute(subagent_input)
        
        return {
            "task_description": task_description,
            "subagent": subagent_name,
            "success": result.success,
            "data": result.data,
            "error": result.error_message,
            "confidence": result.confidence_score
        }
    
    async def call_mcp_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Call an MCP tool directly.
        
        Args:
            tool_name: Name of the MCP tool
            parameters: Parameters for the tool call
            
        Returns:
            Tool execution result
        """
        try:
            # Find the MCP tool
            mcp_tool = None
            for tool in self.tools:
                if hasattr(tool, 'name') and tool.name == tool_name:
                    mcp_tool = tool
                    break
            
            if not mcp_tool:
                raise AgentError(f"MCP tool '{tool_name}' not found")
            
            # Execute the tool
            result = await asyncio.to_thread(mcp_tool.call, **parameters)
            
            self.logger.info(
                "MCP tool executed successfully",
                agent=self.agent_name,
                tool=tool_name,
                parameters=parameters
            )
            
            return {
                "success": True,
                "result": result,
                "tool": tool_name
            }
            
        except Exception as e:
            self.logger.error(
                "MCP tool execution failed",
                agent=self.agent_name,
                tool=tool_name,
                error=str(e)
            )
            
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    def create_tool_function(self, func_name: str, func_description: str):
        """
        Decorator to create tool functions for this agent.
        
        Args:
            func_name: Name of the tool function
            func_description: Description of what the function does
            
        Returns:
            Decorator function
        """
        def decorator(func):
            # Create tool definition
            tool_def = {
                "type": "function",
                "function": {
                    "name": func_name,
                    "description": func_description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # Add to agent's tools
            if not hasattr(self, '_custom_tools'):
                self._custom_tools = []
            self._custom_tools.append((tool_def, func))
            
            return func
        
        return decorator


class AgentRegistry:
    """Enhanced registry for managing agent instances with OpenAI Agents SDK."""
    
    _agents: Dict[str, BaseSevenStepsAgent] = {}
    _agent_metadata: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(
        cls, 
        name: str, 
        agent: BaseSevenStepsAgent,
        capabilities: Optional[List[str]] = None,
        mcp_tools: Optional[List[str]] = None
    ) -> None:
        """Register an agent with enhanced metadata."""
        cls._agents[name] = agent
        cls._agent_metadata[name] = {
            "name": agent.agent_name,
            "description": agent.agent_description,
            "capabilities": capabilities or [],
            "mcp_tools": mcp_tools or [],
            "input_model": agent.input_model.__name__,
            "output_model": agent.output_model.__name__,
            "supports_structured_output": hasattr(agent, 'outputType') and agent.outputType is not None,
            "available_tools": len(agent.tools) if agent.tools else 0
        }
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseSevenStepsAgent]:
        """Get an agent by name."""
        return cls._agents.get(name)
    
    @classmethod
    def get_metadata(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get agent metadata by name."""
        return cls._agent_metadata.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, BaseSevenStepsAgent]:
        """Get all registered agents."""
        return cls._agents.copy()
    
    @classmethod
    def list_names(cls) -> List[str]:
        """Get list of registered agent names."""
        return list(cls._agents.keys())
    
    @classmethod
    def get_agents_with_capability(cls, capability: str) -> List[str]:
        """Get agents that have a specific capability."""
        matching_agents = []
        for name, metadata in cls._agent_metadata.items():
            if capability in metadata.get("capabilities", []):
                matching_agents.append(name)
        return matching_agents
    
    @classmethod
    def get_agents_with_mcp_tool(cls, mcp_tool: str) -> List[str]:
        """Get agents that have access to a specific MCP tool."""
        matching_agents = []
        for name, metadata in cls._agent_metadata.items():
            if mcp_tool in metadata.get("mcp_tools", []):
                matching_agents.append(name)
        return matching_agents


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    return AgentRegistry