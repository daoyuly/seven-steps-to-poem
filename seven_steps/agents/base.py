"""
Base classes for AI agents using OpenAI Agent framework.

This module provides the foundational classes and interfaces
for all AI agents in the Seven Steps system.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from seven_steps.core.config import get_settings
from seven_steps.core.exceptions import LLMError, ValidationError
from seven_steps.core.logging import LoggerMixin, get_correlation_id


class AgentContext(BaseModel):
    """Context information passed to agents."""
    
    user_id: str
    organization_id: str
    problem_id: str
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AgentInput(BaseModel):
    """Base input model for agents."""
    
    context: AgentContext


class AgentOutput(BaseModel):
    """Base output model for agents."""
    
    success: bool
    data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    confidence_score: Optional[float] = None
    execution_time: Optional[float] = None


class BaseAgent(ABC, LoggerMixin):
    """
    Base class for all AI agents.
    
    This class provides common functionality for interacting with
    OpenAI's API, handling retries, validation, and logging.
    """
    
    def __init__(self):
        """Initialize the base agent."""
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=self.settings.llm.openai_api_key)
        self.model = self.settings.llm.model
        self.temperature = self.settings.llm.temperature
        self.max_tokens = self.settings.llm.max_tokens
        self.timeout = self.settings.llm.timeout
    
    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Return the name of this agent."""
        pass
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        pass
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class for this agent."""
        return AgentInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class for this agent."""
        return AgentOutput
    
    @abstractmethod
    async def process(self, input_data: AgentInput) -> AgentOutput:
        """
        Process the input and return the result.
        
        Args:
            input_data: Validated input data for the agent
            
        Returns:
            AgentOutput: The processed result
        """
        pass
    
    @retry(
        retry=retry_if_exception_type(LLMError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def call_llm(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a call to the OpenAI API with retry logic.
        
        Args:
            messages: List of messages for the conversation
            functions: Optional function definitions for function calling
            function_call: Optional function call specification
            **kwargs: Additional parameters for the API call
            
        Returns:
            Dict containing the API response
            
        Raises:
            LLMError: If the API call fails after retries
        """
        correlation_id = get_correlation_id()
        
        try:
            self.logger.info(
                "Calling OpenAI API",
                agent=self.agent_name,
                model=self.model,
                correlation_id=correlation_id,
                message_count=len(messages)
            )
            
            # Prepare API call parameters
            call_params = {
                "model": self.model,
                "messages": messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "timeout": kwargs.get("timeout", self.timeout),
            }
            
            # Add function calling parameters if provided
            if functions:
                call_params["functions"] = functions
            if function_call:
                call_params["function_call"] = function_call
            
            # Make the API call
            response = await self.client.chat.completions.create(**call_params)
            
            # Convert response to dict for easier handling
            result = {
                "id": response.id,
                "choices": [
                    {
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                            "function_call": (
                                {
                                    "name": choice.message.function_call.name,
                                    "arguments": choice.message.function_call.arguments,
                                }
                                if choice.message.function_call
                                else None
                            ),
                        },
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in response.choices
                ],
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }
            
            self.logger.info(
                "OpenAI API call successful",
                agent=self.agent_name,
                correlation_id=correlation_id,
                tokens_used=result["usage"]["total_tokens"],
                finish_reason=result["choices"][0]["finish_reason"]
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "OpenAI API call failed",
                agent=self.agent_name,
                correlation_id=correlation_id,
                error=str(e)
            )
            raise LLMError(
                message=f"OpenAI API call failed: {str(e)}",
                details={
                    "agent": self.agent_name,
                    "model": self.model,
                    "correlation_id": correlation_id,
                }
            ) from e
    
    def build_messages(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        additional_context: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Build the message list for OpenAI API call.
        
        Args:
            user_message: The main user message
            context: Optional context data to include
            additional_context: Optional additional context string
            
        Returns:
            List of message dictionaries
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add context information if provided
        if context or additional_context:
            context_message = "Context Information:\n"
            
            if context:
                context_message += f"Context Data: {json.dumps(context, indent=2)}\n"
            
            if additional_context:
                context_message += f"Additional Context: {additional_context}\n"
            
            messages.append({"role": "system", "content": context_message})
        
        # Add the main user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def parse_json_response(
        self,
        response_content: str,
        expected_model: Optional[Type[BaseModel]] = None
    ) -> Dict[str, Any]:
        """
        Parse JSON response from LLM and validate if model provided.
        
        Args:
            response_content: Raw response content from LLM
            expected_model: Optional Pydantic model for validation
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValidationError: If JSON parsing or validation fails
        """
        try:
            # Try to parse JSON
            data = json.loads(response_content)
            
            # Validate against model if provided
            if expected_model:
                validated_data = expected_model(**data)
                return validated_data.dict()
            
            return data
            
        except json.JSONDecodeError as e:
            self.logger.error(
                "Failed to parse JSON response",
                agent=self.agent_name,
                response_content=response_content[:500] + "..." if len(response_content) > 500 else response_content,
                error=str(e)
            )
            raise ValidationError(
                message="Failed to parse JSON response from LLM",
                details={
                    "agent": self.agent_name,
                    "error": str(e),
                    "response_preview": response_content[:200]
                }
            ) from e
            
        except Exception as e:
            self.logger.error(
                "Failed to validate response data",
                agent=self.agent_name,
                error=str(e)
            )
            raise ValidationError(
                message="Failed to validate response data",
                details={
                    "agent": self.agent_name,
                    "error": str(e),
                }
            ) from e
    
    async def execute(self, raw_input: Dict[str, Any]) -> AgentOutput:
        """
        Execute the agent with raw input data.
        
        This method handles input validation, processing, and output formatting.
        
        Args:
            raw_input: Raw input data dictionary
            
        Returns:
            AgentOutput: Processed result with success/failure information
        """
        start_time = asyncio.get_event_loop().time()
        correlation_id = get_correlation_id()
        
        try:
            # Validate input
            self.logger.info(
                "Starting agent execution",
                agent=self.agent_name,
                correlation_id=correlation_id
            )
            
            input_data = self.input_model(**raw_input)
            
            # Process the input
            result = await self.process(input_data)
            
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


class AgentRegistry:
    """Registry for managing agent instances."""
    
    _agents: Dict[str, BaseAgent] = {}
    
    @classmethod
    def register(cls, name: str, agent: BaseAgent) -> None:
        """Register an agent with a given name."""
        cls._agents[name] = agent
    
    @classmethod
    def get(cls, name: str) -> Optional[BaseAgent]:
        """Get an agent by name."""
        return cls._agents.get(name)
    
    @classmethod
    def get_all(cls) -> Dict[str, BaseAgent]:
        """Get all registered agents."""
        return cls._agents.copy()
    
    @classmethod
    def list_names(cls) -> List[str]:
        """Get list of registered agent names."""
        return list(cls._agents.keys())