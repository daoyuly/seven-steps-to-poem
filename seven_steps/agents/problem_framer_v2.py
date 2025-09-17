"""
Problem Framer Agent implementation using OpenAI Agents SDK.

This agent analyzes raw business problems and creates structured
problem frames with goals, scope, KPIs, and clarifying questions.
"""

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from seven_steps.core.schemas import ProblemFrame, ClarifyingQuestion, KPI
from .base_v2 import BaseSevenStepsAgent, AgentInput, AgentOutput, AgentContext


# Structured Output Models
class ProblemFramingOutput(BaseModel):
    """Structured output for problem framing."""
    
    goal: str = Field(description="Clear, SMART goal statement")
    scope: Dict[str, List[str]] = Field(
        description="Scope with 'include' and 'exclude' lists"
    )
    kpis: List[Dict[str, Any]] = Field(
        description="List of key performance indicators"
    )
    stakeholders: List[str] = Field(
        description="Key stakeholder groups"
    )
    assumptions: List[str] = Field(
        description="Key assumptions being made"
    )
    constraints: List[str] = Field(
        description="Time, budget, or resource constraints"
    )
    confidence: str = Field(
        description="Confidence level: low|medium|high"
    )
    clarifying_questions: List[Dict[str, Any]] = Field(
        description="Questions needed for clarity"
    )


class ProblemFramerInput(AgentInput):
    """Input model for Problem Framer agent."""
    
    raw_text: str = Field(..., description="Raw problem description")
    submitter: str = Field(..., description="Problem submitter email")
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context metadata"
    )
    previous_clarifications: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Previous clarification answers"
    )


class ProblemFramerOutput(AgentOutput):
    """Output model for Problem Framer agent."""
    
    problem_frame: Optional[ProblemFrame] = None
    needs_clarification: bool = False
    clarifying_questions: List[ClarifyingQuestion] = Field(default_factory=list)


class ProblemFramer(BaseSevenStepsAgent):
    """
    Problem Framer Agent using OpenAI Agents SDK.
    
    This agent takes raw problem descriptions and converts them into
    structured problem frames following McKinsey's problem-solving methodology.
    """
    
    def __init__(self):
        """Initialize the Problem Framer agent."""
        
        instructions = """You are a senior business analyst specializing in problem structuring using McKinsey's methodology. Your role is to analyze raw business problems and create structured problem frames.

RESPONSIBILITIES:
1. Parse raw problem descriptions and extract key information
2. Define clear, measurable goals with specific success criteria
3. Establish scope boundaries (what's included/excluded)
4. Identify key stakeholders and their interests
5. Surface underlying assumptions that need validation
6. Identify constraints (time, budget, resources, regulatory)
7. Generate clarifying questions when information is insufficient

OUTPUT REQUIREMENTS:
- Goals must be SMART (Specific, Measurable, Achievable, Relevant, Time-bound)
- KPIs must have clear baselines and targets where possible
- Assumptions should be testable hypotheses
- Clarifying questions should be categorized and prioritized

PROBLEM ANALYSIS APPROACH:
1. Read the raw problem text carefully
2. Identify the core business challenge
3. Extract any quantifiable metrics mentioned
4. Determine stakeholder groups
5. Identify what information is missing
6. Structure the problem following MECE principles

If critical information is missing, generate specific clarifying questions rather than making assumptions. Focus on business impact and measurable outcomes."""

        # Define MCP tools for business analysis
        mcp_tools = [
            "data-source-connector",  # For accessing business data
            "stakeholder-analyzer",   # For stakeholder mapping
            "kpi-validator"          # For validating KPI definitions
        ]
        
        super().__init__(
            name="ProblemFramer",
            description="Structures raw business problems into actionable frameworks",
            instructions=instructions,
            output_type=ProblemFramingOutput,
            mcp_tools=mcp_tools
        )
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class."""
        return ProblemFramerInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class."""
        return ProblemFramerOutput
    
    async def process_request(self, input_data: ProblemFramerInput) -> ProblemFramerOutput:
        """
        Process raw problem text into structured problem frame.
        
        Args:
            input_data: Raw problem information
            
        Returns:
            ProblemFramerOutput with structured problem frame or clarifying questions
        """
        try:
            # Build comprehensive prompt
            prompt = self._create_analysis_prompt(input_data)
            
            # Add context data
            context = {
                "submitter": input_data.submitter,
                "organization_id": input_data.context.organization_id,
                "metadata": input_data.metadata,
                "previous_clarifications": input_data.previous_clarifications
            }
            
            # Run structured completion
            structured_result = await self.run_structured_completion(
                prompt=prompt,
                context=context,
                output_type=ProblemFramingOutput
            )
            
            # Determine if we need clarifications
            needs_clarification = self._needs_clarification(structured_result, input_data)
            
            if needs_clarification:
                # Extract clarifying questions
                clarifying_questions = self._extract_clarifying_questions(structured_result)
                
                return ProblemFramerOutput(
                    success=True,
                    needs_clarification=True,
                    clarifying_questions=clarifying_questions,
                    confidence_score=0.6,  # Lower confidence when needing clarification
                    data={
                        "partial_frame": structured_result.dict(),
                        "clarification_needed": True
                    }
                )
            else:
                # Create complete problem frame
                problem_frame = await self._create_problem_frame(structured_result, input_data)
                
                return ProblemFramerOutput(
                    success=True,
                    problem_frame=problem_frame,
                    needs_clarification=False,
                    confidence_score=self._calculate_confidence(structured_result),
                    data={
                        "problem_frame": problem_frame.dict() if problem_frame else None,
                        "structured_output": structured_result.dict()
                    }
                )
                
        except Exception as e:
            self.logger.error(
                "Problem framing failed",
                error=str(e),
                problem_id=input_data.context.problem_id
            )
            
            return ProblemFramerOutput(
                success=False,
                error_message=f"Problem framing failed: {str(e)}"
            )
    
    def _create_analysis_prompt(self, input_data: ProblemFramerInput) -> str:
        """Create the main analysis prompt for structured output."""
        
        prompt = f"""Analyze the following business problem and create a structured problem frame:

PROBLEM DESCRIPTION:
{input_data.raw_text}

SUBMITTER: {input_data.submitter}
"""
        
        if input_data.metadata:
            prompt += f"\nADDITIONAL CONTEXT:\n"
            for key, value in input_data.metadata.items():
                prompt += f"- {key}: {value}\n"
        
        if input_data.previous_clarifications:
            prompt += f"\nPREVIOUS CLARIFICATIONS:\n"
            for question, answer in input_data.previous_clarifications.items():
                prompt += f"- Q: {question}\n  A: {answer}\n"
        
        prompt += """

Provide a comprehensive analysis following the structured output format. Focus on:

1. GOAL: Create a SMART goal that clearly defines success
2. SCOPE: Explicitly define what is included and excluded
3. KPIs: Identify measurable metrics with baselines and targets
4. STAKEHOLDERS: List all relevant parties and their interests
5. ASSUMPTIONS: Surface key assumptions that need validation
6. CONSTRAINTS: Identify time, budget, resource, or regulatory limits
7. CLARIFYING QUESTIONS: Generate specific questions if information is missing

Ensure all elements are specific, actionable, and business-focused."""
        
        return prompt
    
    def _needs_clarification(
        self, 
        structured_result: ProblemFramingOutput, 
        input_data: ProblemFramerInput
    ) -> bool:
        """Determine if clarification is needed before creating problem frame."""
        
        # Check if there are required clarifying questions
        clarifying_questions = structured_result.clarifying_questions
        required_questions = [
            q for q in clarifying_questions 
            if q.get("required", False)
        ]
        
        # Need clarification if there are required questions and no previous answers
        if required_questions and not input_data.previous_clarifications:
            return True
        
        # Check if goal is too vague
        goal = structured_result.goal.strip()
        if len(goal) < 30 or not any(char.isdigit() for char in goal):
            return True
        
        # Check if KPIs lack specific targets
        kpis = structured_result.kpis
        if kpis:
            missing_targets = any(
                not kpi.get("baseline") and not kpi.get("target")
                for kpi in kpis
            )
            if missing_targets:
                return True
        
        return False
    
    def _extract_clarifying_questions(
        self, 
        structured_result: ProblemFramingOutput
    ) -> List[ClarifyingQuestion]:
        """Extract and structure clarifying questions."""
        
        questions = []
        
        for q_data in structured_result.clarifying_questions:
            question = ClarifyingQuestion(
                question=q_data.get("question", ""),
                category=q_data.get("category", "business"),
                required=q_data.get("required", False)
            )
            questions.append(question)
        
        return questions
    
    async def _create_problem_frame(
        self, 
        structured_result: ProblemFramingOutput,
        input_data: ProblemFramerInput
    ) -> ProblemFrame:
        """Create a structured problem frame from the analysis."""
        
        # Process KPIs with validation
        kpis = []
        for kpi_data in structured_result.kpis:
            # Validate KPI using MCP tool if available
            validated_kpi = await self._validate_kpi(kpi_data)
            
            kpi = KPI(
                name=validated_kpi.get("name", ""),
                baseline=validated_kpi.get("baseline"),
                target=validated_kpi.get("target"),
                window=validated_kpi.get("window", ""),
                unit=validated_kpi.get("unit")
            )
            kpis.append(kpi)
        
        # Process clarifying questions
        questions = self._extract_clarifying_questions(structured_result)
        
        # Create the problem frame
        problem_frame = ProblemFrame(
            goal=structured_result.goal,
            scope=structured_result.scope,
            kpis=kpis,
            stakeholders=structured_result.stakeholders,
            assumptions=structured_result.assumptions,
            constraints=structured_result.constraints,
            confidence=structured_result.confidence,
            clarifying_questions=questions
        )
        
        return problem_frame
    
    async def _validate_kpi(self, kpi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate KPI definition using MCP tool."""
        try:
            # Call KPI validator MCP tool
            validation_result = await self.call_mcp_tool(
                tool_name="kpi-validator",
                parameters={
                    "kpi_name": kpi_data.get("name"),
                    "baseline": kpi_data.get("baseline"),
                    "target": kpi_data.get("target"),
                    "unit": kpi_data.get("unit")
                }
            )
            
            if validation_result["success"]:
                return validation_result["result"].get("validated_kpi", kpi_data)
            else:
                self.logger.warning(
                    "KPI validation failed, using original data",
                    kpi=kpi_data,
                    error=validation_result["error"]
                )
                return kpi_data
                
        except Exception as e:
            self.logger.warning(
                "KPI validation error, using original data",
                kpi=kpi_data,
                error=str(e)
            )
            return kpi_data
    
    def _calculate_confidence(self, structured_result: ProblemFramingOutput) -> float:
        """Calculate confidence score based on analysis quality."""
        
        score = 0.5  # Base score
        
        # Goal quality check
        goal = structured_result.goal
        if len(goal) > 40 and any(char.isdigit() for char in goal):
            score += 0.15
        
        # KPI quality check
        kpis = structured_result.kpis
        if kpis and len(kpis) >= 2:
            score += 0.1
            
            # Check for quantified KPIs
            quantified_kpis = sum(
                1 for kpi in kpis 
                if kpi.get("baseline") or kpi.get("target")
            )
            if quantified_kpis / len(kpis) >= 0.5:
                score += 0.1
        
        # Stakeholder coverage
        if len(structured_result.stakeholders) >= 3:
            score += 0.1
        
        # Assumption quality
        if len(structured_result.assumptions) >= 2:
            score += 0.05
        
        # Constraint identification
        if len(structured_result.constraints) >= 1:
            score += 0.05
        
        # Confidence self-assessment
        confidence_level = structured_result.confidence
        if confidence_level == "high":
            score += 0.05
        elif confidence_level == "low":
            score -= 0.05
        
        return min(1.0, max(0.0, score))
    
    @BaseSevenStepsAgent.create_tool_function(
        "analyze_stakeholders", 
        "Analyze stakeholder relationships and influence mapping"
    )
    async def analyze_stakeholders(self, stakeholders: List[str]) -> Dict[str, Any]:
        """Tool function to analyze stakeholder relationships."""
        
        # Call stakeholder analyzer MCP tool
        result = await self.call_mcp_tool(
            tool_name="stakeholder-analyzer",
            parameters={"stakeholders": stakeholders}
        )
        
        if result["success"]:
            return result["result"]
        else:
            # Fallback analysis
            return {
                "stakeholder_matrix": {
                    stakeholder: {
                        "influence": "medium",
                        "interest": "medium",
                        "impact": "medium"
                    }
                    for stakeholder in stakeholders
                },
                "key_relationships": [],
                "communication_strategy": "Regular updates recommended"
            }