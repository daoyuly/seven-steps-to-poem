"""
Problem Framer Agent implementation.

This agent analyzes raw business problems and creates structured
problem frames with goals, scope, KPIs, and clarifying questions.
"""

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from seven_steps.core.schemas import ClarifyingQuestion, KPI, ProblemFrame
from .base import AgentContext, AgentInput, AgentOutput, BaseAgent


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


class ProblemFramer(BaseAgent):
    """
    Problem Framer Agent for structured problem analysis.
    
    This agent takes raw problem descriptions and converts them into
    structured problem frames following McKinsey's problem-solving methodology.
    """
    
    @property
    def agent_name(self) -> str:
        """Return the name of this agent."""
        return "ProblemFramer"
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return """You are a senior business analyst specializing in problem structuring using McKinsey's methodology. Your role is to analyze raw business problems and create structured problem frames.

RESPONSIBILITIES:
1. Parse raw problem descriptions and extract key information
2. Define clear, measurable goals with specific success criteria
3. Establish scope boundaries (what's included/excluded)
4. Identify key stakeholders and their interests
5. Surface underlying assumptions that need validation
6. Identify constraints (time, budget, resources, regulatory)
7. Generate clarifying questions when information is insufficient

OUTPUT REQUIREMENTS:
- Return valid JSON matching the ProblemFrame schema
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
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class."""
        return ProblemFramerInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class."""
        return ProblemFramerOutput
    
    async def process(self, input_data: ProblemFramerInput) -> ProblemFramerOutput:
        """
        Process raw problem text into structured problem frame.
        
        Args:
            input_data: Raw problem information
            
        Returns:
            ProblemFramerOutput with structured problem frame or clarifying questions
        """
        try:
            # Build context for the LLM
            context = self._build_analysis_context(input_data)
            
            # Create the analysis prompt
            user_message = self._create_analysis_prompt(input_data)
            
            # Build messages for API call
            messages = self.build_messages(
                user_message=user_message,
                context=context,
                additional_context=self._get_additional_context(input_data)
            )
            
            # Call the LLM
            response = await self.call_llm(messages)
            response_content = response["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            parsed_response = self.parse_json_response(response_content)
            
            # Determine if we have enough information or need clarifications
            if self._needs_clarification(parsed_response, input_data):
                return ProblemFramerOutput(
                    success=True,
                    needs_clarification=True,
                    clarifying_questions=self._extract_clarifying_questions(parsed_response),
                    confidence_score=0.6  # Lower confidence when needing clarification
                )
            else:
                # Create the problem frame
                problem_frame = self._create_problem_frame(parsed_response)
                
                return ProblemFramerOutput(
                    success=True,
                    problem_frame=problem_frame,
                    needs_clarification=False,
                    confidence_score=self._calculate_confidence(parsed_response, input_data)
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
    
    def _build_analysis_context(self, input_data: ProblemFramerInput) -> Dict[str, Any]:
        """Build context information for problem analysis."""
        return {
            "submitter": input_data.submitter,
            "organization_id": input_data.context.organization_id,
            "metadata": input_data.metadata,
            "has_clarifications": bool(input_data.previous_clarifications),
            "clarifications": input_data.previous_clarifications or {}
        }
    
    def _create_analysis_prompt(self, input_data: ProblemFramerInput) -> str:
        """Create the main analysis prompt for the LLM."""
        prompt = f"""Analyze the following business problem and create a structured problem frame:

PROBLEM DESCRIPTION:
{input_data.raw_text}

SUBMITTER: {input_data.submitter}
"""
        
        if input_data.metadata:
            prompt += f"\nADDITIONAL CONTEXT:\n{json.dumps(input_data.metadata, indent=2)}"
        
        if input_data.previous_clarifications:
            prompt += f"\nPREVIOUS CLARIFICATIONS:\n{json.dumps(input_data.previous_clarifications, indent=2)}"
        
        prompt += """

Please return a JSON object with the following structure:
{
  "goal": "Clear, SMART goal statement",
  "scope": {
    "include": ["what is included in scope"],
    "exclude": ["what is explicitly excluded"]
  },
  "kpis": [
    {
      "name": "KPI name",
      "baseline": current_value_if_known,
      "target": target_value_if_known,
      "window": "time_frame",
      "unit": "measurement_unit"
    }
  ],
  "stakeholders": ["key stakeholder groups"],
  "assumptions": ["key assumptions being made"],
  "constraints": ["time, budget, or resource constraints"],
  "confidence": "low|medium|high",
  "clarifying_questions": [
    {
      "question": "specific question text",
      "category": "data|business|technical|stakeholder",
      "required": true|false
    }
  ]
}

If you don't have enough information to create a complete problem frame, focus on generating high-quality clarifying questions in the clarifying_questions array."""
        
        return prompt
    
    def _get_additional_context(self, input_data: ProblemFramerInput) -> str:
        """Generate additional context string."""
        context_parts = []
        
        if input_data.metadata.get("industry"):
            context_parts.append(f"Industry: {input_data.metadata['industry']}")
        
        if input_data.metadata.get("company_size"):
            context_parts.append(f"Company Size: {input_data.metadata['company_size']}")
        
        if input_data.metadata.get("urgency"):
            context_parts.append(f"Urgency: {input_data.metadata['urgency']}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _needs_clarification(
        self, 
        parsed_response: Dict[str, Any], 
        input_data: ProblemFramerInput
    ) -> bool:
        """Determine if clarification is needed before creating problem frame."""
        # Check if there are clarifying questions
        clarifying_questions = parsed_response.get("clarifying_questions", [])
        required_questions = [q for q in clarifying_questions if q.get("required", False)]
        
        # Need clarification if there are required questions and no previous clarifications
        if required_questions and not input_data.previous_clarifications:
            return True
        
        # Check if goal is too vague or generic
        goal = parsed_response.get("goal", "").strip()
        if len(goal) < 20 or any(word in goal.lower() for word in ["improve", "increase", "better", "optimize"]) and "%" not in goal:
            return True
        
        # Check if KPIs are missing baseline or target values
        kpis = parsed_response.get("kpis", [])
        if kpis:
            missing_values = any(
                kpi.get("baseline") is None and kpi.get("target") is None 
                for kpi in kpis
            )
            if missing_values:
                return True
        
        return False
    
    def _extract_clarifying_questions(
        self, 
        parsed_response: Dict[str, Any]
    ) -> List[ClarifyingQuestion]:
        """Extract and structure clarifying questions."""
        questions_data = parsed_response.get("clarifying_questions", [])
        questions = []
        
        for q_data in questions_data:
            question = ClarifyingQuestion(
                question=q_data.get("question", ""),
                category=q_data.get("category", "business"),
                required=q_data.get("required", False)
            )
            questions.append(question)
        
        return questions
    
    def _create_problem_frame(self, parsed_response: Dict[str, Any]) -> ProblemFrame:
        """Create a structured problem frame from parsed response."""
        # Process KPIs
        kpis_data = parsed_response.get("kpis", [])
        kpis = []
        for kpi_data in kpis_data:
            kpi = KPI(
                name=kpi_data.get("name", ""),
                baseline=kpi_data.get("baseline"),
                target=kpi_data.get("target"),
                window=kpi_data.get("window", ""),
                unit=kpi_data.get("unit")
            )
            kpis.append(kpi)
        
        # Process clarifying questions (for reference)
        questions_data = parsed_response.get("clarifying_questions", [])
        questions = self._extract_clarifying_questions(parsed_response)
        
        # Create the problem frame
        problem_frame = ProblemFrame(
            goal=parsed_response.get("goal", ""),
            scope=parsed_response.get("scope", {"include": [], "exclude": []}),
            kpis=kpis,
            stakeholders=parsed_response.get("stakeholders", []),
            assumptions=parsed_response.get("assumptions", []),
            constraints=parsed_response.get("constraints", []),
            confidence=parsed_response.get("confidence", "medium"),
            clarifying_questions=questions
        )
        
        return problem_frame
    
    def _calculate_confidence(
        self, 
        parsed_response: Dict[str, Any], 
        input_data: ProblemFramerInput
    ) -> float:
        """Calculate confidence score based on information completeness."""
        score = 0.5  # Base score
        
        # Increase confidence based on completeness
        if parsed_response.get("goal") and len(parsed_response["goal"]) > 30:
            score += 0.1
        
        if parsed_response.get("kpis") and any(
            kpi.get("baseline") and kpi.get("target") 
            for kpi in parsed_response["kpis"]
        ):
            score += 0.1
        
        if len(parsed_response.get("stakeholders", [])) >= 3:
            score += 0.1
        
        if len(parsed_response.get("assumptions", [])) >= 2:
            score += 0.1
        
        if input_data.previous_clarifications:
            score += 0.1
        
        # Adjust based on LLM's confidence
        llm_confidence = parsed_response.get("confidence", "medium")
        if llm_confidence == "high":
            score += 0.05
        elif llm_confidence == "low":
            score -= 0.05
        
        return min(1.0, max(0.0, score))