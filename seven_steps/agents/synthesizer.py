"""
Synthesizer Agent implementation.

This agent integrates analysis results using pyramid principle to generate
structured conclusions, evidence chains, and actionable business recommendations.
"""

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from seven_steps.core.schemas import SynthesisResult, BusinessRecommendation, EvidenceChain
from .base import AgentContext, AgentInput, AgentOutput, BaseAgent


class SynthesizerInput(AgentInput):
    """Input model for Synthesizer agent."""
    
    analysis_results: List[Dict[str, Any]] = Field(..., description="Analysis results to synthesize")
    synthesis_framework: str = Field(
        default="pyramid_principle",
        description="Framework for synthesis: pyramid_principle|hypothesis_driven|impact_based"
    )
    business_objectives: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Original business objectives and success criteria"
    )
    stakeholder_priorities: Optional[List[str]] = Field(
        default=None,
        description="Stakeholder priorities for recommendation ranking"
    )


class SynthesizerOutput(AgentOutput):
    """Output model for Synthesizer agent."""
    
    synthesis_result: Optional[SynthesisResult] = None
    executive_summary: Optional[str] = None
    evidence_map: Optional[Dict[str, Any]] = None


class SynthesizerAgent(BaseAgent):
    """
    Synthesizer Agent for evidence-based business recommendations.
    
    This agent synthesizes multiple analysis results using structured thinking
    frameworks to generate coherent, actionable business recommendations with
    strong evidence chains and clear implementation guidance.
    """
    
    @property
    def agent_name(self) -> str:
        """Return the name of this agent."""
        return "SynthesizerAgent"
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return """You are a senior strategy consultant expert in synthesis and business recommendation development using McKinsey's pyramid principle and structured thinking. Your role is to integrate analytical findings into actionable business insights.

RESPONSIBILITIES:
1. Synthesize multiple analysis results into coherent insights
2. Build evidence chains supporting each recommendation
3. Apply pyramid principle for structured recommendation hierarchy
4. Quantify business impact and implementation requirements
5. Assess recommendation feasibility and risk factors
6. Create executive-level communication materials
7. Provide implementation roadmaps with success metrics

SYNTHESIS FRAMEWORKS:

PYRAMID PRINCIPLE:
- Start with conclusion/recommendation (top of pyramid)
- Support with 3-5 key arguments (middle layer)
- Back each argument with specific evidence (bottom layer)
- Ensure MECE organization at each level
- Maintain logical flow from evidence to conclusion

HYPOTHESIS-DRIVEN SYNTHESIS:
- Validate/refute original hypotheses from issue tree
- Link findings back to root problem statements
- Identify surprising insights and implications
- Build new hypotheses based on emergent patterns

IMPACT-BASED PRIORITIZATION:
- Quantify potential business value of each insight
- Assess implementation difficulty and timeline
- Consider stakeholder alignment and organizational readiness
- Balance quick wins vs long-term strategic moves

EVIDENCE CHAIN CONSTRUCTION:
- Link data observations to analytical insights
- Connect insights to business implications
- Trace implications to recommended actions
- Quantify confidence levels at each link
- Identify assumptions and potential failure points

RECOMMENDATION QUALITY CRITERIA:
- Specific and actionable (not vague guidance)
- Quantified impact estimates with confidence ranges
- Clear owner and timeline for implementation
- Risk assessment with mitigation strategies
- Success metrics and monitoring approach
- Resource requirements and dependencies

BUSINESS COMMUNICATION:
- Executive summary with key takeaways
- Supporting evidence organized by recommendation
- Implementation timeline with milestones
- Resource and budget requirements
- Risk register with mitigation plans
- Success metrics and tracking approach

OUTPUT REQUIREMENTS:
- Generate structured synthesis following pyramid principle
- Provide 3-5 prioritized business recommendations
- Include detailed evidence chains and impact estimates
- Create executive-ready summary and supporting materials
- Enable stakeholder alignment and decision-making"""
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class."""
        return SynthesizerInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class."""
        return SynthesizerOutput
    
    async def process(self, input_data: SynthesizerInput) -> SynthesizerOutput:
        """
        Process analysis results into synthesized business recommendations.
        
        Args:
            input_data: Analysis results and synthesis parameters
            
        Returns:
            SynthesizerOutput with synthesized insights and recommendations
        """
        try:
            # Build synthesis context
            context = self._build_synthesis_context(input_data)
            
            # Create synthesis prompt
            user_message = self._create_synthesis_prompt(input_data)
            
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
            
            # Create synthesis result
            synthesis_result = self._create_synthesis_result(parsed_response, input_data)
            
            # Generate executive summary
            executive_summary = self._create_executive_summary(synthesis_result, input_data)
            
            # Create evidence map
            evidence_map = self._create_evidence_map(synthesis_result, input_data.analysis_results)
            
            return SynthesizerOutput(
                success=True,
                synthesis_result=synthesis_result,
                executive_summary=executive_summary,
                evidence_map=evidence_map,
                confidence_score=self._calculate_confidence(parsed_response, input_data)
            )
                
        except Exception as e:
            self.logger.error(
                "Synthesis failed",
                error=str(e),
                problem_id=input_data.context.problem_id
            )
            
            return SynthesizerOutput(
                success=False,
                error_message=f"Synthesis failed: {str(e)}"
            )
    
    def _build_synthesis_context(self, input_data: SynthesizerInput) -> Dict[str, Any]:
        """Build context information for synthesis."""
        return {
            "total_analyses": len(input_data.analysis_results),
            "synthesis_framework": input_data.synthesis_framework,
            "business_objectives": input_data.business_objectives or {},
            "stakeholder_priorities": input_data.stakeholder_priorities or [],
            "problem_id": input_data.context.problem_id
        }
    
    def _create_synthesis_prompt(self, input_data: SynthesizerInput) -> str:
        """Create the main synthesis prompt for the LLM."""
        prompt = f"""Synthesize the following {len(input_data.analysis_results)} analysis results into actionable business recommendations using {input_data.synthesis_framework}.

ANALYSIS RESULTS TO SYNTHESIZE:
"""
        
        for i, result in enumerate(input_data.analysis_results, 1):
            prompt += f"""
{i}. ANALYSIS: {result.get('plan_id', f'Analysis_{i}')}
   Status: {result.get('execution_status', 'completed')}
   Summary: {result.get('execution_summary', '')}
   
   KEY FINDINGS:
   {chr(10).join(['   - ' + finding for finding in result.get('key_findings', [])])}
   
   RECOMMENDATIONS:
   {chr(10).join(['   - ' + rec for rec in result.get('recommendations', [])])}
   
   QUALITY METRICS:
   {json.dumps(result.get('quality_metrics', {}), indent=6)}
   
   ARTIFACTS GENERATED:
   {chr(10).join(['   - ' + art.get('name', 'Unknown') + ': ' + art.get('description', '') for art in result.get('artifacts', [])])}
"""
        
        if input_data.business_objectives:
            prompt += f"\nORIGINAL BUSINESS OBJECTIVES:\n{json.dumps(input_data.business_objectives, indent=2)}"
        
        if input_data.stakeholder_priorities:
            prompt += f"\nSTAKEHOLDER PRIORITIES: {', '.join(input_data.stakeholder_priorities)}"
        
        prompt += f"""

Please synthesize these results using the {input_data.synthesis_framework} framework and return JSON in this format:

{{
  "synthesis_overview": {{
    "main_conclusion": "Single, clear statement of the primary finding",
    "confidence_level": 0.85,
    "evidence_strength": "strong|moderate|weak",
    "business_impact_category": "high|medium|low"
  }},
  "key_insights": [
    {{
      "insight": "Specific insight derived from analysis",
      "supporting_analyses": ["plan_id_1", "plan_id_2"],
      "evidence_quality": "strong|moderate|weak",
      "business_relevance": "high|medium|low",
      "quantitative_support": "15% improvement in retention observed",
      "confidence_level": 0.8
    }}
  ],
  "business_recommendations": [
    {{
      "recommendation_id": "R1",
      "title": "Implement targeted customer retention program",
      "description": "Detailed description of what should be done",
      "priority": "high|medium|low",
      "evidence_chain": [
        {{
          "level": "data_observation",
          "statement": "Customer churn increases 40% in month 2",
          "source_analysis": "plan_id_1",
          "confidence": 0.9
        }},
        {{
          "level": "analytical_insight", 
          "statement": "Early intervention prevents 60% of month-2 churn",
          "source_analysis": "plan_id_2",
          "confidence": 0.8
        }},
        {{
          "level": "business_implication",
          "statement": "Retention program could save $2M annually",
          "confidence": 0.7
        }},
        {{
          "level": "recommended_action",
          "statement": "Deploy automated retention campaign",
          "confidence": 0.8
        }}
      ],
      "impact_estimate": {{
        "metric": "customer_retention_rate",
        "baseline": 0.75,
        "target": 0.85,
        "timeframe": "6_months",
        "confidence_range": {{"low": 0.80, "high": 0.90}},
        "financial_impact": {{"annual_value": 2000000, "investment_required": 300000}}
      }},
      "implementation": {{
        "owner": "Customer Success Team",
        "timeline": "3 months implementation + 3 months measurement",
        "key_milestones": ["Campaign design (Month 1)", "Pilot launch (Month 2)", "Full rollout (Month 3)"],
        "resource_requirements": ["2 FTE developers", "$50K marketing budget"],
        "dependencies": ["CRM system upgrade", "Marketing automation platform"],
        "success_metrics": ["Retention rate", "Customer satisfaction", "Revenue impact"]
      }},
      "risks": [
        {{
          "risk": "Customer fatigue from increased communications",
          "probability": "medium",
          "impact": "medium",
          "mitigation": "A/B test communication frequency and personalization"
        }}
      ]
    }}
  ],
  "evidence_synthesis": {{
    "strong_evidence_areas": ["Customer behavior patterns", "Retention intervention effectiveness"],
    "weak_evidence_areas": ["Long-term impact projections", "Competitive response"],
    "contradictory_findings": ["Analysis X showed Y, but Analysis Z showed opposite"],
    "data_quality_assessment": "overall_strong|mixed|concerning",
    "additional_analysis_needed": ["Competitive benchmarking", "Customer satisfaction correlation"]
  }},
  "implementation_roadmap": {{
    "phase_1": {{
      "name": "Quick Wins (0-3 months)",
      "recommendations": ["R1", "R2"],
      "expected_impact": "10-15% improvement in key metrics",
      "resource_requirement": "Low to moderate"
    }},
    "phase_2": {{
      "name": "Strategic Initiatives (3-12 months)", 
      "recommendations": ["R3", "R4"],
      "expected_impact": "20-30% improvement in key metrics",
      "resource_requirement": "High"
    }},
    "success_tracking": {{
      "key_metrics": ["metric1", "metric2"],
      "measurement_frequency": "monthly|quarterly", 
      "review_checkpoints": ["Month 3", "Month 6", "Month 12"]
    }}
  }},
  "executive_summary": {{
    "situation": "Brief description of current situation based on analysis",
    "key_findings": ["Top 3 most important findings"],
    "recommended_actions": ["Top 3 recommended actions"],
    "expected_outcomes": ["Top 3 expected business outcomes"],
    "investment_required": "$XXX,XXX total investment",
    "timeline": "X months to full implementation",
    "risk_level": "low|medium|high"
  }}
}}

SYNTHESIS REQUIREMENTS:
1. Apply pyramid principle: conclusion → supporting arguments → evidence
2. Ensure recommendations are specific, measurable, and actionable
3. Build clear evidence chains from data to recommendations  
4. Quantify business impact with confidence ranges
5. Address stakeholder priorities and business objectives
6. Identify quick wins vs strategic initiatives
7. Include comprehensive risk assessment and mitigation
8. Provide implementation timeline with resource requirements"""
        
        return prompt
    
    def _get_additional_context(self, input_data: SynthesizerInput) -> str:
        """Generate additional context string."""
        context_parts = []
        
        context_parts.append(f"Framework: {input_data.synthesis_framework}")
        context_parts.append(f"Analyses: {len(input_data.analysis_results)}")
        
        if input_data.stakeholder_priorities:
            context_parts.append(f"Priorities: {', '.join(input_data.stakeholder_priorities[:3])}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _create_synthesis_result(
        self, 
        parsed_response: Dict[str, Any], 
        input_data: SynthesizerInput
    ) -> SynthesisResult:
        """Create structured synthesis result from parsed response."""
        # Process business recommendations
        recommendations_data = parsed_response.get("business_recommendations", [])
        recommendations = []
        
        for rec_data in recommendations_data:
            # Process evidence chain
            evidence_chain_data = rec_data.get("evidence_chain", [])
            evidence_chain = []
            
            for ec_data in evidence_chain_data:
                evidence = EvidenceChain(
                    level=ec_data.get("level", ""),
                    statement=ec_data.get("statement", ""),
                    source_analysis=ec_data.get("source_analysis", ""),
                    confidence=ec_data.get("confidence", 0.5)
                )
                evidence_chain.append(evidence)
            
            recommendation = BusinessRecommendation(
                recommendation_id=rec_data.get("recommendation_id", ""),
                title=rec_data.get("title", ""),
                description=rec_data.get("description", ""),
                priority=rec_data.get("priority", "medium"),
                evidence_chain=evidence_chain,
                impact_estimate=rec_data.get("impact_estimate", {}),
                implementation=rec_data.get("implementation", {}),
                risks=rec_data.get("risks", [])
            )
            recommendations.append(recommendation)
        
        # Create synthesis result
        synthesis_result = SynthesisResult(
            problem_id=input_data.context.problem_id,
            synthesis_overview=parsed_response.get("synthesis_overview", {}),
            key_insights=parsed_response.get("key_insights", []),
            business_recommendations=recommendations,
            evidence_synthesis=parsed_response.get("evidence_synthesis", {}),
            implementation_roadmap=parsed_response.get("implementation_roadmap", {}),
            executive_summary=parsed_response.get("executive_summary", {}),
            synthesis_metadata={
                "framework_used": input_data.synthesis_framework,
                "analyses_synthesized": len(input_data.analysis_results),
                "synthesis_timestamp": "2024-01-01T00:00:00Z"  # Would be actual timestamp
            }
        )
        
        return synthesis_result
    
    def _create_executive_summary(
        self, 
        synthesis_result: SynthesisResult, 
        input_data: SynthesizerInput
    ) -> str:
        """Create formatted executive summary."""
        exec_summary = synthesis_result.executive_summary
        
        summary = f"""# Executive Summary

## Situation
{exec_summary.get('situation', 'Analysis of business problem completed.')}

## Key Findings
{chr(10).join(['• ' + finding for finding in exec_summary.get('key_findings', [])])}

## Recommended Actions
{chr(10).join(['• ' + action for action in exec_summary.get('recommended_actions', [])])}

## Expected Outcomes
{chr(10).join(['• ' + outcome for outcome in exec_summary.get('expected_outcomes', [])])}

## Investment & Timeline
- **Investment Required**: {exec_summary.get('investment_required', 'TBD')}
- **Timeline**: {exec_summary.get('timeline', 'TBD')}
- **Risk Level**: {exec_summary.get('risk_level', 'Medium')}

## Confidence Assessment
- **Overall Confidence**: {synthesis_result.synthesis_overview.get('confidence_level', 0.7):.0%}
- **Evidence Strength**: {synthesis_result.synthesis_overview.get('evidence_strength', 'Moderate')}
- **Business Impact**: {synthesis_result.synthesis_overview.get('business_impact_category', 'Medium')}
"""
        
        return summary
    
    def _create_evidence_map(
        self, 
        synthesis_result: SynthesisResult, 
        analysis_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create evidence mapping for traceability."""
        evidence_map = {
            "recommendation_evidence": {},
            "analysis_contribution": {},
            "evidence_strength_matrix": {},
            "traceability_links": []
        }
        
        # Map recommendations to supporting evidence
        for rec in synthesis_result.business_recommendations:
            evidence_map["recommendation_evidence"][rec.recommendation_id] = {
                "title": rec.title,
                "evidence_chain_length": len(rec.evidence_chain),
                "supporting_analyses": list(set(
                    evidence.source_analysis for evidence in rec.evidence_chain
                    if evidence.source_analysis
                )),
                "confidence_range": {
                    "min": min(evidence.confidence for evidence in rec.evidence_chain) if rec.evidence_chain else 0,
                    "max": max(evidence.confidence for evidence in rec.evidence_chain) if rec.evidence_chain else 0
                }
            }
        
        # Map analysis contribution to recommendations
        for result in analysis_results:
            plan_id = result.get("plan_id", "unknown")
            contributing_recommendations = []
            
            for rec in synthesis_result.business_recommendations:
                if any(evidence.source_analysis == plan_id for evidence in rec.evidence_chain):
                    contributing_recommendations.append(rec.recommendation_id)
            
            evidence_map["analysis_contribution"][plan_id] = {
                "contributing_recommendations": contributing_recommendations,
                "key_findings_count": len(result.get("key_findings", [])),
                "quality_score": result.get("quality_metrics", {}).get("peer_review_score", 0),
                "execution_status": result.get("execution_status", "unknown")
            }
        
        # Create evidence strength matrix
        for rec in synthesis_result.business_recommendations:
            strength_scores = []
            for evidence in rec.evidence_chain:
                # Score based on confidence and evidence type
                base_score = evidence.confidence
                
                # Boost for quantitative evidence
                if any(char.isdigit() for char in evidence.statement):
                    base_score += 0.1
                
                # Boost for strong source analysis
                source_quality = next(
                    (result.get("quality_metrics", {}).get("peer_review_score", 0) 
                     for result in analysis_results 
                     if result.get("plan_id") == evidence.source_analysis),
                    0
                ) / 5.0  # Normalize to 0-1 scale
                
                base_score += source_quality * 0.1
                strength_scores.append(min(1.0, base_score))
            
            evidence_map["evidence_strength_matrix"][rec.recommendation_id] = {
                "average_strength": sum(strength_scores) / len(strength_scores) if strength_scores else 0,
                "evidence_count": len(strength_scores),
                "strength_distribution": {
                    "strong": len([s for s in strength_scores if s >= 0.8]),
                    "moderate": len([s for s in strength_scores if 0.6 <= s < 0.8]),
                    "weak": len([s for s in strength_scores if s < 0.6])
                }
            }
        
        # Create traceability links
        for rec in synthesis_result.business_recommendations:
            for evidence in rec.evidence_chain:
                if evidence.source_analysis:
                    evidence_map["traceability_links"].append({
                        "recommendation_id": rec.recommendation_id,
                        "evidence_level": evidence.level,
                        "evidence_statement": evidence.statement[:100] + "..." if len(evidence.statement) > 100 else evidence.statement,
                        "source_analysis": evidence.source_analysis,
                        "confidence": evidence.confidence
                    })
        
        return evidence_map
    
    def _calculate_confidence(
        self, 
        parsed_response: Dict[str, Any], 
        input_data: SynthesizerInput
    ) -> float:
        """Calculate confidence score based on synthesis quality."""
        score = 0.4  # Base score
        
        # Check synthesis overview quality
        synthesis_overview = parsed_response.get("synthesis_overview", {})
        if synthesis_overview.get("confidence_level", 0) >= 0.7:
            score += 0.1
        
        if synthesis_overview.get("evidence_strength") == "strong":
            score += 0.1
        
        # Check recommendation quality
        recommendations = parsed_response.get("business_recommendations", [])
        if not recommendations:
            return 0.2
        
        # Check for detailed evidence chains
        avg_evidence_chain_length = sum(
            len(rec.get("evidence_chain", [])) for rec in recommendations
        ) / len(recommendations)
        
        if avg_evidence_chain_length >= 3:
            score += 0.1
        
        # Check for quantified impact estimates
        has_quantified_impacts = all(
            rec.get("impact_estimate", {}).get("financial_impact") or 
            rec.get("impact_estimate", {}).get("target")
            for rec in recommendations
        )
        if has_quantified_impacts:
            score += 0.1
        
        # Check for implementation details
        has_implementation_details = all(
            rec.get("implementation", {}).get("timeline") and
            rec.get("implementation", {}).get("resource_requirements")
            for rec in recommendations
        )
        if has_implementation_details:
            score += 0.1
        
        # Check for risk assessment
        has_risk_assessment = all(
            len(rec.get("risks", [])) > 0 for rec in recommendations
        )
        if has_risk_assessment:
            score += 0.05
        
        # Check executive summary completeness
        exec_summary = parsed_response.get("executive_summary", {})
        exec_completeness = len([
            key for key in ["situation", "key_findings", "recommended_actions", "expected_outcomes"]
            if exec_summary.get(key)
        ]) / 4
        
        score += exec_completeness * 0.05
        
        return min(1.0, max(0.0, score))