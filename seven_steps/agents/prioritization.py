"""
Prioritization Agent implementation.

This agent evaluates issue tree leaf nodes using impact × feasibility
scoring to determine analysis priorities and resource allocation.
"""

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from seven_steps.core.schemas import PriorityMatrix, PriorityScore
from .base import AgentContext, AgentInput, AgentOutput, BaseAgent


class PrioritizationInput(AgentInput):
    """Input model for Prioritization agent."""
    
    issue_tree: Dict[str, Any] = Field(..., description="Issue tree from previous step")
    scoring_criteria: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom weights for scoring criteria"
    )
    resource_constraints: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Available resources and constraints"
    )


class PrioritizationOutput(AgentOutput):
    """Output model for Prioritization agent."""
    
    priority_matrix: Optional[PriorityMatrix] = None
    recommended_sequence: List[str] = Field(default_factory=list)  # Ordered list of node IDs
    resource_allocation: Optional[Dict[str, Any]] = None


class PrioritizationAgent(BaseAgent):
    """
    Prioritization Agent for impact × feasibility analysis.
    
    This agent scores issue tree leaf nodes based on business impact,
    implementation feasibility, data availability, and resource requirements.
    """
    
    @property
    def agent_name(self) -> str:
        """Return the name of this agent."""
        return "PrioritizationAgent"
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return """You are a senior business analyst expert in priority matrix analysis and resource allocation. Your role is to evaluate and prioritize analysis tasks based on impact and feasibility.

RESPONSIBILITIES:
1. Score each leaf node on impact (1-10) and feasibility (1-10)
2. Calculate priority scores using weighted criteria
3. Provide detailed reasoning for each score
4. Consider resource constraints and data availability
5. Recommend execution sequence based on dependencies
6. Estimate effort and timeline for each analysis

SCORING CRITERIA:

IMPACT (1-10):
- Business value potential (revenue, cost savings, risk reduction)
- Stakeholder importance and urgency
- Strategic alignment with company goals
- Market/competitive implications
- Customer satisfaction impact

FEASIBILITY (1-10):
- Data availability and quality
- Technical complexity of analysis
- Required skills and expertise
- Time to complete analysis
- Resource requirements (people, tools, budget)

PRIORITIZATION LOGIC:
- High Impact + High Feasibility = Quick Wins (Priority 1)
- High Impact + Low Feasibility = Major Projects (Priority 2)
- Low Impact + High Feasibility = Fill-ins (Priority 3)
- Low Impact + Low Feasibility = Questionable (Priority 4)

OUTPUT REQUIREMENTS:
- Return valid JSON with detailed scoring and reasoning
- Include confidence levels for each assessment
- Provide resource estimates (hours, skills needed)
- Consider dependencies between analyses
- Suggest parallel vs sequential execution"""
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class."""
        return PrioritizationInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class."""
        return PrioritizationOutput
    
    async def process(self, input_data: PrioritizationInput) -> PrioritizationOutput:
        """
        Process issue tree into prioritized analysis plan.
        
        Args:
            input_data: Issue tree and scoring parameters
            
        Returns:
            PrioritizationOutput with priority matrix and recommendations
        """
        try:
            # Extract leaf nodes from issue tree
            leaf_nodes = self._extract_leaf_nodes(input_data.issue_tree)
            
            # Build context for scoring
            context = self._build_scoring_context(input_data, leaf_nodes)
            
            # Create the analysis prompt
            user_message = self._create_scoring_prompt(input_data, leaf_nodes)
            
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
            
            # Create priority matrix
            priority_matrix = self._create_priority_matrix(parsed_response, leaf_nodes, input_data)
            
            # Generate recommended sequence
            recommended_sequence = self._generate_execution_sequence(priority_matrix)
            
            # Calculate resource allocation
            resource_allocation = self._calculate_resource_allocation(
                priority_matrix, input_data.resource_constraints
            )
            
            return PrioritizationOutput(
                success=True,
                priority_matrix=priority_matrix,
                recommended_sequence=recommended_sequence,
                resource_allocation=resource_allocation,
                confidence_score=self._calculate_confidence(parsed_response, input_data)
            )
                
        except Exception as e:
            self.logger.error(
                "Prioritization analysis failed",
                error=str(e),
                problem_id=input_data.context.problem_id
            )
            
            return PrioritizationOutput(
                success=False,
                error_message=f"Prioritization analysis failed: {str(e)}"
            )
    
    def _extract_leaf_nodes(self, issue_tree: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract all leaf nodes from the issue tree."""
        leaf_nodes = []
        
        def traverse_tree(node: Dict[str, Any], path: List[str] = None):
            if path is None:
                path = []
            
            current_path = path + [node.get("id", "")]
            
            # If no children, it's a leaf node
            children = node.get("children", [])
            if not children:
                leaf_node = node.copy()
                leaf_node["path"] = current_path
                leaf_node["full_title"] = " > ".join([
                    n.get("title", "") for n in self._get_node_path(issue_tree, current_path)
                ])
                leaf_nodes.append(leaf_node)
            else:
                # Continue traversing children
                for child in children:
                    traverse_tree(child, current_path)
        
        root = issue_tree.get("root", {})
        if root:
            traverse_tree(root)
        
        return leaf_nodes
    
    def _get_node_path(self, issue_tree: Dict[str, Any], path: List[str]) -> List[Dict[str, Any]]:
        """Get the full path of nodes from root to leaf."""
        nodes_path = []
        current_node = issue_tree.get("root", {})
        nodes_path.append(current_node)
        
        for node_id in path[1:]:  # Skip root
            for child in current_node.get("children", []):
                if child.get("id") == node_id:
                    current_node = child
                    nodes_path.append(current_node)
                    break
        
        return nodes_path
    
    def _build_scoring_context(self, input_data: PrioritizationInput, leaf_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build context information for scoring."""
        return {
            "total_leaf_nodes": len(leaf_nodes),
            "scoring_criteria": input_data.scoring_criteria or {
                "impact_weight": 0.6,
                "feasibility_weight": 0.4
            },
            "resource_constraints": input_data.resource_constraints or {},
            "problem_id": input_data.context.problem_id
        }
    
    def _create_scoring_prompt(self, input_data: PrioritizationInput, leaf_nodes: List[Dict[str, Any]]) -> str:
        """Create the main scoring prompt for the LLM."""
        prompt = f"""Analyze and prioritize the following {len(leaf_nodes)} leaf nodes from an issue tree. 
Each node represents a specific analysis that needs to be conducted.

LEAF NODES TO SCORE:
"""
        
        for i, node in enumerate(leaf_nodes, 1):
            prompt += f"""
{i}. ID: {node.get('id')}
   Title: {node.get('title')}
   Full Path: {node.get('full_title')}
   Hypotheses: {node.get('hypotheses', [])}
   Required Data: {node.get('required_data', [])}
   Analysis Methods: {node.get('analysis_methods', [])}
   Current Priority Hint: {node.get('priority_hint', 'medium')}
"""
        
        if input_data.resource_constraints:
            prompt += f"\nRESOURCE CONSTRAINTS:\n{json.dumps(input_data.resource_constraints, indent=2)}"
        
        if input_data.scoring_criteria:
            prompt += f"\nSCORING WEIGHTS:\n{json.dumps(input_data.scoring_criteria, indent=2)}"
        
        prompt += """

Please score each node and return JSON in this format:

{
  "priority_scores": [
    {
      "node_id": "1.1",
      "title": "Node title",
      "impact_score": 8,
      "impact_reasoning": "Detailed explanation of impact score",
      "feasibility_score": 6, 
      "feasibility_reasoning": "Detailed explanation of feasibility score",
      "priority_category": "quick_win|major_project|fill_in|questionable",
      "estimated_effort_hours": 40,
      "required_skills": ["data_analysis", "sql", "statistics"],
      "data_confidence": "high|medium|low",
      "dependencies": ["1.2", "2.1"],
      "risk_factors": ["Limited data quality", "External stakeholder dependency"],
      "confidence_level": 0.8
    }
  ],
  "overall_recommendations": {
    "quick_wins": ["node_ids for immediate execution"],
    "major_projects": ["node_ids requiring significant investment"],
    "suggested_parallel_analyses": [["node_id1", "node_id2"]],
    "critical_path": ["ordered sequence of dependent nodes"]
  }
}

SCORING GUIDELINES:
Impact (1-10): Consider business value, stakeholder importance, strategic alignment
Feasibility (1-10): Consider data availability, technical complexity, resource requirements
Priority Categories:
- Quick Win: High Impact (7-10) + High Feasibility (7-10)
- Major Project: High Impact (7-10) + Low Feasibility (1-6) 
- Fill-in: Low Impact (1-6) + High Feasibility (7-10)
- Questionable: Low Impact (1-6) + Low Feasibility (1-6)"""
        
        return prompt
    
    def _get_additional_context(self, input_data: PrioritizationInput) -> str:
        """Generate additional context string."""
        context_parts = []
        
        # Add resource context
        if input_data.resource_constraints:
            constraints = input_data.resource_constraints
            if constraints.get("timeline"):
                context_parts.append(f"Timeline: {constraints['timeline']}")
            if constraints.get("team_size"):
                context_parts.append(f"Team Size: {constraints['team_size']}")
            if constraints.get("budget"):
                context_parts.append(f"Budget: {constraints['budget']}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _create_priority_matrix(
        self, 
        parsed_response: Dict[str, Any], 
        leaf_nodes: List[Dict[str, Any]], 
        input_data: PrioritizationInput
    ) -> PriorityMatrix:
        """Create priority matrix from parsed response."""
        priority_scores_data = parsed_response.get("priority_scores", [])
        priority_scores = []
        
        for score_data in priority_scores_data:
            priority_score = PriorityScore(
                node_id=score_data.get("node_id", ""),
                title=score_data.get("title", ""),
                impact_score=score_data.get("impact_score", 5),
                impact_reasoning=score_data.get("impact_reasoning", ""),
                feasibility_score=score_data.get("feasibility_score", 5),
                feasibility_reasoning=score_data.get("feasibility_reasoning", ""),
                priority_category=score_data.get("priority_category", "fill_in"),
                estimated_effort_hours=score_data.get("estimated_effort_hours", 20),
                required_skills=score_data.get("required_skills", []),
                data_confidence=score_data.get("data_confidence", "medium"),
                dependencies=score_data.get("dependencies", []),
                risk_factors=score_data.get("risk_factors", []),
                confidence_level=score_data.get("confidence_level", 0.7)
            )
            priority_scores.append(priority_score)
        
        # Calculate weighted priority scores
        weights = input_data.scoring_criteria or {"impact_weight": 0.6, "feasibility_weight": 0.4}
        for score in priority_scores:
            weighted_score = (
                score.impact_score * weights.get("impact_weight", 0.6) +
                score.feasibility_score * weights.get("feasibility_weight", 0.4)
            )
            score.weighted_score = round(weighted_score, 2)
        
        return PriorityMatrix(
            problem_id=input_data.context.problem_id,
            priority_scores=priority_scores,
            scoring_criteria=weights,
            overall_recommendations=parsed_response.get("overall_recommendations", {})
        )
    
    def _generate_execution_sequence(self, priority_matrix: PriorityMatrix) -> List[str]:
        """Generate recommended execution sequence based on priority and dependencies."""
        scores = priority_matrix.priority_scores.copy()
        
        # Sort by weighted score (descending) and category priority
        category_weights = {
            "quick_win": 4,
            "major_project": 3, 
            "fill_in": 2,
            "questionable": 1
        }
        
        scores.sort(key=lambda x: (
            category_weights.get(x.priority_category, 0),
            x.weighted_score,
            -x.estimated_effort_hours  # Prefer shorter analyses when scores are equal
        ), reverse=True)
        
        # Consider dependencies when building sequence
        sequence = []
        completed = set()
        
        def can_execute(node_score: PriorityScore) -> bool:
            return all(dep in completed for dep in node_score.dependencies)
        
        while len(sequence) < len(scores):
            added_any = False
            
            for score in scores:
                if score.node_id not in completed and can_execute(score):
                    sequence.append(score.node_id)
                    completed.add(score.node_id)
                    added_any = True
                    break
            
            # If no dependencies can be satisfied, add highest priority remaining
            if not added_any:
                for score in scores:
                    if score.node_id not in completed:
                        sequence.append(score.node_id)
                        completed.add(score.node_id)
                        break
        
        return sequence
    
    def _calculate_resource_allocation(
        self, 
        priority_matrix: PriorityMatrix,
        resource_constraints: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate resource allocation recommendations."""
        total_effort = sum(score.estimated_effort_hours for score in priority_matrix.priority_scores)
        
        # Group by category
        by_category = {}
        for score in priority_matrix.priority_scores:
            category = score.priority_category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(score)
        
        # Calculate category statistics
        category_stats = {}
        for category, scores in by_category.items():
            category_stats[category] = {
                "count": len(scores),
                "total_hours": sum(s.estimated_effort_hours for s in scores),
                "avg_impact": sum(s.impact_score for s in scores) / len(scores),
                "avg_feasibility": sum(s.feasibility_score for s in scores) / len(scores),
                "node_ids": [s.node_id for s in scores]
            }
        
        # Required skills analysis
        all_skills = set()
        for score in priority_matrix.priority_scores:
            all_skills.update(score.required_skills)
        
        skill_demand = {}
        for skill in all_skills:
            skill_demand[skill] = sum(
                score.estimated_effort_hours 
                for score in priority_matrix.priority_scores 
                if skill in score.required_skills
            )
        
        return {
            "total_estimated_hours": total_effort,
            "category_breakdown": category_stats,
            "required_skills": list(all_skills),
            "skill_demand_hours": skill_demand,
            "timeline_estimate": {
                "quick_wins": f"{category_stats.get('quick_win', {}).get('total_hours', 0) // 40 + 1} weeks",
                "major_projects": f"{category_stats.get('major_project', {}).get('total_hours', 0) // 40 + 1} weeks",
                "total_duration": f"{total_effort // 40 + 1} weeks (single analyst)"
            },
            "parallelization_opportunities": self._identify_parallel_opportunities(priority_matrix),
            "resource_constraints_impact": resource_constraints or {}
        }
    
    def _identify_parallel_opportunities(self, priority_matrix: PriorityMatrix) -> List[List[str]]:
        """Identify analyses that can be run in parallel."""
        parallel_groups = []
        
        # Group nodes with no dependencies between them
        independent_nodes = []
        for score in priority_matrix.priority_scores:
            # Check if this node has dependencies on other nodes in our list
            has_internal_deps = any(
                dep in [s.node_id for s in priority_matrix.priority_scores]
                for dep in score.dependencies
            )
            if not has_internal_deps:
                independent_nodes.append(score.node_id)
        
        # Group independent nodes by category and feasibility
        if len(independent_nodes) > 1:
            parallel_groups.append(independent_nodes)
        
        return parallel_groups
    
    def _calculate_confidence(
        self, 
        parsed_response: Dict[str, Any], 
        input_data: PrioritizationInput
    ) -> float:
        """Calculate confidence score based on prioritization quality."""
        score = 0.5  # Base score
        
        priority_scores = parsed_response.get("priority_scores", [])
        
        # Check for detailed reasoning
        has_detailed_reasoning = all(
            len(ps.get("impact_reasoning", "")) > 20 and len(ps.get("feasibility_reasoning", "")) > 20
            for ps in priority_scores
        )
        if has_detailed_reasoning:
            score += 0.2
        
        # Check for realistic effort estimates
        effort_estimates = [ps.get("estimated_effort_hours", 0) for ps in priority_scores]
        if effort_estimates and all(10 <= est <= 200 for est in effort_estimates):
            score += 0.1
        
        # Check for good category distribution
        categories = [ps.get("priority_category", "") for ps in priority_scores]
        unique_categories = set(categories)
        if len(unique_categories) >= 2:  # Good spread across categories
            score += 0.1
        
        # Check for confidence levels in individual assessments
        confidence_levels = [ps.get("confidence_level", 0) for ps in priority_scores if ps.get("confidence_level")]
        if confidence_levels:
            avg_confidence = sum(confidence_levels) / len(confidence_levels)
            if avg_confidence >= 0.7:
                score += 0.1
        
        return min(1.0, max(0.0, score))