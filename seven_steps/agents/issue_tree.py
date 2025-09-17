"""
Issue Tree Agent implementation.

This agent takes structured problem frames and generates MECE 
(Mutually Exclusive, Collectively Exhaustive) issue trees with 
testable hypotheses for each leaf node.
"""

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from seven_steps.core.schemas import IssueTree, IssueNode
from .base import AgentContext, AgentInput, AgentOutput, BaseAgent


class IssueTreeInput(AgentInput):
    """Input model for Issue Tree agent."""
    
    problem_frame: Dict[str, Any] = Field(..., description="Structured problem frame from Problem Framer")
    max_depth: int = Field(default=3, description="Maximum depth of issue tree")
    focus_areas: Optional[List[str]] = Field(
        default=None, 
        description="Specific areas to focus on (optional)"
    )


class IssueTreeOutput(AgentOutput):
    """Output model for Issue Tree agent."""
    
    issue_tree: Optional[IssueTree] = None
    visualization_data: Optional[Dict[str, Any]] = None  # For d3.js rendering


class IssueTreeAgent(BaseAgent):
    """
    Issue Tree Agent for MECE problem decomposition.
    
    This agent decomposes complex problems into structured, mutually exclusive
    and collectively exhaustive sub-problems with testable hypotheses.
    """
    
    @property
    def agent_name(self) -> str:
        """Return the name of this agent."""
        return "IssueTreeAgent"
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return """You are a senior strategy consultant expert in MECE problem decomposition using issue trees. Your role is to break down complex business problems into structured, analyzable components.

RESPONSIBILITIES:
1. Decompose the main problem into mutually exclusive sub-problems
2. Ensure sub-problems are collectively exhaustive (cover all aspects)
3. Create testable hypotheses for each leaf node
4. Identify data requirements for testing each hypothesis
5. Structure the tree with logical parent-child relationships
6. Limit depth to maintain actionability

MECE PRINCIPLES:
- Mutually Exclusive: No overlap between categories at the same level
- Collectively Exhaustive: All relevant aspects are covered
- Action-oriented: Each leaf should be testable/analyzable
- Hypothesis-driven: Each node contains specific, testable assumptions

OUTPUT REQUIREMENTS:
- Return valid JSON matching the IssueTree schema
- Each leaf node must have specific testable hypotheses
- Include data requirements for each hypothesis
- Suggest analysis methods where relevant
- Ensure logical flow from root to leaves

DECOMPOSITION APPROACH:
1. Start with the main problem goal
2. Identify primary categories (usually 3-5)
3. Break each category into sub-components
4. Continue until hypotheses become testable
5. Validate MECE at each level

Common decomposition frameworks:
- Revenue = Customers × Revenue per Customer
- Profitability = Revenue - Costs
- Customer Satisfaction = Product Quality × Service Quality × Value Perception
- Market Entry = Market Attractiveness × Competitive Position"""
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class."""
        return IssueTreeInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class."""
        return IssueTreeOutput
    
    async def process(self, input_data: IssueTreeInput) -> IssueTreeOutput:
        """
        Process problem frame into structured issue tree.
        
        Args:
            input_data: Problem frame and decomposition parameters
            
        Returns:
            IssueTreeOutput with MECE issue tree and visualization data
        """
        try:
            # Build context for decomposition
            context = self._build_decomposition_context(input_data)
            
            # Create the analysis prompt
            user_message = self._create_decomposition_prompt(input_data)
            
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
            
            # Create issue tree structure
            issue_tree = self._create_issue_tree(parsed_response, input_data)
            
            # Generate visualization data for d3.js
            visualization_data = self._create_visualization_data(issue_tree)
            
            return IssueTreeOutput(
                success=True,
                issue_tree=issue_tree,
                visualization_data=visualization_data,
                confidence_score=self._calculate_confidence(parsed_response, input_data)
            )
                
        except Exception as e:
            self.logger.error(
                "Issue tree generation failed",
                error=str(e),
                problem_id=input_data.context.problem_id
            )
            
            return IssueTreeOutput(
                success=False,
                error_message=f"Issue tree generation failed: {str(e)}"
            )
    
    def _build_decomposition_context(self, input_data: IssueTreeInput) -> Dict[str, Any]:
        """Build context information for problem decomposition."""
        return {
            "problem_goal": input_data.problem_frame.get("goal"),
            "scope": input_data.problem_frame.get("scope", {}),
            "kpis": input_data.problem_frame.get("kpis", []),
            "stakeholders": input_data.problem_frame.get("stakeholders", []),
            "max_depth": input_data.max_depth,
            "focus_areas": input_data.focus_areas or []
        }
    
    def _create_decomposition_prompt(self, input_data: IssueTreeInput) -> str:
        """Create the main decomposition prompt for the LLM."""
        problem_frame = input_data.problem_frame
        
        prompt = f"""Based on the following structured problem frame, create a MECE issue tree:

PROBLEM FRAME:
Goal: {problem_frame.get('goal')}
Scope: {json.dumps(problem_frame.get('scope', {}), indent=2)}
KPIs: {json.dumps(problem_frame.get('kpis', []), indent=2)}
Stakeholders: {problem_frame.get('stakeholders', [])}
Assumptions: {problem_frame.get('assumptions', [])}
Constraints: {problem_frame.get('constraints', [])}
"""
        
        if input_data.focus_areas:
            prompt += f"\nFOCUS AREAS: {input_data.focus_areas}"
        
        prompt += f"""

Please create a MECE issue tree with maximum depth of {input_data.max_depth}. Return JSON with this structure:

{{
  "root": {{
    "id": "root",
    "title": "Main problem statement",
    "level": 0,
    "hypotheses": ["Main hypothesis about the problem"],
    "required_data": ["Data needed to validate"],
    "analysis_methods": ["Suggested analysis approaches"],
    "children": [
      {{
        "id": "1",
        "title": "First major category", 
        "level": 1,
        "hypotheses": ["Specific testable hypothesis"],
        "required_data": ["Specific data sources"],
        "analysis_methods": ["Cohort analysis", "Statistical testing"],
        "priority_hint": "high|medium|low",
        "children": [
          {{
            "id": "1.1",
            "title": "Sub-component",
            "level": 2,
            "hypotheses": ["Very specific testable hypothesis"],
            "required_data": ["Exact data requirements"],
            "analysis_methods": ["Specific analytical method"],
            "priority_hint": "high|medium|low"
          }}
        ]
      }}
    ]
  }}
}}

REQUIREMENTS:
1. Each level must be mutually exclusive and collectively exhaustive
2. Each leaf node must have specific, testable hypotheses
3. Include realistic data requirements and analysis methods
4. Use business logic appropriate for the problem domain
5. Ensure hypotheses are specific enough to be validated/refuted"""
        
        return prompt
    
    def _get_additional_context(self, input_data: IssueTreeInput) -> str:
        """Generate additional context string."""
        context_parts = []
        
        problem_frame = input_data.problem_frame
        
        # Extract industry context if available
        if "metadata" in problem_frame:
            metadata = problem_frame["metadata"]
            if metadata.get("industry"):
                context_parts.append(f"Industry: {metadata['industry']}")
            if metadata.get("business_model"):
                context_parts.append(f"Business Model: {metadata['business_model']}")
        
        # Add constraint context
        constraints = problem_frame.get("constraints", [])
        if constraints:
            context_parts.append(f"Key Constraints: {', '.join(constraints[:3])}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _create_issue_tree(self, parsed_response: Dict[str, Any], input_data: IssueTreeInput) -> IssueTree:
        """Create structured issue tree from parsed response."""
        root_data = parsed_response.get("root", {})
        
        def build_node(node_data: Dict[str, Any]) -> IssueNode:
            """Recursively build issue tree nodes."""
            node = IssueNode(
                id=node_data.get("id", ""),
                title=node_data.get("title", ""),
                level=node_data.get("level", 0),
                hypotheses=node_data.get("hypotheses", []),
                required_data=node_data.get("required_data", []),
                analysis_methods=node_data.get("analysis_methods", []),
                priority_hint=node_data.get("priority_hint", "medium"),
                children=[]
            )
            
            # Recursively build children
            for child_data in node_data.get("children", []):
                child_node = build_node(child_data)
                node.children.append(child_node)
            
            return node
        
        root_node = build_node(root_data)
        
        return IssueTree(
            problem_id=input_data.context.problem_id,
            root=root_node,
            max_depth=input_data.max_depth,
            total_nodes=self._count_nodes(root_node)
        )
    
    def _count_nodes(self, node: IssueNode) -> int:
        """Count total nodes in the tree."""
        count = 1  # Count current node
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def _create_visualization_data(self, issue_tree: IssueTree) -> Dict[str, Any]:
        """Create data structure for d3.js tree visualization."""
        
        def node_to_viz_data(node: IssueNode) -> Dict[str, Any]:
            """Convert issue node to visualization format."""
            viz_node = {
                "id": node.id,
                "name": node.title,
                "level": node.level,
                "hypotheses_count": len(node.hypotheses),
                "data_requirements_count": len(node.required_data),
                "priority": node.priority_hint,
                "has_children": len(node.children) > 0
            }
            
            if node.children:
                viz_node["children"] = [node_to_viz_data(child) for child in node.children]
            
            return viz_node
        
        return {
            "tree_data": node_to_viz_data(issue_tree.root),
            "metadata": {
                "total_nodes": issue_tree.total_nodes,
                "max_depth": issue_tree.max_depth,
                "problem_id": issue_tree.problem_id
            }
        }
    
    def _calculate_confidence(
        self, 
        parsed_response: Dict[str, Any], 
        input_data: IssueTreeInput
    ) -> float:
        """Calculate confidence score based on tree quality."""
        score = 0.5  # Base score
        
        root = parsed_response.get("root", {})
        
        # Check for good MECE structure
        children = root.get("children", [])
        if len(children) >= 3:  # Good decomposition has 3+ main categories
            score += 0.1
        
        # Check for specific hypotheses
        total_hypotheses = 0
        total_nodes = 0
        
        def count_hypotheses(node_data: Dict[str, Any]):
            nonlocal total_hypotheses, total_nodes
            total_nodes += 1
            hypotheses = node_data.get("hypotheses", [])
            total_hypotheses += len(hypotheses)
            
            for child in node_data.get("children", []):
                count_hypotheses(child)
        
        count_hypotheses(root)
        
        if total_nodes > 0:
            avg_hypotheses_per_node = total_hypotheses / total_nodes
            if avg_hypotheses_per_node >= 1.5:  # Good hypothesis coverage
                score += 0.2
        
        # Check for data requirements
        has_data_requirements = self._check_data_requirements(root)
        if has_data_requirements:
            score += 0.1
        
        # Check for analysis methods
        has_analysis_methods = self._check_analysis_methods(root)
        if has_analysis_methods:
            score += 0.1
        
        return min(1.0, max(0.0, score))
    
    def _check_data_requirements(self, node_data: Dict[str, Any]) -> bool:
        """Check if nodes have meaningful data requirements."""
        required_data = node_data.get("required_data", [])
        if required_data and any(len(req) > 5 for req in required_data):
            return True
        
        for child in node_data.get("children", []):
            if self._check_data_requirements(child):
                return True
        
        return False
    
    def _check_analysis_methods(self, node_data: Dict[str, Any]) -> bool:
        """Check if nodes have suggested analysis methods."""
        methods = node_data.get("analysis_methods", [])
        if methods and any(len(method) > 5 for method in methods):
            return True
        
        for child in node_data.get("children", []):
            if self._check_analysis_methods(child):
                return True
        
        return False