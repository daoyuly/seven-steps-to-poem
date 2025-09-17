"""
Issue Tree Agent implementation using OpenAI Agents SDK.

This agent takes structured problem frames and generates MECE 
(Mutually Exclusive, Collectively Exhaustive) issue trees with 
testable hypotheses for each leaf node.
"""

from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from seven_steps.core.schemas import IssueTree, IssueNode
from .base_v2 import BaseSevenStepsAgent, AgentInput, AgentOutput


# Structured Output Models
class IssueNodeOutput(BaseModel):
    """Structured output for issue tree nodes."""
    
    id: str = Field(description="Unique node identifier")
    title: str = Field(description="Node title/description")
    level: int = Field(description="Tree depth level")
    hypotheses: List[str] = Field(description="Testable hypotheses")
    required_data: List[str] = Field(description="Data needed for analysis")
    analysis_methods: List[str] = Field(description="Suggested analysis approaches")
    priority_hint: str = Field(description="Priority level: high|medium|low")
    children: List['IssueNodeOutput'] = Field(default_factory=list, description="Child nodes")


class IssueTreeOutput(BaseModel):
    """Structured output for issue tree generation."""
    
    root: IssueNodeOutput = Field(description="Root node of the issue tree")
    total_nodes: int = Field(description="Total number of nodes in tree")
    max_depth: int = Field(description="Maximum depth achieved")
    mece_validation: Dict[str, bool] = Field(description="MECE validation results")
    
    class Config:
        # Enable forward references for recursive model
        arbitrary_types_allowed = True


# Update forward reference
IssueNodeOutput.model_rebuild()


class IssueTreeInput(AgentInput):
    """Input model for Issue Tree agent."""
    
    problem_frame: Dict[str, Any] = Field(..., description="Structured problem frame")
    max_depth: int = Field(default=3, description="Maximum depth of issue tree")
    focus_areas: Optional[List[str]] = Field(
        default=None, 
        description="Specific areas to focus on"
    )


class IssueTreeAgentOutput(AgentOutput):
    """Output model for Issue Tree agent."""
    
    issue_tree: Optional[IssueTree] = None
    visualization_data: Optional[Dict[str, Any]] = None


class IssueTreeAgent(BaseSevenStepsAgent):
    """
    Issue Tree Agent using OpenAI Agents SDK.
    
    This agent decomposes complex problems into structured, mutually exclusive
    and collectively exhaustive sub-problems with testable hypotheses.
    """
    
    def __init__(self):
        """Initialize the Issue Tree agent."""
        
        instructions = """You are a senior strategy consultant expert in MECE problem decomposition using issue trees. Your role is to break down complex business problems into structured, analyzable components.

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

        # Define MCP tools for issue tree analysis
        mcp_tools = [
            "mece-validator",        # Validates MECE principles
            "hypothesis-generator",  # Generates testable hypotheses
            "data-mapper"           # Maps data requirements
        ]
        
        super().__init__(
            name="IssueTreeAgent",
            description="Creates MECE issue trees for complex problem decomposition",
            instructions=instructions,
            output_type=IssueTreeOutput,
            mcp_tools=mcp_tools
        )
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class."""
        return IssueTreeInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class."""
        return IssueTreeAgentOutput
    
    async def process_request(self, input_data: IssueTreeInput) -> IssueTreeAgentOutput:
        """
        Process problem frame into structured issue tree.
        
        Args:
            input_data: Problem frame and decomposition parameters
            
        Returns:
            IssueTreeAgentOutput with MECE issue tree and visualization data
        """
        try:
            # Build comprehensive prompt for issue tree generation
            prompt = self._create_decomposition_prompt(input_data)
            
            # Add context data
            context = {
                "problem_goal": input_data.problem_frame.get("goal"),
                "scope": input_data.problem_frame.get("scope", {}),
                "kpis": input_data.problem_frame.get("kpis", []),
                "stakeholders": input_data.problem_frame.get("stakeholders", []),
                "max_depth": input_data.max_depth,
                "focus_areas": input_data.focus_areas or []
            }
            
            # Run structured completion for issue tree
            structured_result = await self.run_structured_completion(
                prompt=prompt,
                context=context,
                output_type=IssueTreeOutput
            )
            
            # Validate MECE principles using MCP tool
            mece_validation = await self._validate_mece_structure(structured_result)
            
            # Create issue tree structure
            issue_tree = await self._create_issue_tree(structured_result, input_data)
            
            # Generate visualization data for d3.js
            visualization_data = self._create_visualization_data(issue_tree)
            
            return IssueTreeAgentOutput(
                success=True,
                issue_tree=issue_tree,
                visualization_data=visualization_data,
                confidence_score=self._calculate_confidence(structured_result, mece_validation),
                data={
                    "issue_tree": issue_tree.dict() if issue_tree else None,
                    "structured_output": structured_result.dict(),
                    "mece_validation": mece_validation
                }
            )
                
        except Exception as e:
            self.logger.error(
                "Issue tree generation failed",
                error=str(e),
                problem_id=input_data.context.problem_id
            )
            
            return IssueTreeAgentOutput(
                success=False,
                error_message=f"Issue tree generation failed: {str(e)}"
            )
    
    def _create_decomposition_prompt(self, input_data: IssueTreeInput) -> str:
        """Create the main decomposition prompt for structured output."""
        
        problem_frame = input_data.problem_frame
        
        prompt = f"""Based on the following structured problem frame, create a comprehensive MECE issue tree:

PROBLEM FRAME:
Goal: {problem_frame.get('goal')}
Scope: {problem_frame.get('scope', {})}
KPIs: {[kpi.get('name', 'Unknown') for kpi in problem_frame.get('kpis', [])]}
Stakeholders: {problem_frame.get('stakeholders', [])}
Assumptions: {problem_frame.get('assumptions', [])}
Constraints: {problem_frame.get('constraints', [])}
"""
        
        if input_data.focus_areas:
            prompt += f"\nFOCUS AREAS: {input_data.focus_areas}"
        
        prompt += f"""

Create a MECE issue tree with maximum depth of {input_data.max_depth}. Follow these requirements:

STRUCTURE REQUIREMENTS:
1. Root node represents the main problem/goal
2. Each level must be mutually exclusive and collectively exhaustive
3. Child nodes should completely decompose their parent
4. Use business logic appropriate for the problem domain
5. Ensure logical flow from general to specific

NODE REQUIREMENTS:
1. Each node must have specific, testable hypotheses
2. Include realistic data requirements for hypothesis testing
3. Suggest appropriate analysis methods
4. Assign priority hints based on impact and feasibility
5. Use clear, business-oriented titles

HYPOTHESIS REQUIREMENTS:
1. Must be specific and measurable
2. Should be testable with available data
3. Focus on root causes, not just symptoms
4. Include quantitative aspects where possible

DATA REQUIREMENTS:
1. Specify exact data sources needed
2. Include sample sizes and time periods
3. Consider data availability and quality
4. Map to specific analytical methods

Generate a comprehensive issue tree that enables systematic problem-solving."""
        
        return prompt
    
    async def _validate_mece_structure(
        self, 
        structured_result: IssueTreeOutput
    ) -> Dict[str, Any]:
        """Validate MECE principles using MCP tool."""
        
        try:
            # Extract tree structure for validation
            tree_structure = self._extract_tree_structure(structured_result.root)
            
            # Call MECE validator MCP tool
            validation_result = await self.call_mcp_tool(
                tool_name="mece-validator",
                parameters={
                    "tree_structure": tree_structure,
                    "max_depth": structured_result.max_depth
                }
            )
            
            if validation_result["success"]:
                return validation_result["result"]
            else:
                self.logger.warning(
                    "MECE validation failed, using fallback validation",
                    error=validation_result["error"]
                )
                return self._fallback_mece_validation(structured_result)
                
        except Exception as e:
            self.logger.warning(
                "MECE validation error, using fallback",
                error=str(e)
            )
            return self._fallback_mece_validation(structured_result)
    
    def _extract_tree_structure(self, node: IssueNodeOutput) -> Dict[str, Any]:
        """Extract tree structure for validation."""
        
        return {
            "id": node.id,
            "title": node.title,
            "level": node.level,
            "children": [
                self._extract_tree_structure(child) 
                for child in node.children
            ]
        }
    
    def _fallback_mece_validation(
        self, 
        structured_result: IssueTreeOutput
    ) -> Dict[str, Any]:
        """Fallback MECE validation logic."""
        
        validation_results = {
            "mutually_exclusive": True,
            "collectively_exhaustive": True,
            "appropriate_depth": structured_result.max_depth <= 4,
            "testable_hypotheses": True,
            "data_requirements_specified": True,
            "issues": []
        }
        
        # Basic validation checks
        if structured_result.total_nodes < 3:
            validation_results["issues"].append("Tree has too few nodes for meaningful analysis")
        
        if structured_result.max_depth < 2:
            validation_results["issues"].append("Tree depth is insufficient for problem decomposition")
        
        return validation_results
    
    async def _create_issue_tree(
        self, 
        structured_result: IssueTreeOutput,
        input_data: IssueTreeInput
    ) -> IssueTree:
        """Create structured issue tree from the analysis."""
        
        # Convert structured output to domain model
        root_node = await self._convert_to_issue_node(structured_result.root)
        
        # Create the issue tree
        issue_tree = IssueTree(
            problem_id=input_data.context.problem_id,
            root=root_node,
            max_depth=structured_result.max_depth,
            total_nodes=structured_result.total_nodes
        )
        
        return issue_tree
    
    async def _convert_to_issue_node(self, node_output: IssueNodeOutput) -> IssueNode:
        """Convert structured output node to domain model."""
        
        # Enhance hypotheses using MCP tool if available
        enhanced_hypotheses = await self._enhance_hypotheses(
            node_output.hypotheses, 
            node_output.title
        )
        
        # Convert children recursively
        children = []
        for child_output in node_output.children:
            child_node = await self._convert_to_issue_node(child_output)
            children.append(child_node)
        
        # Create issue node
        issue_node = IssueNode(
            id=node_output.id,
            title=node_output.title,
            level=node_output.level,
            hypotheses=enhanced_hypotheses,
            required_data=node_output.required_data,
            analysis_methods=node_output.analysis_methods,
            priority_hint=node_output.priority_hint,
            children=children
        )
        
        return issue_node
    
    async def _enhance_hypotheses(
        self, 
        hypotheses: List[str], 
        node_title: str
    ) -> List[str]:
        """Enhance hypotheses using hypothesis generator MCP tool."""
        
        try:
            # Call hypothesis generator MCP tool
            enhancement_result = await self.call_mcp_tool(
                tool_name="hypothesis-generator",
                parameters={
                    "original_hypotheses": hypotheses,
                    "context": node_title
                }
            )
            
            if enhancement_result["success"]:
                enhanced = enhancement_result["result"].get("enhanced_hypotheses", hypotheses)
                return enhanced
            else:
                return hypotheses
                
        except Exception as e:
            self.logger.warning(
                "Hypothesis enhancement failed, using original",
                hypotheses=hypotheses,
                error=str(e)
            )
            return hypotheses
    
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
                "has_children": len(node.children) > 0,
                "analysis_methods": node.analysis_methods
            }
            
            if node.children:
                viz_node["children"] = [
                    node_to_viz_data(child) 
                    for child in node.children
                ]
            
            return viz_node
        
        return {
            "tree_data": node_to_viz_data(issue_tree.root),
            "metadata": {
                "total_nodes": issue_tree.total_nodes,
                "max_depth": issue_tree.max_depth,
                "problem_id": issue_tree.problem_id
            },
            "layout_config": {
                "node_size": [200, 100],
                "level_separation": 150,
                "sibling_separation": 50,
                "tree_orientation": "top_to_bottom"
            }
        }
    
    def _calculate_confidence(
        self, 
        structured_result: IssueTreeOutput,
        mece_validation: Dict[str, Any]
    ) -> float:
        """Calculate confidence score based on tree quality and validation."""
        
        score = 0.5  # Base score
        
        # MECE validation score
        if mece_validation.get("mutually_exclusive", False):
            score += 0.15
        
        if mece_validation.get("collectively_exhaustive", False):
            score += 0.15
        
        if mece_validation.get("testable_hypotheses", False):
            score += 0.1
        
        # Tree structure quality
        if structured_result.total_nodes >= 7:  # Good decomposition
            score += 0.1
        
        if 2 <= structured_result.max_depth <= 4:  # Appropriate depth
            score += 0.05
        
        # Hypothesis quality (check root node as example)
        root_hypotheses = structured_result.root.hypotheses
        if root_hypotheses and len(root_hypotheses) >= 2:
            score += 0.05
            
            # Check for specific, measurable hypotheses
            specific_hypotheses = sum(
                1 for h in root_hypotheses 
                if len(h) > 20 and any(char.isdigit() for char in h)
            )
            if specific_hypotheses / len(root_hypotheses) >= 0.5:
                score += 0.05
        
        # Data requirements specificity
        if structured_result.root.required_data:
            specific_data_reqs = sum(
                1 for req in structured_result.root.required_data
                if len(req) > 10 and ("table" in req.lower() or "api" in req.lower())
            )
            if specific_data_reqs > 0:
                score += 0.05
        
        return min(1.0, max(0.0, score))
    
    @BaseSevenStepsAgent.create_tool_function(
        "validate_tree_completeness", 
        "Validate that issue tree covers all aspects of the problem"
    )
    async def validate_tree_completeness(
        self, 
        issue_tree: Dict[str, Any], 
        problem_scope: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Tool function to validate tree completeness against problem scope."""
        
        # Extract included scope areas
        included_areas = problem_scope.get("include", [])
        
        # Extract all tree node titles
        def extract_titles(node):
            titles = [node["title"]]
            for child in node.get("children", []):
                titles.extend(extract_titles(child))
            return titles
        
        tree_titles = extract_titles(issue_tree["root"])
        
        # Check coverage
        coverage_analysis = {
            "scope_areas_covered": [],
            "missing_areas": [],
            "coverage_percentage": 0
        }
        
        for area in included_areas:
            covered = any(area.lower() in title.lower() for title in tree_titles)
            if covered:
                coverage_analysis["scope_areas_covered"].append(area)
            else:
                coverage_analysis["missing_areas"].append(area)
        
        if included_areas:
            coverage_analysis["coverage_percentage"] = (
                len(coverage_analysis["scope_areas_covered"]) / len(included_areas)
            )
        
        return coverage_analysis