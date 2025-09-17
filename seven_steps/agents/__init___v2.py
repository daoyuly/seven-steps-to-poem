"""
Seven Steps Agents Module - OpenAI Agents SDK Implementation.

This module contains all the agent implementations for the McKinsey Seven Steps
methodology using OpenAI's structured outputs, tool integration, and MCP ecosystem.
"""

from .base_v2 import (
    BaseSevenStepsAgent, 
    AgentRegistry, 
    AgentInput, 
    AgentOutput, 
    AgentContext,
    get_agent_registry
)
from .problem_framer_v2 import ProblemFramer, ProblemFramerInput, ProblemFramerOutput
from .issue_tree_v2 import IssueTreeAgent, IssueTreeInput, IssueTreeAgentOutput
from .orchestrator_v2 import (
    SevenStepsOrchestrator, 
    WorkflowMode, 
    WorkflowStep,
    get_orchestrator
)

# MCP Tools Registry
MCP_TOOLS_REGISTRY = {
    # Business Analysis Tools
    "data-source-connector": {
        "description": "Connects to various business data sources",
        "capabilities": ["database_access", "api_integration", "file_processing"],
        "supported_agents": ["problem_framer", "data_analysis", "planner"]
    },
    "stakeholder-analyzer": {
        "description": "Analyzes stakeholder relationships and influence mapping", 
        "capabilities": ["stakeholder_mapping", "influence_analysis", "communication_planning"],
        "supported_agents": ["problem_framer", "synthesizer"]
    },
    "kpi-validator": {
        "description": "Validates KPI definitions and benchmarks",
        "capabilities": ["kpi_validation", "benchmark_analysis", "metric_standardization"],
        "supported_agents": ["problem_framer", "prioritization"]
    },
    
    # Issue Tree Analysis Tools
    "mece-validator": {
        "description": "Validates MECE principles in problem decomposition",
        "capabilities": ["mece_validation", "logic_checking", "completeness_analysis"],
        "supported_agents": ["issue_tree", "synthesizer"]
    },
    "hypothesis-generator": {
        "description": "Generates and enhances testable hypotheses",
        "capabilities": ["hypothesis_generation", "testability_validation", "evidence_mapping"],
        "supported_agents": ["issue_tree", "data_analysis"]
    },
    "data-mapper": {
        "description": "Maps data requirements to available sources",
        "capabilities": ["data_mapping", "source_discovery", "quality_assessment"],
        "supported_agents": ["issue_tree", "planner", "data_analysis"]
    },
    
    # Analysis and Planning Tools
    "impact-calculator": {
        "description": "Calculates business impact and ROI estimates",
        "capabilities": ["impact_modeling", "roi_calculation", "sensitivity_analysis"],
        "supported_agents": ["prioritization", "synthesizer"]
    },
    "resource-planner": {
        "description": "Plans resource allocation and timeline optimization",
        "capabilities": ["resource_planning", "timeline_optimization", "capacity_management"],
        "supported_agents": ["planner", "orchestrator"]
    },
    "sql-generator": {
        "description": "Generates optimized SQL queries for analysis",
        "capabilities": ["query_generation", "performance_optimization", "data_validation"],
        "supported_agents": ["data_analysis"]
    },
    
    # Synthesis and Presentation Tools
    "evidence-synthesizer": {
        "description": "Synthesizes evidence using pyramid principle",
        "capabilities": ["evidence_synthesis", "logical_structuring", "argument_validation"],
        "supported_agents": ["synthesizer"]
    },
    "risk-assessor": {
        "description": "Assesses risks and mitigation strategies",
        "capabilities": ["risk_analysis", "mitigation_planning", "scenario_modeling"],
        "supported_agents": ["synthesizer", "planner"]
    },
    "presentation-designer": {
        "description": "Designs professional presentation layouts",
        "capabilities": ["slide_design", "visual_hierarchy", "brand_compliance"],
        "supported_agents": ["presentation"]
    },
    
    # Workflow Management Tools
    "workflow-tracker": {
        "description": "Tracks workflow progress and quality metrics",
        "capabilities": ["progress_tracking", "quality_monitoring", "performance_analytics"],
        "supported_agents": ["orchestrator"]
    },
    "quality-assessor": {
        "description": "Assesses output quality across all steps",
        "capabilities": ["quality_assessment", "completeness_checking", "consistency_validation"],
        "supported_agents": ["orchestrator", "all_agents"]
    },
    "result-synthesizer": {
        "description": "Synthesizes results across multiple analysis steps",
        "capabilities": ["result_aggregation", "insight_extraction", "recommendation_prioritization"],
        "supported_agents": ["orchestrator", "synthesizer"]
    }
}


def register_all_agents_v2() -> Dict[str, BaseSevenStepsAgent]:
    """Register all agents with OpenAI Agents SDK implementation."""
    
    # Create agent instances with MCP tools
    agents = {
        "problem_framer": ProblemFramer(),
        "issue_tree": IssueTreeAgent(),
        # TODO: Create remaining agents with v2 implementation
        # "prioritization": PrioritizationAgent(),
        # "planner": PlannerAgent(), 
        # "data_analysis": DataAnalysisAgent(),
        # "synthesizer": SynthesizerAgent(),
        # "presentation": PresentationAgent()
    }
    
    # Register agents with enhanced metadata
    registry = get_agent_registry()
    
    for name, agent in agents.items():
        # Get MCP tools for this agent
        agent_mcp_tools = get_mcp_tools_for_agent(name)
        
        # Get capabilities for this agent
        capabilities = get_agent_capabilities(name)
        
        # Register with metadata
        registry.register(
            name=name,
            agent=agent,
            capabilities=capabilities,
            mcp_tools=agent_mcp_tools
        )
    
    return agents


def get_mcp_tools_for_agent(agent_name: str) -> List[str]:
    """Get MCP tools available for a specific agent."""
    
    mcp_tools = []
    
    for tool_name, tool_info in MCP_TOOLS_REGISTRY.items():
        supported_agents = tool_info.get("supported_agents", [])
        if agent_name in supported_agents or "all_agents" in supported_agents:
            mcp_tools.append(tool_name)
    
    return mcp_tools


def get_agent_capabilities(agent_name: str) -> List[str]:
    """Get capabilities for a specific agent."""
    
    capability_map = {
        "problem_framer": [
            "problem_structuring", "goal_definition", "scope_clarification",
            "kpi_identification", "stakeholder_analysis", "assumption_validation"
        ],
        "issue_tree": [
            "mece_decomposition", "hypothesis_generation", "tree_visualization",
            "data_requirement_mapping", "priority_hinting", "completeness_validation"
        ],
        "prioritization": [
            "impact_assessment", "feasibility_analysis", "priority_matrix_creation",
            "resource_optimization", "risk_evaluation", "sequence_planning"
        ],
        "planner": [
            "analysis_planning", "resource_allocation", "timeline_creation",
            "methodology_selection", "milestone_definition", "risk_mitigation"
        ],
        "data_analysis": [
            "data_processing", "statistical_analysis", "model_development",
            "hypothesis_testing", "artifact_generation", "quality_validation"
        ],
        "synthesizer": [
            "evidence_synthesis", "recommendation_development", "impact_quantification",
            "risk_assessment", "pyramid_structuring", "stakeholder_alignment"
        ],
        "presentation": [
            "slide_creation", "narrative_development", "visual_design",
            "audience_adaptation", "multimedia_generation", "delivery_optimization"
        ]
    }
    
    return capability_map.get(agent_name, [])


# Agent step mapping for workflow orchestration
AGENT_STEP_MAPPING_V2 = {
    "frame": "problem_framer",
    "tree": "issue_tree",
    "prioritize": "prioritization", 
    "plan": "planner",
    "analyze": "data_analysis",
    "synthesize": "synthesizer",
    "present": "presentation"
}


# Workflow templates for common business scenarios
WORKFLOW_TEMPLATES = {
    "customer_retention": {
        "name": "Customer Retention Analysis",
        "description": "Analyze customer churn and develop retention strategies",
        "steps": ["frame", "tree", "prioritize", "analyze", "synthesize", "present"],
        "recommended_mcp_tools": [
            "data-source-connector", "impact-calculator", "sql-generator", 
            "evidence-synthesizer", "presentation-designer"
        ],
        "typical_duration": "2-3 weeks",
        "key_outputs": ["churn_analysis", "retention_strategies", "roi_projections"]
    },
    
    "market_entry": {
        "name": "Market Entry Assessment", 
        "description": "Evaluate market opportunities and entry strategies",
        "steps": ["frame", "tree", "prioritize", "plan", "analyze", "synthesize", "present"],
        "recommended_mcp_tools": [
            "stakeholder-analyzer", "risk-assessor", "impact-calculator",
            "evidence-synthesizer", "presentation-designer"
        ],
        "typical_duration": "4-6 weeks",
        "key_outputs": ["market_assessment", "entry_strategy", "financial_projections"]
    },
    
    "operational_efficiency": {
        "name": "Operational Efficiency Improvement",
        "description": "Identify and implement operational improvements",
        "steps": ["frame", "tree", "prioritize", "plan", "analyze", "synthesize", "present"],
        "recommended_mcp_tools": [
            "data-source-connector", "sql-generator", "resource-planner",
            "impact-calculator", "presentation-designer"
        ],
        "typical_duration": "3-4 weeks", 
        "key_outputs": ["process_analysis", "improvement_recommendations", "implementation_roadmap"]
    }
}


def create_workflow_from_template(
    template_name: str,
    problem_description: str,
    customizations: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a workflow configuration from a template."""
    
    if template_name not in WORKFLOW_TEMPLATES:
        raise ValueError(f"Unknown workflow template: {template_name}")
    
    template = WORKFLOW_TEMPLATES[template_name].copy()
    
    # Apply customizations
    if customizations:
        template.update(customizations)
    
    # Create workflow configuration
    workflow_config = {
        "template": template_name,
        "problem_description": problem_description,
        "steps": template["steps"],
        "mcp_tools": template["recommended_mcp_tools"],
        "estimated_duration": template["typical_duration"],
        "expected_outputs": template["key_outputs"],
        "workflow_mode": WorkflowMode.AGENT_AS_TOOL,
        "quality_gates": True
    }
    
    return workflow_config


# Export all classes and functions
__all__ = [
    # Base classes
    "BaseSevenStepsAgent", "AgentInput", "AgentOutput", "AgentContext", 
    "AgentRegistry", "get_agent_registry",
    
    # Agents (v2 implementations)
    "ProblemFramer", "ProblemFramerInput", "ProblemFramerOutput",
    "IssueTreeAgent", "IssueTreeInput", "IssueTreeAgentOutput",
    
    # Orchestration
    "SevenStepsOrchestrator", "WorkflowMode", "WorkflowStep", "get_orchestrator",
    
    # Registry and tools
    "register_all_agents_v2", "MCP_TOOLS_REGISTRY", "AGENT_STEP_MAPPING_V2",
    "get_mcp_tools_for_agent", "get_agent_capabilities",
    
    # Workflow templates
    "WORKFLOW_TEMPLATES", "create_workflow_from_template"
]