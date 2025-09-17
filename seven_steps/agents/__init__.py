"""
Seven Steps Agents Module.

This module contains all the agent implementations for the McKinsey Seven Steps
methodology: Problem Framing, Issue Tree, Prioritization, Planning, Data Analysis,
Synthesis, and Presentation.
"""

from .base import AgentRegistry, BaseAgent, AgentInput, AgentOutput, AgentContext
from .problem_framer import ProblemFramer, ProblemFramerInput, ProblemFramerOutput
from .issue_tree import IssueTreeAgent, IssueTreeInput, IssueTreeOutput
from .prioritization import PrioritizationAgent, PrioritizationInput, PrioritizationOutput
from .planner import PlannerAgent, PlannerInput, PlannerOutput
from .data_analysis import DataAnalysisAgent, DataAnalysisInput, DataAnalysisOutput
from .synthesizer import SynthesizerAgent, SynthesizerInput, SynthesizerOutput
from .presentation import PresentationAgent, PresentationInput, PresentationOutput

# Agent registry for easy access
def register_all_agents():
    """Register all agents in the global registry."""
    # Create agent instances
    problem_framer = ProblemFramer()
    issue_tree_agent = IssueTreeAgent()
    prioritization_agent = PrioritizationAgent()
    planner_agent = PlannerAgent()
    data_analysis_agent = DataAnalysisAgent()
    synthesizer_agent = SynthesizerAgent()
    presentation_agent = PresentationAgent()
    
    # Register agents
    AgentRegistry.register("problem_framer", problem_framer)
    AgentRegistry.register("issue_tree", issue_tree_agent)
    AgentRegistry.register("prioritization", prioritization_agent)
    AgentRegistry.register("planner", planner_agent)
    AgentRegistry.register("data_analysis", data_analysis_agent)
    AgentRegistry.register("synthesizer", synthesizer_agent)
    AgentRegistry.register("presentation", presentation_agent)
    
    return {
        "problem_framer": problem_framer,
        "issue_tree": issue_tree_agent,
        "prioritization": prioritization_agent,
        "planner": planner_agent,
        "data_analysis": data_analysis_agent,
        "synthesizer": synthesizer_agent,
        "presentation": presentation_agent
    }

# Agent step mapping for orchestration
AGENT_STEP_MAPPING = {
    "frame": "problem_framer",
    "tree": "issue_tree", 
    "prioritize": "prioritization",
    "plan": "planner",
    "analyze": "data_analysis",
    "synthesize": "synthesizer",
    "present": "presentation"
}

# Export all classes and functions
__all__ = [
    # Base classes
    "BaseAgent", "AgentInput", "AgentOutput", "AgentContext", "AgentRegistry",
    
    # Problem Framer
    "ProblemFramer", "ProblemFramerInput", "ProblemFramerOutput",
    
    # Issue Tree
    "IssueTreeAgent", "IssueTreeInput", "IssueTreeOutput",
    
    # Prioritization
    "PrioritizationAgent", "PrioritizationInput", "PrioritizationOutput",
    
    # Planner
    "PlannerAgent", "PlannerInput", "PlannerOutput",
    
    # Data Analysis
    "DataAnalysisAgent", "DataAnalysisInput", "DataAnalysisOutput",
    
    # Synthesizer
    "SynthesizerAgent", "SynthesizerInput", "SynthesizerOutput",
    
    # Presentation
    "PresentationAgent", "PresentationInput", "PresentationOutput",
    
    # Registry functions
    "register_all_agents",
    "AGENT_STEP_MAPPING"
]