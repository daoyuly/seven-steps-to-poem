"""
Pydantic schemas for API request/response validation and serialization.

This module defines the data transfer objects (DTOs) used for API
communication with proper validation and documentation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    class Config:
        from_attributes = True
        use_enum_values = True
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


# Request schemas
class ProblemCreateRequest(BaseSchema):
    """Request schema for creating a new problem."""
    
    raw_text: str = Field(
        ...,
        min_length=10,
        max_length=10000,
        description="Natural language description of the business problem"
    )
    submitter: str = Field(
        ...,
        max_length=255,
        description="Email address of the person submitting the problem"
    )
    urgency: str = Field(
        default="medium",
        description="Priority level of the problem"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional context and metadata"
    )
    
    @validator("urgency")
    def validate_urgency(cls, v: str) -> str:
        if v not in ["low", "medium", "high", "critical"]:
            raise ValueError("Urgency must be low, medium, high, or critical")
        return v


class ProblemFrameUpdate(BaseSchema):
    """Request schema for updating problem frame."""
    
    clarifications: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Answers to clarifying questions"
    )
    updated_assumptions: Optional[List[str]] = Field(
        default_factory=list,
        description="Updated or additional assumptions"
    )
    additional_constraints: Optional[List[str]] = Field(
        default_factory=list,
        description="Additional constraints to consider"
    )


class PriorityUpdate(BaseSchema):
    """Request schema for updating node priorities."""
    
    node_id: str = Field(..., description="Node identifier")
    impact_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Impact score from 1 to 10"
    )
    feasibility_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Feasibility score from 1 to 10"
    )
    rationale: Optional[str] = Field(
        default=None,
        description="Rationale for the scoring"
    )


class PriorityUpdateRequest(BaseSchema):
    """Request schema for batch priority updates."""
    
    priority_updates: List[PriorityUpdate] = Field(
        ...,
        description="List of priority updates"
    )


class StepExecutionRequest(BaseSchema):
    """Request schema for executing analysis steps."""
    
    step: str = Field(
        ...,
        description="The analysis step to execute"
    )
    options: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Step-specific options"
    )
    
    @validator("step")
    def validate_step(cls, v: str) -> str:
        valid_steps = ["frame", "tree", "prioritize", "plan", "analyze", "synthesize", "present"]
        if v not in valid_steps:
            raise ValueError(f"Step must be one of: {', '.join(valid_steps)}")
        return v


class FeedbackRequest(BaseSchema):
    """Request schema for user feedback."""
    
    rating: int = Field(
        ...,
        ge=1,
        le=5,
        description="Rating from 1 to 5"
    )
    feedback_text: Optional[str] = Field(
        default=None,
        description="Detailed feedback text"
    )
    improvement_suggestions: Optional[List[str]] = Field(
        default_factory=list,
        description="Suggestions for improvement"
    )


# Response schemas
class ProblemResponse(BaseSchema):
    """Response schema for problem creation."""
    
    id: str = Field(..., description="Problem unique identifier")
    status: str = Field(..., description="Current problem status")
    created_at: datetime = Field(..., description="Creation timestamp")
    estimated_completion: Optional[datetime] = Field(
        default=None,
        description="Estimated completion time"
    )


class ProblemSummary(BaseSchema):
    """Summary schema for problem listings."""
    
    id: str
    title: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime
    urgency: str
    progress_percentage: Optional[int] = Field(default=0)


class ProblemDetail(BaseSchema):
    """Detailed problem information."""
    
    id: str
    raw_text: str
    submitter: str
    status: str
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]
    progress: Optional[Dict[str, Any]] = Field(default=None)


class KPI(BaseSchema):
    """Key Performance Indicator definition."""
    
    name: str = Field(..., description="KPI name")
    baseline: Optional[float] = Field(default=None, description="Current baseline value")
    target: Optional[float] = Field(default=None, description="Target value")
    window: str = Field(..., description="Time window for measurement")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")


class ClarifyingQuestion(BaseSchema):
    """Clarifying question structure."""
    
    question: str = Field(..., description="The question text")
    category: str = Field(..., description="Question category")
    required: bool = Field(default=False, description="Whether answer is required")
    answer: Optional[str] = Field(default=None, description="User's answer")


class ProblemFrame(BaseSchema):
    """Problem frame schema."""
    
    goal: str = Field(..., description="Primary goal statement")
    scope: Dict[str, List[str]] = Field(
        default_factory=lambda: {"include": [], "exclude": []},
        description="Scope definition with inclusions and exclusions"
    )
    kpis: List[KPI] = Field(default_factory=list, description="Key performance indicators")
    stakeholders: List[str] = Field(default_factory=list, description="Key stakeholders")
    assumptions: List[str] = Field(default_factory=list, description="Key assumptions")
    constraints: List[str] = Field(default_factory=list, description="Constraints")
    confidence: str = Field(default="medium", description="Confidence level")
    clarifying_questions: List[ClarifyingQuestion] = Field(
        default_factory=list,
        description="Questions for clarification"
    )


class TreeNode(BaseSchema):
    """Issue tree node structure."""
    
    id: str = Field(..., description="Node identifier")
    title: str = Field(..., description="Node title")
    hypotheses: List[str] = Field(default_factory=list, description="Testable hypotheses")
    required_data: List[str] = Field(default_factory=list, description="Required data sources")
    priority_hint: Optional[str] = Field(default=None, description="Priority hint")
    children: List["TreeNode"] = Field(default_factory=list, description="Child nodes")
    notes: Optional[str] = Field(default=None, description="Additional notes")


TreeNode.model_rebuild()  # Required for self-referencing model


class IssueTree(BaseSchema):
    """Complete issue tree structure."""
    
    root: TreeNode = Field(..., description="Root node of the tree")
    visualization_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="D3.js compatible visualization data"
    )


class PriorityScore(BaseSchema):
    """Priority scoring for a tree node."""
    
    node_id: str
    impact_score: int = Field(ge=1, le=10)
    feasibility_score: int = Field(ge=1, le=10)
    priority_score: float
    rationale: Optional[str] = None
    estimated_cost: Optional[Dict[str, Union[int, float]]] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class PrioritizationResults(BaseSchema):
    """Complete prioritization results."""
    
    scored_nodes: List[PriorityScore]
    recommended_order: List[str] = Field(
        description="Recommended execution order by node ID"
    )


class DataRequirement(BaseSchema):
    """Data requirement specification."""
    
    source: str = Field(..., description="Data source name")
    table: Optional[str] = Field(default=None, description="Table or dataset name")
    columns: Optional[List[str]] = Field(default_factory=list, description="Required columns")
    filters: Optional[str] = Field(default=None, description="Filter criteria")
    sample_size: Optional[int] = Field(default=None, description="Sample size needed")


class AnalysisPlan(BaseSchema):
    """Analysis execution plan."""
    
    task_id: str
    question_id: str
    title: str
    methods: List[str] = Field(description="Analysis methods to use")
    data_requirements: List[DataRequirement]
    estimated_hours: Optional[int] = None
    owner: Optional[str] = None
    dependencies: List[str] = Field(default_factory=list)
    acceptance_criteria: List[str] = Field(default_factory=list)


class AnalysisArtifact(BaseSchema):
    """Analysis artifact metadata."""
    
    type: str = Field(..., description="Artifact type")
    url: str = Field(..., description="Download URL")
    description: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResult(BaseSchema):
    """Analysis execution results."""
    
    task_id: str
    artifacts: List[AnalysisArtifact] = Field(default_factory=list)
    summary: Optional[str] = None
    key_findings: List[str] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)
    confidence_score: Optional[float] = Field(default=None, ge=0, le=1)
    execution_time: Optional[int] = Field(default=None, description="Execution time in seconds")
    completed_at: datetime


class ActionItem(BaseSchema):
    """Recommendation action item."""
    
    step: str = Field(..., description="Action step description")
    description: str = Field(..., description="Detailed description")
    owner: Optional[str] = Field(default=None, description="Responsible person")
    due_date: Optional[datetime] = Field(default=None, description="Due date")
    estimated_effort: Optional[str] = Field(default=None, description="Effort estimate")
    success_criteria: Optional[str] = Field(default=None, description="Success criteria")


class ExpectedImpact(BaseSchema):
    """Expected impact specification."""
    
    kpi: str = Field(..., description="KPI name")
    delta_percentage: Optional[float] = Field(default=None, description="Percentage change")
    delta_absolute: Optional[float] = Field(default=None, description="Absolute change")
    timeframe: str = Field(..., description="Expected timeframe")


class Risk(BaseSchema):
    """Risk assessment."""
    
    risk: str = Field(..., description="Risk description")
    mitigation: str = Field(..., description="Mitigation strategy")
    probability: str = Field(..., description="Risk probability")
    impact: Optional[str] = Field(default=None, description="Risk impact level")
    
    @validator("probability", "impact")
    def validate_risk_level(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ["low", "medium", "high"]:
            raise ValueError("Risk level must be low, medium, or high")
        return v


class Recommendation(BaseSchema):
    """Complete recommendation structure."""
    
    id: str
    title: str
    summary: str
    evidence: List[str] = Field(description="Supporting evidence")
    actions: List[ActionItem] = Field(description="Action items")
    expected_impact: ExpectedImpact = Field(description="Expected impact")
    risks: List[Risk] = Field(default_factory=list, description="Risk assessment")
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    priority: Optional[int] = Field(default=None, ge=1, le=3)


class ExecutionStatus(BaseSchema):
    """Execution status information."""
    
    execution_id: str
    status: str
    progress_percentage: int = Field(ge=0, le=100)
    current_activity: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_remaining: Optional[int] = Field(
        default=None,
        description="Estimated remaining time in seconds"
    )
    error_details: Optional[Dict[str, Any]] = None


class ExecutionLog(BaseSchema):
    """Execution log entry."""
    
    timestamp: datetime
    level: str
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Deliverables(BaseSchema):
    """Available deliverables."""
    
    ppt_url: Optional[str] = Field(default=None, description="PowerPoint download URL")
    pdf_url: Optional[str] = Field(default=None, description="PDF download URL")
    video_url: Optional[str] = Field(default=None, description="Video download URL")
    audio_url: Optional[str] = Field(default=None, description="Audio download URL")
    executive_summary: Optional[Dict[str, str]] = Field(
        default=None,
        description="Executive summary in short and long versions"
    )
    generated_at: Optional[datetime] = None


class AuditEntry(BaseSchema):
    """Audit log entry."""
    
    id: str
    timestamp: datetime
    action: str
    actor: str
    resource: str
    details: Dict[str, Any] = Field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class PaginationInfo(BaseSchema):
    """Pagination metadata."""
    
    page: int = Field(ge=1)
    limit: int = Field(ge=1, le=100)
    total_items: int = Field(ge=0)
    total_pages: int = Field(ge=0)
    has_next: bool
    has_previous: bool


class ErrorResponse(BaseSchema):
    """Standard error response."""
    
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional error details"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")


class HealthCheck(BaseSchema):
    """Health check response."""
    
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Individual service statuses"
    )


class SystemMetrics(BaseSchema):
    """System performance metrics."""
    
    total_problems: int = Field(ge=0)
    active_problems: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    average_processing_time: int = Field(ge=0, description="Average processing time in seconds")
    agent_performance: Dict[str, Dict[str, Union[float, int]]] = Field(
        default_factory=dict
    )