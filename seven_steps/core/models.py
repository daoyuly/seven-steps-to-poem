"""
Core data models for Seven Steps to Poem system.

This module defines the SQLAlchemy models that represent the core
business entities in the system.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from .database import Base


def generate_uuid() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps."""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class Organization(Base, TimestampMixin):
    """Organization model for multi-tenancy."""
    
    __tablename__ = "organizations"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[Optional[str]] = mapped_column(String(255), unique=True)
    settings: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Relationships
    users: Mapped[List["User"]] = relationship(
        "User", back_populates="organization", cascade="all, delete-orphan"
    )
    problems: Mapped[List["Problem"]] = relationship(
        "Problem", back_populates="organization", cascade="all, delete-orphan"
    )


class User(Base, TimestampMixin):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint("role IN ('admin', 'analyst', 'viewer', 'owner')"),
        default="analyst",
        nullable=False,
    )
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id"),
        nullable=False,
    )
    preferences: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    
    # Relationships
    organization: Mapped["Organization"] = relationship(
        "Organization", back_populates="users"
    )
    problems: Mapped[List["Problem"]] = relationship(
        "Problem", back_populates="created_by_user"
    )


class Problem(Base, TimestampMixin):
    """Core problem model representing business problems to solve."""
    
    __tablename__ = "problems"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(String(255))
    submitter: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint(
            "status IN ('created', 'framing', 'tree_generation', 'prioritizing', "
            "'planning', 'analyzing', 'synthesizing', 'presenting', 'completed', "
            "'failed', 'cancelled')"
        ),
        default="created",
        nullable=False,
    )
    urgency: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint("urgency IN ('low', 'medium', 'high', 'critical')"),
        default="medium",
        nullable=False,
    )
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    # Foreign keys
    created_by: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("users.id"),
        nullable=False,
    )
    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id"),
        nullable=False,
    )
    
    # Relationships
    created_by_user: Mapped["User"] = relationship("User", back_populates="problems")
    organization: Mapped["Organization"] = relationship(
        "Organization", back_populates="problems"
    )
    problem_frame: Mapped[Optional["ProblemFrame"]] = relationship(
        "ProblemFrame", back_populates="problem", uselist=False
    )
    executions: Mapped[List["Execution"]] = relationship(
        "Execution", back_populates="problem", cascade="all, delete-orphan"
    )
    recommendations: Mapped[List["Recommendation"]] = relationship(
        "Recommendation", back_populates="problem", cascade="all, delete-orphan"
    )


class ProblemFrame(Base, TimestampMixin):
    """Structured problem definition from Problem Framer agent."""
    
    __tablename__ = "problem_frames"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    problem_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("problems.id"),
        unique=True,
        nullable=False,
    )
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    scope: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        default={"include": [], "exclude": []},
        nullable=False,
    )
    kpis: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    stakeholders: Mapped[List[str]] = mapped_column(JSON, default=list)
    assumptions: Mapped[List[str]] = mapped_column(JSON, default=list)
    constraints: Mapped[List[str]] = mapped_column(JSON, default=list)
    confidence: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint("confidence IN ('low', 'medium', 'high')"),
        default="medium",
        nullable=False,
    )
    clarifying_questions: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON, default=list
    )
    
    # Relationships
    problem: Mapped["Problem"] = relationship("Problem", back_populates="problem_frame")


class NodePriority(Base, TimestampMixin):
    """Priority scoring for issue tree nodes."""
    
    __tablename__ = "node_priorities"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    problem_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("problems.id"),
        nullable=False,
    )
    node_id: Mapped[str] = mapped_column(String(255), nullable=False)
    impact_score: Mapped[int] = mapped_column(
        Integer,
        CheckConstraint("impact_score >= 1 AND impact_score <= 10"),
        nullable=False,
    )
    feasibility_score: Mapped[int] = mapped_column(
        Integer,
        CheckConstraint("feasibility_score >= 1 AND feasibility_score <= 10"),
        nullable=False,
    )
    priority_score: Mapped[float] = mapped_column(Numeric(4, 2), nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text)
    estimated_hours: Mapped[Optional[int]] = mapped_column(Integer)
    estimated_cost: Mapped[Optional[float]] = mapped_column(Numeric(10, 2))
    confidence: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2),
        CheckConstraint("confidence >= 0 AND confidence <= 1"),
    )
    is_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    
    __table_args__ = (
        UniqueConstraint("problem_id", "node_id", name="uq_problem_node_priority"),
    )


class AnalysisPlan(Base, TimestampMixin):
    """Analysis execution plans from Planner agent."""
    
    __tablename__ = "analysis_plans"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    problem_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("problems.id"),
        nullable=False,
    )
    question_id: Mapped[str] = mapped_column(String(255), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    methods: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    data_requirements: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    estimated_hours: Mapped[Optional[int]] = mapped_column(Integer)
    owner: Mapped[Optional[str]] = mapped_column(String(255))
    dependencies: Mapped[List[str]] = mapped_column(JSON, default=list)
    acceptance_criteria: Mapped[List[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint(
            "status IN ('planned', 'in_progress', 'completed', 'failed', 'cancelled')"
        ),
        default="planned",
        nullable=False,
    )


class AnalysisResult(Base, TimestampMixin):
    """Results from Analysis agent execution."""
    
    __tablename__ = "analysis_results"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    plan_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("analysis_plans.id"),
        nullable=False,
    )
    problem_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("problems.id"),
        nullable=False,
    )
    summary: Mapped[Optional[str]] = mapped_column(Text)
    key_findings: Mapped[List[str]] = mapped_column(JSON, default=list)
    metrics: Mapped[Dict[str, float]] = mapped_column(JSON, default=dict)
    confidence_score: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1"),
    )
    execution_time: Mapped[Optional[int]] = mapped_column(Integer)
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    execution_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("executions.id"),
    )


class Recommendation(Base, TimestampMixin):
    """Final recommendations from Synthesizer agent."""
    
    __tablename__ = "recommendations"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    problem_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("problems.id"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[List[str]] = mapped_column(JSON, default=list, nullable=False)
    actions: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON, default=list, nullable=False
    )
    expected_impact: Mapped[Dict[str, Any]] = mapped_column(
        JSON, default=dict, nullable=False
    )
    risks: Mapped[List[Dict[str, Any]]] = mapped_column(JSON, default=list)
    confidence: Mapped[Optional[float]] = mapped_column(
        Numeric(3, 2),
        CheckConstraint("confidence >= 0 AND confidence <= 1"),
    )
    priority: Mapped[Optional[int]] = mapped_column(
        Integer,
        CheckConstraint("priority >= 1 AND priority <= 3"),
    )
    
    # Relationships
    problem: Mapped["Problem"] = relationship(
        "Problem", back_populates="recommendations"
    )


class Execution(Base, TimestampMixin):
    """Workflow execution tracking."""
    
    __tablename__ = "executions"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    problem_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("problems.id"),
        nullable=False,
    )
    step: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint(
            "step IN ('frame', 'tree', 'prioritize', 'plan', 'analyze', "
            "'synthesize', 'present')"
        ),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(50),
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed', 'cancelled')"
        ),
        default="queued",
        nullable=False,
    )
    progress_percentage: Mapped[int] = mapped_column(
        Integer,
        CheckConstraint("progress_percentage >= 0 AND progress_percentage <= 100"),
        default=0,
        nullable=False,
    )
    current_activity: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    estimated_duration: Mapped[Optional[int]] = mapped_column(Integer)
    actual_duration: Mapped[Optional[int]] = mapped_column(Integer)
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Relationships
    problem: Mapped["Problem"] = relationship("Problem", back_populates="executions")
    logs: Mapped[List["ExecutionLog"]] = relationship(
        "ExecutionLog", back_populates="execution", cascade="all, delete-orphan"
    )


class ExecutionLog(Base):
    """Detailed execution logs for monitoring."""
    
    __tablename__ = "execution_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    execution_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("executions.id"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    level: Mapped[str] = mapped_column(
        String(20),
        CheckConstraint("level IN ('debug', 'info', 'warning', 'error')"),
        default="info",
        nullable=False,
    )
    message: Mapped[str] = mapped_column(Text, nullable=False)
    metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    
    # Relationships
    execution: Mapped["Execution"] = relationship("Execution", back_populates="logs")


class AuditEntry(Base):
    """Audit trail for compliance and security."""
    
    __tablename__ = "audit_entries"
    
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=generate_uuid,
    )
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    action: Mapped[str] = mapped_column(String(100), nullable=False)
    actor: Mapped[str] = mapped_column(String(255), nullable=False)
    resource_type: Mapped[str] = mapped_column(String(100), nullable=False)
    resource_id: Mapped[str] = mapped_column(String(255), nullable=False)
    details: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    ip_address: Mapped[Optional[str]] = mapped_column(String(45))  # IPv6 compatible
    user_agent: Mapped[Optional[str]] = mapped_column(Text)
    session_id: Mapped[Optional[str]] = mapped_column(String(255))
    organization_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id"),
    )
    problem_id: Mapped[Optional[str]] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("problems.id"),
    )