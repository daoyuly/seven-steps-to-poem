# Data Models and Database Schema

## Database Design

The Seven Steps to Poem system uses a multi-database architecture:
- **PostgreSQL**: Primary transactional data (problems, users, metadata)
- **Neo4j**: Issue tree relationships and graph queries
- **Redis**: Caching, session storage, and real-time data
- **Vector Database**: Embeddings for RAG and similarity search
- **Object Storage**: Artifacts, files, and generated deliverables

## Core Entity Models

### 1. Problem Entity

```sql
-- Main problem tracking table
CREATE TABLE problems (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    raw_text TEXT NOT NULL,
    title VARCHAR(255),
    submitter VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'created',
    urgency VARCHAR(20) DEFAULT 'medium',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- Audit fields
    created_by UUID REFERENCES users(id),
    organization_id UUID REFERENCES organizations(id),
    
    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('created', 'framing', 'tree_generation', 
        'prioritizing', 'planning', 'analyzing', 'synthesizing', 'presenting', 
        'completed', 'failed', 'cancelled')),
    CONSTRAINT valid_urgency CHECK (urgency IN ('low', 'medium', 'high', 'critical'))
);

-- Indexes
CREATE INDEX idx_problems_status ON problems(status);
CREATE INDEX idx_problems_submitter ON problems(submitter);
CREATE INDEX idx_problems_created_at ON problems(created_at);
CREATE INDEX idx_problems_organization ON problems(organization_id);
```

### 2. Problem Frame

```sql
-- Structured problem definition
CREATE TABLE problem_frames (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    goal TEXT NOT NULL,
    scope JSONB NOT NULL DEFAULT '{"include": [], "exclude": []}',
    kpis JSONB NOT NULL DEFAULT '[]',
    stakeholders JSONB NOT NULL DEFAULT '[]',
    assumptions JSONB NOT NULL DEFAULT '[]',
    constraints JSONB NOT NULL DEFAULT '[]',
    confidence VARCHAR(20) DEFAULT 'medium',
    clarifying_questions JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_confidence CHECK (confidence IN ('low', 'medium', 'high')),
    CONSTRAINT unique_problem_frame UNIQUE (problem_id)
);
```

### 3. Issue Tree (Neo4j)

```cypher
// Neo4j node and relationship definitions

// Tree Node
CREATE CONSTRAINT node_id_unique FOR (n:TreeNode) REQUIRE n.id IS UNIQUE;

// Node properties
(:TreeNode {
    id: string,
    problem_id: string,
    title: string,
    hypotheses: [string],
    required_data: [string],
    priority_hint: string, // 'high', 'medium', 'low'
    notes: string,
    level: integer,
    created_at: datetime,
    updated_at: datetime
})

// Relationships
(:TreeNode)-[:PARENT_OF]->(:TreeNode)
(:TreeNode)-[:CHILD_OF]->(:TreeNode)
(:TreeNode)-[:SIBLING_OF]->(:TreeNode)

// Root node identification
(:TreeNode)-[:ROOT_OF]->(:Problem {id: string})
```

### 4. Prioritization Results

```sql
CREATE TABLE node_priorities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    node_id VARCHAR(255) NOT NULL,
    impact_score INTEGER NOT NULL CHECK (impact_score >= 1 AND impact_score <= 10),
    feasibility_score INTEGER NOT NULL CHECK (feasibility_score >= 1 AND feasibility_score <= 10),
    priority_score DECIMAL(4,2) NOT NULL,
    rationale TEXT,
    estimated_hours INTEGER,
    estimated_cost DECIMAL(10,2),
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    is_recommended BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure unique priority per node per problem
    CONSTRAINT unique_node_priority UNIQUE (problem_id, node_id)
);

CREATE INDEX idx_node_priorities_problem ON node_priorities(problem_id);
CREATE INDEX idx_node_priorities_score ON node_priorities(priority_score DESC);
```

### 5. Analysis Plans

```sql
CREATE TABLE analysis_plans (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    question_id VARCHAR(255) NOT NULL,
    title VARCHAR(255) NOT NULL,
    methods JSONB NOT NULL DEFAULT '[]',
    data_requirements JSONB NOT NULL DEFAULT '[]',
    estimated_hours INTEGER,
    owner VARCHAR(255),
    dependencies JSONB DEFAULT '[]',
    acceptance_criteria JSONB DEFAULT '[]',
    status VARCHAR(50) DEFAULT 'planned',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_plan_status CHECK (status IN ('planned', 'in_progress', 'completed', 'failed', 'cancelled'))
);

CREATE INDEX idx_analysis_plans_problem ON analysis_plans(problem_id);
CREATE INDEX idx_analysis_plans_status ON analysis_plans(status);
```

### 6. Analysis Results

```sql
CREATE TABLE analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    plan_id UUID NOT NULL REFERENCES analysis_plans(id) ON DELETE CASCADE,
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    summary TEXT,
    key_findings JSONB DEFAULT '[]',
    metrics JSONB DEFAULT '{}',
    confidence_score DECIMAL(3,2) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    execution_time INTEGER, -- seconds
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key to execution tracking
    execution_id UUID REFERENCES executions(id)
);

-- Artifact storage (references to object storage)
CREATE TABLE analysis_artifacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    result_id UUID NOT NULL REFERENCES analysis_results(id) ON DELETE CASCADE,
    artifact_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(100),
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_artifact_type CHECK (artifact_type IN ('chart', 'csv', 'notebook', 'model', 'sql_query', 'visualization'))
);

CREATE INDEX idx_artifacts_result ON analysis_artifacts(result_id);
CREATE INDEX idx_artifacts_type ON analysis_artifacts(artifact_type);
```

### 7. Recommendations

```sql
CREATE TABLE recommendations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    summary TEXT NOT NULL,
    evidence JSONB NOT NULL DEFAULT '[]',
    actions JSONB NOT NULL DEFAULT '[]',
    expected_impact JSONB NOT NULL DEFAULT '{}',
    risks JSONB DEFAULT '[]',
    confidence DECIMAL(3,2) CHECK (confidence >= 0 AND confidence <= 1),
    priority INTEGER CHECK (priority >= 1 AND priority <= 3),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_recommendations_problem ON recommendations(problem_id);
CREATE INDEX idx_recommendations_priority ON recommendations(priority);
```

### 8. Executions & Task Tracking

```sql
CREATE TABLE executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    step VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'queued',
    progress_percentage INTEGER DEFAULT 0 CHECK (progress_percentage >= 0 AND progress_percentage <= 100),
    current_activity TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    estimated_duration INTEGER, -- seconds
    actual_duration INTEGER, -- seconds
    error_details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_execution_status CHECK (status IN ('queued', 'running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT valid_step CHECK (step IN ('frame', 'tree', 'prioritize', 'plan', 'analyze', 'synthesize', 'present'))
);

-- Execution logs for real-time monitoring
CREATE TABLE execution_logs (
    id BIGSERIAL PRIMARY KEY,
    execution_id UUID NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    level VARCHAR(20) NOT NULL DEFAULT 'info',
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    
    CONSTRAINT valid_log_level CHECK (level IN ('debug', 'info', 'warning', 'error'))
);

CREATE INDEX idx_execution_logs_execution ON execution_logs(execution_id);
CREATE INDEX idx_execution_logs_timestamp ON execution_logs(timestamp);
```

### 9. Deliverables

```sql
CREATE TABLE deliverables (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    deliverable_type VARCHAR(50) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size BIGINT,
    mime_type VARCHAR(100),
    status VARCHAR(50) DEFAULT 'generating',
    generated_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    download_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_deliverable_type CHECK (deliverable_type IN ('ppt', 'pdf', 'video', 'audio', 'executive_summary')),
    CONSTRAINT valid_deliverable_status CHECK (status IN ('generating', 'ready', 'failed', 'expired'))
);

CREATE INDEX idx_deliverables_problem ON deliverables(problem_id);
CREATE INDEX idx_deliverables_type ON deliverables(deliverable_type);
```

### 10. Audit Trail

```sql
CREATE TABLE audit_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    action VARCHAR(100) NOT NULL,
    actor VARCHAR(255) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id VARCHAR(255) NOT NULL,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    
    -- Additional context
    organization_id UUID REFERENCES organizations(id),
    problem_id UUID REFERENCES problems(id)
);

CREATE INDEX idx_audit_timestamp ON audit_entries(timestamp);
CREATE INDEX idx_audit_actor ON audit_entries(actor);
CREATE INDEX idx_audit_resource ON audit_entries(resource_type, resource_id);
CREATE INDEX idx_audit_problem ON audit_entries(problem_id);
```

### 11. User Management

```sql
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    domain VARCHAR(255),
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT unique_org_domain UNIQUE (domain)
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'analyst',
    organization_id UUID NOT NULL REFERENCES organizations(id),
    preferences JSONB DEFAULT '{}',
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT unique_user_email UNIQUE (email),
    CONSTRAINT valid_role CHECK (role IN ('admin', 'analyst', 'viewer', 'owner'))
);

CREATE INDEX idx_users_organization ON users(organization_id);
CREATE INDEX idx_users_email ON users(email);
```

### 12. Feedback System

```sql
CREATE TABLE feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    problem_id UUID NOT NULL REFERENCES problems(id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    improvement_suggestions JSONB DEFAULT '[]',
    category VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Prevent duplicate feedback per user per problem
    CONSTRAINT unique_user_problem_feedback UNIQUE (problem_id, user_id)
);

CREATE INDEX idx_feedback_problem ON feedback(problem_id);
CREATE INDEX idx_feedback_rating ON feedback(rating);
```

## JSON Schema Definitions

### ProblemFrame JSON Structure
```json
{
  "goal": "string",
  "scope": {
    "include": ["string"],
    "exclude": ["string"]
  },
  "kpis": [
    {
      "name": "string",
      "baseline": "number",
      "target": "number",
      "window": "string",
      "unit": "string"
    }
  ],
  "stakeholders": ["string"],
  "assumptions": ["string"],
  "constraints": ["string"],
  "confidence": "low|medium|high",
  "clarifying_questions": [
    {
      "question": "string",
      "category": "string",
      "required": "boolean",
      "answer": "string"
    }
  ]
}
```

### AnalysisPlan JSON Structure
```json
{
  "methods": ["cohort", "regression", "segmentation"],
  "data_requirements": [
    {
      "source": "string",
      "table": "string",
      "columns": ["string"],
      "filters": "string",
      "sample_size": "number"
    }
  ],
  "dependencies": ["string"],
  "acceptance_criteria": ["string"]
}
```

### Recommendation Actions Structure
```json
{
  "actions": [
    {
      "step": "string",
      "description": "string",
      "owner": "string",
      "due_date": "ISO8601",
      "estimated_effort": "string",
      "success_criteria": "string",
      "resources_needed": ["string"]
    }
  ],
  "expected_impact": {
    "kpi": "string",
    "delta_percentage": "number",
    "delta_absolute": "number",
    "timeframe": "string",
    "assumptions": ["string"]
  },
  "risks": [
    {
      "risk": "string",
      "mitigation": "string",
      "probability": "low|medium|high",
      "impact": "low|medium|high"
    }
  ]
}
```

## Redis Data Structures

### Session Management
```redis
# User sessions
SET session:{session_id} "{user_id: uuid, expires_at: timestamp, permissions: []}"
EXPIRE session:{session_id} 3600

# Active executions
SET execution:{execution_id} "{status: 'running', progress: 45, current_activity: 'Data analysis'}"
```

### Real-time Updates
```redis
# Problem status updates (pub/sub)
PUBLISH problem_updates "{problem_id: uuid, status: 'analyzing', progress: 60}"

# Execution progress
HSET execution_progress {execution_id} progress 75 activity "Generating recommendations"
```

### Caching Layer
```redis
# Cache expensive queries
SET cache:problem_tree:{problem_id} "{tree_data}" EX 3600
SET cache:analysis_results:{problem_id} "{results}" EX 1800
```

## Vector Database Schema

### Problem Embeddings (Pinecone/Weaviate)
```json
{
  "id": "problem_{uuid}",
  "vector": [0.1, 0.2, ...], // 1536 dimensions for OpenAI embeddings
  "metadata": {
    "problem_id": "uuid",
    "industry": "string",
    "problem_type": "string",
    "keywords": ["string"],
    "created_at": "timestamp"
  }
}
```

### Analysis Artifacts Embeddings
```json
{
  "id": "artifact_{uuid}",
  "vector": [0.1, 0.2, ...],
  "metadata": {
    "problem_id": "uuid",
    "artifact_type": "chart|notebook|model",
    "summary": "string",
    "confidence": "number"
  }
}
```

## Database Migration Strategy

### Initial Schema Creation
```sql
-- Migration: 001_initial_schema.sql
-- Create core tables with proper constraints and indexes
-- Ensure foreign key relationships are properly defined

-- Migration: 002_add_audit_system.sql  
-- Add audit triggers and audit table
-- Create audit functions for automated logging

-- Migration: 003_add_full_text_search.sql
-- Add GIN indexes for JSONB columns
-- Create text search vectors for problem descriptions

-- Migration: 004_add_partitioning.sql
-- Partition large tables by date for performance
-- Implement retention policies for old data
```

### Performance Considerations

1. **Indexing Strategy**
   - Composite indexes for common query patterns
   - Partial indexes for filtered queries
   - GIN indexes for JSONB searches

2. **Partitioning**
   - Partition audit_entries by month
   - Partition execution_logs by week
   - Archive old data to cold storage

3. **Query Optimization**
   - Use materialized views for complex aggregations
   - Implement read replicas for analytical queries
   - Cache frequent queries in Redis

## Data Retention & Compliance

### Retention Policies
```sql
-- Delete old execution logs (keep 30 days)
DELETE FROM execution_logs 
WHERE timestamp < NOW() - INTERVAL '30 days';

-- Archive completed problems (keep active 1 year)
UPDATE problems 
SET archived_at = NOW() 
WHERE status = 'completed' AND completed_at < NOW() - INTERVAL '1 year';

-- Purge deleted problems (30 day grace period)
DELETE FROM problems 
WHERE deleted_at IS NOT NULL AND deleted_at < NOW() - INTERVAL '30 days';
```

### GDPR Compliance
```sql
-- Data subject deletion (right to be forgotten)
CREATE OR REPLACE FUNCTION gdpr_delete_user(user_email TEXT)
RETURNS VOID AS $$
BEGIN
    -- Anonymize audit entries
    UPDATE audit_entries SET actor = 'deleted_user' WHERE actor = user_email;
    
    -- Delete personal data
    DELETE FROM feedback WHERE user_id = (SELECT id FROM users WHERE email = user_email);
    DELETE FROM users WHERE email = user_email;
    
    -- Log the deletion
    INSERT INTO audit_entries (action, actor, resource_type, resource_id, details) 
    VALUES ('gdpr_deletion', 'system', 'user', user_email, '{"reason": "gdpr_request"}');
END;
$$ LANGUAGE plpgsql;
```

This comprehensive data model supports the full lifecycle of the Seven Steps to Poem AI Agent system while ensuring scalability, auditability, and compliance with data protection regulations.