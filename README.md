# Seven Steps to Poem

🚀 **AI Agent System for Automated Business Problem Solving**

An enterprise-grade system that implements McKinsey's 7-step problem-solving methodology using orchestrated AI agents to automatically generate comprehensive business solutions.

## 🎯 Overview

Seven Steps to Poem transforms raw business problems into actionable solutions through:

1. **Problem Framing** - Structure and clarify business problems
2. **Issue Tree Generation** - MECE decomposition of complex problems  
3. **Prioritization** - Impact × Feasibility scoring
4. **Analysis Planning** - Executable analysis roadmaps
5. **Data Analysis** - Statistical analysis and insights
6. **Solution Synthesis** - Evidence-based recommendations
7. **Presentation** - Multi-format deliverables (PPT, PDF, Video, Audio)

## 🏗️ Architecture

- **Microservices Architecture**: Independent, scalable AI agents
- **Event-Driven**: Asynchronous workflow orchestration
- **Multi-Database**: PostgreSQL + Neo4j + Redis + Vector DB
- **Enterprise Security**: RBAC, audit trails, data encryption
- **Cloud-Native**: Kubernetes-ready with Docker containers

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Poetry (for dependency management)
- OpenAI API key

### 1. Clone and Setup

```bash
git clone <repository-url>
cd seven-steps-to-poem

# Run the automated setup script (for monolithic mode)
./scripts/dev-setup.sh

# OR run microservices setup
./scripts/start-services.sh
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY="your-openai-api-key-here"
```

### 3. Start Infrastructure

```bash
# Start databases and services
docker-compose -f docker-compose.dev.yml up -d
```

### 4. Initialize Database

```bash
# Install dependencies
poetry install

# Run database migrations
poetry run python -m seven_steps.cli migrate --auto -m "Initial schema"
```

### 5. Start the API Server

```bash
# Start development server
poetry run python -m seven_steps.cli run-server --reload

# Or alternatively
poetry run python -m seven_steps.api.main
```

### 6. Test the System

```bash
# Health check
poetry run python -m seven_steps.cli health-check

# Test Problem Framer agent
poetry run python -m seven_steps.cli test-agent problemframer

# API documentation at http://localhost:8000/docs (monolithic)
# OR http://localhost/docs (microservices)
```

## 📊 System Status

The development environment includes:

- ✅ **Core Framework**: FastAPI + SQLAlchemy + Pydantic
- ✅ **Problem Framer Agent**: Fully implemented with OpenAI integration
- ✅ **Issue Tree Agent**: MECE problem decomposition with mock data
- ✅ **Workflow Orchestrator**: Multi-mode (monolithic + microservices)
- ✅ **API Gateway**: Nginx reverse proxy with load balancing
- ✅ **Database Models**: Complete PostgreSQL schema + migrations
- ✅ **Microservices Architecture**: Independent, containerized services
- ✅ **Local Development**: Multiple deployment modes
- ✅ **Service Communication**: HTTP-based inter-service calls
- 🚧 **Remaining Agents**: Prioritization, Planner, Analysis, etc.
- 🚧 **Neo4j Integration**: Graph database for problem trees
- 🚧 **Vector Database**: Semantic search and RAG
- 🚧 **Complete Test Suite**: Comprehensive testing

## 🔧 Development Tools

### CLI Commands

```bash
# Database management
poetry run python -m seven_steps.cli init-db
poetry run python -m seven_steps.cli migrate --auto -m "Description"

# Server management  
poetry run python -m seven_steps.cli run-server --reload
poetry run python -m seven_steps.cli health-check

# Agent testing
poetry run python -m seven_steps.cli test-agent problemframer
poetry run python -m seven_steps.cli create-sample-data

# System info
poetry run python -m seven_steps.cli version
```

### Docker Services

#### Monolithic Mode
```bash
# Start infrastructure only
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f [service]

# Stop services
docker-compose -f docker-compose.dev.yml down
```

#### Microservices Mode  
```bash
# Start all microservices
./scripts/start-services.sh

# View specific service logs  
./scripts/start-services.sh logs [service-name]

# Stop all services
./scripts/start-services.sh stop

# Test all services
./scripts/test-services.sh
```

### Service URLs

#### Monolithic Mode
- **API Server**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs  

#### Microservices Mode  
- **API Gateway**: http://localhost (Nginx)
- **API Docs**: http://localhost/docs
- **Orchestrator**: http://localhost:8000
- **Problem Framer**: http://localhost:8001  
- **Issue Tree**: http://localhost:8002

#### Infrastructure (Both Modes)
- **PostgreSQL**: localhost:5432 (postgres/password)
- **Redis**: localhost:6379
- **Neo4j**: http://localhost:7474 (neo4j/password)
- **MinIO**: http://localhost:9001 (minioadmin/minioadmin)
- **Grafana**: http://localhost:3000 (admin/admin)

## 🧪 Testing

### Example API Usage

```bash
# Create a problem
curl -X POST "http://localhost:8000/v1/problems" \
  -H "Content-Type: application/json" \
  -d '{
    "raw_text": "Our SaaS churn rate increased 20% over 3 months. Need to reduce by 5% in 90 days.",
    "submitter": "pm@company.com",
    "urgency": "high",
    "metadata": {"industry": "SaaS", "company_size": "medium"}
  }'

# Execute problem framing step
curl -X POST "http://localhost:8000/v1/problems/{problem_id}/run-step" \
  -H "Content-Type: application/json" \
  -d '{"step": "frame"}'

# Check execution status
curl "http://localhost:8000/v1/problems/{problem_id}/executions/{execution_id}"
```

### Agent Testing

```bash
# Test Problem Framer with custom input
echo '{
  "raw_text": "Sales team productivity is down 15%. Need solution in 30 days.",
  "submitter": "sales.director@company.com",
  "metadata": {"urgency": "critical", "budget": "$50K"}
}' > test-input.json

poetry run python -m seven_steps.cli test-agent problemframer --input-file test-input.json
```

## 📚 Documentation

- [Technical Architecture](./TECHNICAL_ARCHITECTURE.md)
- [API Specification](./API_SPECIFICATION.yaml)
- [Data Models](./DATA_MODELS.md)
- [System Components](./SYSTEM_COMPONENTS.md)
- [Deployment Guide](./DEPLOYMENT_GUIDE.md)

## 🔐 Security

- **Authentication**: JWT-based with role-based access control
- **Data Protection**: Encryption at rest and in transit
- **Audit Trails**: Complete activity logging
- **PII Handling**: Automatic detection and masking
- **Compliance**: GDPR, SOC2, ISO 27001 ready

## 🌟 Key Features

### Business Value
- **Automated Analysis**: End-to-end problem solving in minutes
- **McKinsey Methodology**: Proven problem-solving framework
- **Scalable Solution**: Handle 100+ concurrent problems
- **Quality Assurance**: Confidence scoring and human oversight

### Technical Excellence
- **Enterprise Architecture**: Microservices + Event-driven
- **High Availability**: Multi-AZ deployment with auto-scaling
- **Observability**: Comprehensive logging, metrics, and tracing
- **Developer Experience**: Rich CLI tools and comprehensive docs

## 🗺️ Roadmap

### Phase 1: MVP (Current)
- ✅ Problem Framer Agent
- ✅ Basic Orchestration
- ✅ API Framework
- 🚧 Issue Tree Agent

### Phase 2: Core Agents
- 🔲 Prioritization Agent
- 🔲 Planner Agent  
- 🔲 Analysis Agent
- 🔲 Complete Workflow

### Phase 3: Advanced Features
- 🔲 Synthesizer Agent
- 🔲 Presentation Agent
- 🔲 Multi-media Deliverables
- 🔲 Advanced Analytics

### Phase 4: Enterprise
- 🔲 Real-time Collaboration
- 🔲 Advanced Security
- 🔲 Custom Integrations
- 🔲 Performance Optimization

## 🤝 Contributing

1. **Setup Development Environment**
   ```bash
   ./scripts/dev-setup.sh
   ```

2. **Code Standards**
   - Python 3.11+ with type hints
   - Black formatting, isort imports
   - Comprehensive docstrings
   - 90%+ test coverage

3. **Development Workflow**
   ```bash
   # Create feature branch
   git checkout -b feature/new-agent
   
   # Make changes and test
   poetry run python -m seven_steps.cli test-agent newagent
   
   # Format code
   poetry run black .
   poetry run isort .
   
   # Commit and push
   git commit -m "feat: implement new agent"
   git push origin feature/new-agent
   ```

## 📄 License

MIT License - see [LICENSE](./LICENSE) file for details.

## 🆘 Support

- **Issues**: GitHub Issues for bugs and feature requests
- **Documentation**: Comprehensive docs in `/docs` folder
- **CLI Help**: `poetry run python -m seven_steps.cli --help`

---

**Seven Steps to Poem** - Transforming business problems into poetry through AI 🎭✨
