# Seven Steps to Poem - OpenAI Agents SDK Architecture

## 概述

基于第一原理重构的七步成诗商业分析系统，采用OpenAI Agents SDK框架实现，具备结构化输出、工具集成和MCP生态支持。

## 🏗️ 核心架构

### 基于OpenAI Agents SDK的Agent系统
```
seven_steps/agents/
├── base_v2.py              # OpenAI Agents SDK基础框架
├── problem_framer_v2.py    # 问题框架化Agent
├── issue_tree_v2.py        # MECE问题树Agent  
├── prioritization_v2.py    # 优先级评估Agent
├── planner_v2.py          # 分析计划Agent
├── data_analysis_v2.py    # 数据分析Agent
├── synthesizer_v2.py      # 综合分析Agent
├── presentation_v2.py     # 呈现生成Agent
├── orchestrator_v2.py     # 高级协调器
└── __init___v2.py         # 模块注册和工具映射
```

## 🎯 核心特性

### 1. 结构化输出
- **Pydantic模型驱动**: 每个Agent定义严格的输入/输出模型
- **类型安全**: 编译时类型检查和运行时验证
- **一致性保证**: 标准化的数据结构在Agent间传递

```python
class ProblemFramingOutput(BaseModel):
    goal: str = Field(description="Clear, SMART goal statement")
    scope: Dict[str, List[str]] = Field(description="Scope with include/exclude lists")
    kpis: List[Dict[str, Any]] = Field(description="Key performance indicators")
    stakeholders: List[str] = Field(description="Key stakeholder groups")
    # ... 更多字段
```

### 2. MCP工具生态集成
- **标准化工具访问**: 通过MCP协议访问外部工具和数据源
- **工具注册机制**: 动态发现和注册可用工具
- **多种连接方式**: 支持HTTP、Hosted、stdio MCP服务器

```python
# MCP工具定义
MCP_TOOLS_REGISTRY = {
    "data-source-connector": {
        "description": "Connects to various business data sources",
        "capabilities": ["database_access", "api_integration", "file_processing"],
        "supported_agents": ["problem_framer", "data_analysis", "planner"]
    },
    "stakeholder-analyzer": {
        "description": "Analyzes stakeholder relationships",
        "capabilities": ["stakeholder_mapping", "influence_analysis"],
        "supported_agents": ["problem_framer", "synthesizer"]
    }
}
```

### 3. 高级Agent协调

#### Agent-as-Tool模式
```python
class SevenStepsOrchestrator(Agent):
    """Portfolio Manager式协调器"""
    
    async def _execute_agent_as_tool_mode(self, context, input_data):
        # 智能协调七个专家Agent
        # 每个Agent作为工具被调用
        # 动态决策执行流程
```

#### 多种执行模式
- **SEQUENTIAL**: 依次执行，依赖管理
- **PARALLEL**: 并行执行独立步骤
- **AGENT_AS_TOOL**: Agent间相互调用
- **HANDOFF**: Agent间传递控制权

### 4. 工具函数装饰器
```python
@BaseSevenStepsAgent.create_tool_function(
    "analyze_stakeholders", 
    "Analyze stakeholder relationships and influence mapping"
)
async def analyze_stakeholders(self, stakeholders: List[str]) -> Dict[str, Any]:
    # 工具函数实现
    result = await self.call_mcp_tool("stakeholder-analyzer", {...})
    return result
```

## 🔧 使用方式

### 基础工作流执行
```python
from seven_steps.agents import get_orchestrator, WorkflowMode

orchestrator = get_orchestrator()

# 执行完整七步工作流
results = await orchestrator.execute_workflow(
    problem_id="prob_001",
    user_id="user_001",
    organization_id="org_001", 
    initial_input={
        "raw_text": "客户流失率上升，需要制定挽回策略",
        "submitter": "business@company.com",
        "metadata": {"industry": "SaaS", "urgency": "high"}
    },
    workflow_mode=WorkflowMode.AGENT_AS_TOOL
)
```

### 单步骤执行
```python
from seven_steps.agents import get_agent_registry

registry = get_agent_registry()
problem_framer = registry.get("problem_framer")

result = await problem_framer.execute({
    "context": {...},
    "raw_text": "业务问题描述...",
    "submitter": "user@company.com"
})
```

### 工作流模板使用
```python
from seven_steps.agents import create_workflow_from_template

workflow_config = create_workflow_from_template(
    template_name="customer_retention",
    problem_description="分析客户流失原因并制定挽回策略",
    customizations={
        "focus_areas": ["高价值客户", "新客户"],
        "timeline": "急迫"
    }
)
```

## 🛠️ MCP工具集成

### 可用工具类别

#### 业务分析工具
- `data-source-connector`: 数据源连接
- `stakeholder-analyzer`: 利益相关者分析  
- `kpi-validator`: KPI验证

#### 问题分解工具
- `mece-validator`: MECE原则验证
- `hypothesis-generator`: 假设生成
- `data-mapper`: 数据需求映射

#### 分析规划工具
- `impact-calculator`: 影响计算
- `resource-planner`: 资源规划
- `sql-generator`: SQL生成

#### 综合呈现工具
- `evidence-synthesizer`: 证据综合
- `risk-assessor`: 风险评估
- `presentation-designer`: 演示设计

### 工具调用示例
```python
# 在Agent中调用MCP工具
validation_result = await self.call_mcp_tool(
    tool_name="kpi-validator",
    parameters={
        "kpi_name": "customer_churn_rate",
        "baseline": 0.05,
        "target": 0.03,
        "unit": "monthly_percentage"
    }
)
```

## 📊 工作流模板

### 内置模板
1. **客户留存分析** (`customer_retention`)
2. **市场进入评估** (`market_entry`)
3. **运营效率改进** (`operational_efficiency`)

### 模板结构
```python
WORKFLOW_TEMPLATES = {
    "customer_retention": {
        "name": "Customer Retention Analysis",
        "steps": ["frame", "tree", "prioritize", "analyze", "synthesize", "present"],
        "recommended_mcp_tools": [
            "data-source-connector", "impact-calculator", 
            "sql-generator", "evidence-synthesizer"
        ],
        "typical_duration": "2-3 weeks",
        "key_outputs": ["churn_analysis", "retention_strategies", "roi_projections"]
    }
}
```

## 🎭 Agent能力矩阵

| Agent | 核心能力 | MCP工具 | 结构化输出 |
|-------|---------|---------|------------|
| ProblemFramer | 问题结构化、目标定义 | stakeholder-analyzer, kpi-validator | ProblemFramingOutput |
| IssueTreeAgent | MECE分解、假设生成 | mece-validator, hypothesis-generator | IssueTreeOutput |
| Orchestrator | 工作流协调、质量控制 | workflow-tracker, quality-assessor | 动态输出 |

## 🚀 扩展性设计

### 新Agent添加
```python
class CustomAgent(BaseSevenStepsAgent):
    def __init__(self):
        super().__init__(
            name="CustomAgent",
            description="Custom business analysis agent",
            instructions="...",
            output_type=CustomOutput,
            mcp_tools=["custom-tool-1", "custom-tool-2"]
        )
```

### 新MCP工具注册
```python
# 在MCP_TOOLS_REGISTRY中添加新工具
"custom-analyzer": {
    "description": "Custom analysis tool",
    "capabilities": ["custom_analysis"],
    "supported_agents": ["custom_agent"]
}
```

## 💡 关键优势

1. **原生结构化**: OpenAI结构化输出确保数据一致性
2. **工具生态**: MCP标准化工具访问，无限扩展能力
3. **智能协调**: Portfolio Manager模式的高级工作流管理
4. **类型安全**: Pydantic模型提供编译时和运行时验证
5. **模板化**: 预定义工作流模板快速部署常见场景
6. **可观测性**: 内置跟踪、日志和质量监控

这个重构后的架构充分利用了OpenAI Agents SDK的现代化能力，为企业级商业分析提供了强大、灵活、可扩展的AI Agent平台。