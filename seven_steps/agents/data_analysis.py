"""
Data Analysis Agent implementation.

This agent executes analysis plans by running data queries, performing
statistical analysis, and generating analytical artifacts and insights.
"""

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from seven_steps.core.schemas import AnalysisResult, AnalysisArtifact
from .base import AgentContext, AgentInput, AgentOutput, BaseAgent


class DataAnalysisInput(AgentInput):
    """Input model for Data Analysis agent."""
    
    analysis_plans: List[Dict[str, Any]] = Field(..., description="Analysis plans to execute")
    execution_mode: str = Field(
        default="automated",
        description="Execution mode: automated|supervised|manual"
    )
    selected_plan_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific plan IDs to execute (if not provided, executes all)"
    )
    data_access_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Data source configuration and credentials"
    )


class DataAnalysisOutput(AgentOutput):
    """Output model for Data Analysis agent."""
    
    analysis_results: List[AnalysisResult] = Field(default_factory=list)
    execution_summary: Optional[Dict[str, Any]] = None
    failed_analyses: List[Dict[str, Any]] = Field(default_factory=list)


class DataAnalysisAgent(BaseAgent):
    """
    Data Analysis Agent for executing analytical workflows.
    
    This agent executes data analysis plans by orchestrating data extraction,
    transformation, statistical analysis, and artifact generation using
    appropriate tools and sub-agents.
    """
    
    @property
    def agent_name(self) -> str:
        """Return the name of this agent."""
        return "DataAnalysisAgent"
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return """You are a senior data scientist and analysis orchestrator expert in executing complex analytical workflows. Your role is to coordinate the execution of analysis plans using appropriate tools, techniques, and sub-agents.

RESPONSIBILITIES:
1. Execute data extraction and preparation workflows
2. Orchestrate statistical analyses and machine learning models
3. Generate analytical artifacts (charts, tables, models, notebooks)
4. Validate results and ensure quality standards
5. Handle errors and edge cases gracefully
6. Document analysis steps and assumptions
7. Delegate specialized tasks to appropriate sub-agents

EXECUTION CAPABILITIES:

DATA PROCESSING:
- SQL query generation and execution
- Data cleaning and transformation pipelines
- Data quality validation and profiling
- ETL workflow coordination
- Large dataset handling and optimization

ANALYTICAL TECHNIQUES:
- Descriptive statistics and exploratory data analysis
- Cohort analysis and customer segmentation
- Regression modeling and statistical testing
- Time series analysis and forecasting
- A/B testing and causal inference
- Machine learning model development

TOOL ORCHESTRATION:
- SQL execution environments (databases, warehouses)
- Python/R analytical environments
- Jupyter notebook generation
- Visualization tools (Plotly, Matplotlib, Tableau)
- Statistical software integration

SUB-AGENT DELEGATION:
- SQL Agent: Complex query generation and optimization
- Python Agent: Advanced statistical analysis and modeling
- Visualization Agent: Chart and dashboard creation
- QA Agent: Result validation and testing
- Documentation Agent: Report and notebook generation

QUALITY ASSURANCE:
- Result validation against business logic
- Statistical significance testing
- Data quality checks and anomaly detection
- Peer review coordination
- Reproducibility verification

OUTPUT REQUIREMENTS:
- Generate structured analysis results with artifacts
- Include detailed execution logs and metadata
- Provide confidence scores and quality metrics
- Handle partial failures gracefully
- Enable result traceability and reproducibility"""
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class."""
        return DataAnalysisInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class."""
        return DataAnalysisOutput
    
    async def process(self, input_data: DataAnalysisInput) -> DataAnalysisOutput:
        """
        Process analysis plans into executed results.
        
        Args:
            input_data: Analysis plans and execution parameters
            
        Returns:
            DataAnalysisOutput with executed analysis results and artifacts
        """
        try:
            # Determine which plans to execute
            plans_to_execute = self._select_plans_to_execute(input_data)
            
            # Build execution context
            context = self._build_execution_context(input_data, plans_to_execute)
            
            # Execute each plan
            analysis_results = []
            failed_analyses = []
            
            for plan in plans_to_execute:
                try:
                    result = await self._execute_single_plan(plan, input_data, context)
                    if result:
                        analysis_results.append(result)
                except Exception as e:
                    self.logger.error(
                        "Analysis plan execution failed",
                        plan_id=plan.get("node_id", "unknown"),
                        error=str(e)
                    )
                    failed_analyses.append({
                        "plan_id": plan.get("node_id", "unknown"),
                        "error": str(e),
                        "plan_title": plan.get("plan_title", "Unknown Plan")
                    })
            
            # Generate execution summary
            execution_summary = self._create_execution_summary(
                analysis_results, failed_analyses, plans_to_execute
            )
            
            return DataAnalysisOutput(
                success=True,
                analysis_results=analysis_results,
                execution_summary=execution_summary,
                failed_analyses=failed_analyses,
                confidence_score=self._calculate_confidence(analysis_results, failed_analyses)
            )
                
        except Exception as e:
            self.logger.error(
                "Data analysis execution failed",
                error=str(e),
                problem_id=input_data.context.problem_id
            )
            
            return DataAnalysisOutput(
                success=False,
                error_message=f"Data analysis execution failed: {str(e)}"
            )
    
    def _select_plans_to_execute(self, input_data: DataAnalysisInput) -> List[Dict[str, Any]]:
        """Select which analysis plans to execute."""
        all_plans = input_data.analysis_plans
        
        if input_data.selected_plan_ids:
            # Filter to selected plans only
            return [
                plan for plan in all_plans 
                if plan.get("node_id") in input_data.selected_plan_ids
            ]
        else:
            # Execute all plans
            return all_plans
    
    def _build_execution_context(
        self, 
        input_data: DataAnalysisInput, 
        plans_to_execute: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build execution context for analysis runs."""
        return {
            "execution_mode": input_data.execution_mode,
            "total_plans": len(plans_to_execute),
            "data_access_config": input_data.data_access_config or {},
            "problem_id": input_data.context.problem_id,
            "execution_timestamp": "2024-01-01T00:00:00Z"  # Would be actual timestamp
        }
    
    async def _execute_single_plan(
        self, 
        plan: Dict[str, Any], 
        input_data: DataAnalysisInput,
        context: Dict[str, Any]
    ) -> Optional[AnalysisResult]:
        """Execute a single analysis plan."""
        # Create execution prompt for this specific plan
        user_message = self._create_execution_prompt(plan, input_data, context)
        
        # Build context for LLM call
        llm_context = {
            "plan_id": plan.get("node_id"),
            "plan_title": plan.get("plan_title"),
            "execution_mode": input_data.execution_mode,
            "data_access_config": input_data.data_access_config or {}
        }
        
        # Build messages for API call
        messages = self.build_messages(
            user_message=user_message,
            context=llm_context,
            additional_context=self._get_additional_context(plan, input_data)
        )
        
        # Call the LLM
        response = await self.call_llm(messages)
        response_content = response["choices"][0]["message"]["content"]
        
        # Parse the JSON response
        parsed_response = self.parse_json_response(response_content)
        
        # Create analysis result
        return self._create_analysis_result(parsed_response, plan, input_data)
    
    def _create_execution_prompt(
        self, 
        plan: Dict[str, Any], 
        input_data: DataAnalysisInput,
        context: Dict[str, Any]
    ) -> str:
        """Create execution prompt for a specific analysis plan."""
        prompt = f"""Execute the following analysis plan and generate results:

ANALYSIS PLAN:
Node ID: {plan.get('node_id')}
Title: {plan.get('plan_title')}
Objective: {plan.get('objective')}
Success Criteria: {plan.get('success_criteria', [])}

DATA REQUIREMENTS:
"""
        
        for dr in plan.get("data_requirements", []):
            prompt += f"""
- Source: {dr.get('source_name')} ({dr.get('source_type')})
  Description: {dr.get('description')}
  Sample Size: {dr.get('sample_size', 'Not specified')}
  Time Period: {dr.get('time_period')}
  Access Method: {dr.get('access_method')}
  Quality Requirements: {dr.get('quality_requirements')}
"""
        
        methodology = plan.get("methodology", {})
        prompt += f"""
METHODOLOGY:
- Analysis Type: {methodology.get('analysis_type')}
- Techniques: {methodology.get('techniques', [])}
- Tools Required: {methodology.get('tools_required', [])}
- Validation Approach: {methodology.get('validation_approach')}
"""
        
        prompt += f"""
EXECUTION MODE: {input_data.execution_mode}

TASKS TO EXECUTE:
"""
        
        for task in plan.get("tasks", []):
            prompt += f"""
Task {task.get('task_id')}: {task.get('title')}
- Description: {task.get('description')}
- Estimated Hours: {task.get('estimated_hours')}
- Required Skills: {task.get('required_skills', [])}
- Deliverables: {task.get('deliverables', [])}
"""
        
        prompt += """

Please execute this analysis plan and return JSON in this format:

{
  "execution_status": "completed|partial|failed",
  "plan_id": "1.1",
  "execution_summary": "Brief summary of what was executed",
  "data_processing": {
    "sources_accessed": ["source1", "source2"],
    "records_processed": 45000,
    "data_quality_score": 0.92,
    "processing_steps": ["extraction", "cleaning", "transformation"],
    "quality_issues": ["5% missing values in column X"]
  },
  "analysis_performed": {
    "techniques_applied": ["cohort_analysis", "statistical_testing"],
    "models_created": ["customer_churn_model"],
    "statistical_tests": [
      {
        "test": "t-test",
        "variable": "conversion_rate",
        "p_value": 0.001,
        "significant": true
      }
    ],
    "key_metrics": {
      "baseline_conversion": 0.15,
      "treatment_conversion": 0.18,
      "lift": 0.20
    }
  },
  "artifacts_generated": [
    {
      "type": "jupyter_notebook",
      "name": "cohort_analysis.ipynb",
      "description": "Complete analysis workflow and results",
      "file_path": "/artifacts/cohort_analysis.ipynb",
      "file_size": "2.5MB"
    },
    {
      "type": "visualization",
      "name": "cohort_heatmap.png",
      "description": "Customer cohort retention heatmap",
      "file_path": "/artifacts/cohort_heatmap.png",
      "insights": ["Month 1 retention is 85%", "Retention stabilizes at 60% by Month 6"]
    },
    {
      "type": "dataset",
      "name": "customer_cohorts.csv", 
      "description": "Processed cohort data for further analysis",
      "file_path": "/artifacts/customer_cohorts.csv",
      "rows": 15000,
      "columns": 12
    },
    {
      "type": "model",
      "name": "churn_prediction_model.pkl",
      "description": "Trained customer churn prediction model",
      "file_path": "/artifacts/churn_prediction_model.pkl",
      "model_metrics": {
        "accuracy": 0.87,
        "precision": 0.82,
        "recall": 0.79
      }
    }
  ],
  "key_findings": [
    "Customer retention drops significantly in month 2",
    "High-value customers have 40% better retention",
    "Email campaigns improve retention by 15%"
  ],
  "recommendations": [
    "Implement targeted retention campaign for month 2 customers",
    "Develop high-value customer success program", 
    "Increase email campaign frequency"
  ],
  "quality_metrics": {
    "data_completeness": 0.95,
    "statistical_significance": 0.01,
    "business_logic_validation": "passed",
    "peer_review_score": 4.2
  },
  "execution_metadata": {
    "start_time": "2024-01-15T09:00:00Z",
    "end_time": "2024-01-15T14:30:00Z",
    "actual_duration_hours": 5.5,
    "tools_used": ["python", "sql", "jupyter", "plotly"],
    "compute_resources": "4 CPU cores, 16GB RAM"
  },
  "risks_encountered": [
    {
      "risk": "Data quality issues in source system",
      "impact": "medium",
      "mitigation_applied": "Additional data validation and cleaning"
    }
  ],
  "next_steps": [
    "Validate findings with business stakeholders",
    "Implement recommended retention campaigns",
    "Monitor model performance in production"
  ]
}

EXECUTION REQUIREMENTS:
1. Process all required data sources
2. Apply specified analytical techniques
3. Generate all planned artifacts (notebooks, charts, models, datasets)
4. Validate results against success criteria
5. Include detailed quality metrics and metadata
6. Provide actionable business recommendations
7. Document any issues or limitations encountered

Note: In automated mode, generate realistic sample results. In supervised/manual mode, provide detailed execution instructions."""
        
        return prompt
    
    def _get_additional_context(self, plan: Dict[str, Any], input_data: DataAnalysisInput) -> str:
        """Generate additional context string."""
        context_parts = []
        
        # Add data access context
        if input_data.data_access_config:
            config = input_data.data_access_config
            if config.get("warehouse_type"):
                context_parts.append(f"Data Warehouse: {config['warehouse_type']}")
            if config.get("available_tools"):
                context_parts.append(f"Tools: {', '.join(config['available_tools'])}")
        
        # Add methodology context
        methodology = plan.get("methodology", {})
        if methodology.get("analysis_type"):
            context_parts.append(f"Analysis Type: {methodology['analysis_type']}")
        
        context_parts.append(f"Execution Mode: {input_data.execution_mode}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _create_analysis_result(
        self, 
        parsed_response: Dict[str, Any], 
        plan: Dict[str, Any],
        input_data: DataAnalysisInput
    ) -> AnalysisResult:
        """Create structured analysis result from parsed response."""
        # Process artifacts
        artifacts_data = parsed_response.get("artifacts_generated", [])
        artifacts = []
        
        for artifact_data in artifacts_data:
            artifact = AnalysisArtifact(
                type=artifact_data.get("type", "unknown"),
                name=artifact_data.get("name", ""),
                description=artifact_data.get("description", ""),
                file_path=artifact_data.get("file_path", ""),
                file_size=artifact_data.get("file_size", ""),
                metadata=artifact_data.get("metadata", {}),
                insights=artifact_data.get("insights", [])
            )
            artifacts.append(artifact)
        
        # Create analysis result
        analysis_result = AnalysisResult(
            plan_id=parsed_response.get("plan_id", plan.get("node_id", "")),
            execution_status=parsed_response.get("execution_status", "completed"),
            execution_summary=parsed_response.get("execution_summary", ""),
            data_processing=parsed_response.get("data_processing", {}),
            analysis_performed=parsed_response.get("analysis_performed", {}),
            artifacts=artifacts,
            key_findings=parsed_response.get("key_findings", []),
            recommendations=parsed_response.get("recommendations", []),
            quality_metrics=parsed_response.get("quality_metrics", {}),
            execution_metadata=parsed_response.get("execution_metadata", {}),
            risks_encountered=parsed_response.get("risks_encountered", []),
            next_steps=parsed_response.get("next_steps", [])
        )
        
        return analysis_result
    
    def _create_execution_summary(
        self,
        analysis_results: List[AnalysisResult],
        failed_analyses: List[Dict[str, Any]],
        total_plans: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create overall execution summary."""
        successful_count = len(analysis_results)
        failed_count = len(failed_analyses)
        total_count = len(total_plans)
        
        # Aggregate execution time
        total_execution_hours = sum(
            result.execution_metadata.get("actual_duration_hours", 0)
            for result in analysis_results
        )
        
        # Aggregate artifacts
        total_artifacts = sum(len(result.artifacts) for result in analysis_results)
        artifact_types = set()
        for result in analysis_results:
            artifact_types.update(artifact.type for artifact in result.artifacts)
        
        # Quality metrics
        quality_scores = []
        for result in analysis_results:
            if result.quality_metrics.get("peer_review_score"):
                quality_scores.append(result.quality_metrics["peer_review_score"])
        
        avg_quality_score = sum(quality_scores) / len(quality_scores) if quality_scores else None
        
        return {
            "execution_overview": {
                "total_plans": total_count,
                "successful_executions": successful_count,
                "failed_executions": failed_count,
                "success_rate": successful_count / total_count if total_count > 0 else 0
            },
            "resource_utilization": {
                "total_execution_hours": total_execution_hours,
                "average_hours_per_plan": total_execution_hours / successful_count if successful_count > 0 else 0,
                "estimated_vs_actual": "tracking needed"  # Would compare with original estimates
            },
            "artifacts_generated": {
                "total_artifacts": total_artifacts,
                "artifact_types": list(artifact_types),
                "average_artifacts_per_plan": total_artifacts / successful_count if successful_count > 0 else 0
            },
            "quality_assessment": {
                "average_quality_score": avg_quality_score,
                "quality_distribution": "analysis needed",
                "validation_status": f"{successful_count} analyses validated"
            },
            "business_impact": {
                "total_findings": sum(len(result.key_findings) for result in analysis_results),
                "total_recommendations": sum(len(result.recommendations) for result in analysis_results),
                "actionable_insights": "assessment needed"
            },
            "next_actions": {
                "results_requiring_review": [
                    result.plan_id for result in analysis_results
                    if result.execution_status != "completed"
                ],
                "failed_plans_for_retry": [failure["plan_id"] for failure in failed_analyses],
                "stakeholder_presentations_needed": successful_count
            }
        }
    
    def _calculate_confidence(
        self, 
        analysis_results: List[AnalysisResult],
        failed_analyses: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence score based on execution quality."""
        if not analysis_results:
            return 0.1
        
        score = 0.3  # Base score for any successful executions
        
        total_attempts = len(analysis_results) + len(failed_analyses)
        success_rate = len(analysis_results) / total_attempts if total_attempts > 0 else 0
        
        # Success rate bonus
        score += success_rate * 0.3
        
        # Quality metrics bonus
        quality_scores = []
        for result in analysis_results:
            # Check for quality indicators
            if result.quality_metrics.get("data_completeness", 0) > 0.9:
                quality_scores.append(0.1)
            
            if result.quality_metrics.get("statistical_significance", 1.0) < 0.05:
                quality_scores.append(0.1)
            
            if result.quality_metrics.get("business_logic_validation") == "passed":
                quality_scores.append(0.1)
            
            # Check for comprehensive artifacts
            if len(result.artifacts) >= 3:
                quality_scores.append(0.1)
            
            # Check for actionable findings
            if len(result.key_findings) >= 3 and len(result.recommendations) >= 2:
                quality_scores.append(0.1)
        
        if quality_scores:
            avg_quality_bonus = sum(quality_scores) / len(quality_scores)
            score += avg_quality_bonus
        
        return min(1.0, max(0.0, score))