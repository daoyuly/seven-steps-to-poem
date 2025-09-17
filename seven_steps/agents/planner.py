"""
Planner Agent implementation.

This agent creates detailed, executable analysis plans for prioritized
issue tree nodes, including data requirements, methods, timelines, and deliverables.
"""

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from seven_steps.core.schemas import AnalysisPlan, DataRequirement, Milestone
from .base import AgentContext, AgentInput, AgentOutput, BaseAgent


class PlannerInput(AgentInput):
    """Input model for Planner agent."""
    
    priority_matrix: Dict[str, Any] = Field(..., description="Priority matrix from previous step")
    selected_nodes: Optional[List[str]] = Field(
        default=None, 
        description="Specific nodes to plan for (if not provided, plans top priorities)"
    )
    planning_horizon: str = Field(default="4_weeks", description="Planning timeframe")
    available_resources: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Available team members, tools, and data sources"
    )


class PlannerOutput(AgentOutput):
    """Output model for Planner agent."""
    
    analysis_plans: List[AnalysisPlan] = Field(default_factory=list)
    execution_timeline: Optional[Dict[str, Any]] = None
    resource_allocation: Optional[Dict[str, Any]] = None


class PlannerAgent(BaseAgent):
    """
    Planner Agent for detailed analysis planning.
    
    This agent creates comprehensive, executable plans for conducting
    business analyses, including data sourcing, methodology, timelines,
    and success criteria.
    """
    
    @property
    def agent_name(self) -> str:
        """Return the name of this agent."""
        return "PlannerAgent"
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return """You are a senior data analytics project manager expert in creating detailed, executable analysis plans. Your role is to convert high-level analysis requirements into actionable project plans.

RESPONSIBILITIES:
1. Create detailed analysis plans for each prioritized issue
2. Specify exact data requirements and sources
3. Define analytical methodologies and tools
4. Establish timelines with realistic milestones  
5. Identify required skills and team members
6. Define success criteria and deliverables
7. Assess risks and mitigation strategies

PLANNING COMPONENTS:

DATA REQUIREMENTS:
- Specific data sources (tables, APIs, files)
- Data quality requirements and validation rules
- Sample sizes and time periods needed
- Access permissions and data governance
- Data transformation and preparation steps

ANALYTICAL METHODOLOGY:
- Statistical methods and techniques
- Tools and software required (SQL, Python, R, Tableau)
- Analysis steps and validation procedures
- Quality assurance checkpoints
- Peer review requirements

PROJECT MANAGEMENT:
- Task breakdown with dependencies
- Resource allocation and skill requirements
- Timeline with realistic estimates
- Risk assessment and mitigation plans
- Communication and reporting schedule

DELIVERABLES DEFINITION:
- Analysis artifacts (reports, charts, models)
- Code documentation and reproducibility
- Presentation materials and formats
- Stakeholder-specific outputs
- Archive and knowledge management

OUTPUT REQUIREMENTS:
- Return detailed, actionable analysis plans
- Include specific technical specifications
- Provide realistic time estimates
- Consider resource constraints and dependencies
- Enable parallel execution where possible"""
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class."""
        return PlannerInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class."""
        return PlannerOutput
    
    async def process(self, input_data: PlannerInput) -> PlannerOutput:
        """
        Process priority matrix into detailed analysis plans.
        
        Args:
            input_data: Priority matrix and planning parameters
            
        Returns:
            PlannerOutput with detailed analysis plans and timelines
        """
        try:
            # Determine which nodes to plan for
            priority_scores = input_data.priority_matrix.get("priority_scores", [])
            selected_nodes = self._select_nodes_to_plan(input_data, priority_scores)
            
            # Build context for planning
            context = self._build_planning_context(input_data, selected_nodes)
            
            # Create the planning prompt
            user_message = self._create_planning_prompt(input_data, selected_nodes)
            
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
            
            # Create analysis plans
            analysis_plans = self._create_analysis_plans(parsed_response, selected_nodes, input_data)
            
            # Generate execution timeline
            execution_timeline = self._create_execution_timeline(analysis_plans, input_data)
            
            # Calculate resource allocation
            resource_allocation = self._calculate_detailed_resource_allocation(analysis_plans, input_data)
            
            return PlannerOutput(
                success=True,
                analysis_plans=analysis_plans,
                execution_timeline=execution_timeline,
                resource_allocation=resource_allocation,
                confidence_score=self._calculate_confidence(parsed_response, input_data)
            )
                
        except Exception as e:
            self.logger.error(
                "Analysis planning failed",
                error=str(e),
                problem_id=input_data.context.problem_id
            )
            
            return PlannerOutput(
                success=False,
                error_message=f"Analysis planning failed: {str(e)}"
            )
    
    def _select_nodes_to_plan(self, input_data: PlannerInput, priority_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select which nodes to create detailed plans for."""
        if input_data.selected_nodes:
            # Filter to selected nodes only
            return [
                score for score in priority_scores 
                if score.get("node_id") in input_data.selected_nodes
            ]
        else:
            # Select top priority nodes based on category and score
            selected = []
            
            # Always include quick wins
            quick_wins = [s for s in priority_scores if s.get("priority_category") == "quick_win"]
            selected.extend(quick_wins)
            
            # Include top 3 major projects
            major_projects = [s for s in priority_scores if s.get("priority_category") == "major_project"]
            major_projects.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
            selected.extend(major_projects[:3])
            
            # Include top 2 fill-ins if we have capacity
            if len(selected) < 8:
                fill_ins = [s for s in priority_scores if s.get("priority_category") == "fill_in"]
                fill_ins.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
                selected.extend(fill_ins[:2])
            
            return selected
    
    def _build_planning_context(self, input_data: PlannerInput, selected_nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build context information for detailed planning."""
        return {
            "planning_horizon": input_data.planning_horizon,
            "total_nodes_to_plan": len(selected_nodes),
            "available_resources": input_data.available_resources or {},
            "problem_id": input_data.context.problem_id,
            "priority_matrix_summary": {
                "total_nodes": len(input_data.priority_matrix.get("priority_scores", [])),
                "selected_for_planning": len(selected_nodes)
            }
        }
    
    def _create_planning_prompt(self, input_data: PlannerInput, selected_nodes: List[Dict[str, Any]]) -> str:
        """Create the main planning prompt for the LLM."""
        prompt = f"""Create detailed, executable analysis plans for the following {len(selected_nodes)} prioritized analysis nodes:

NODES TO PLAN:
"""
        
        for i, node in enumerate(selected_nodes, 1):
            prompt += f"""
{i}. NODE ID: {node.get('node_id')}
   Title: {node.get('title')}
   Impact Score: {node.get('impact_score')}/10
   Feasibility Score: {node.get('feasibility_score')}/10
   Priority Category: {node.get('priority_category')}
   Estimated Effort: {node.get('estimated_effort_hours')} hours
   Required Skills: {node.get('required_skills', [])}
   Data Confidence: {node.get('data_confidence')}
   Dependencies: {node.get('dependencies', [])}
   Risk Factors: {node.get('risk_factors', [])}
"""
        
        prompt += f"\nPLANNING CONSTRAINTS:\n"
        prompt += f"- Planning Horizon: {input_data.planning_horizon}\n"
        
        if input_data.available_resources:
            prompt += f"- Available Resources: {json.dumps(input_data.available_resources, indent=2)}\n"
        
        prompt += """
Please create detailed analysis plans and return JSON in this format:

{
  "analysis_plans": [
    {
      "node_id": "1.1",
      "plan_title": "Descriptive plan title",
      "objective": "Clear statement of what this analysis aims to achieve",
      "success_criteria": ["Specific, measurable success criteria"],
      "data_requirements": [
        {
          "source_name": "customer_events",
          "source_type": "database_table|api|file|survey",
          "description": "Description of data needed",
          "sample_size": 50000,
          "time_period": "last_12_months",
          "access_method": "SQL query via data warehouse",
          "quality_requirements": "95% completeness, no duplicates",
          "estimated_prep_hours": 8
        }
      ],
      "methodology": {
        "analysis_type": "cohort_analysis|regression|segmentation|descriptive",
        "techniques": ["specific statistical methods"],
        "tools_required": ["python", "sql", "tableau"],
        "validation_approach": "cross-validation|holdout|business_logic",
        "quality_checks": ["specific validation steps"]
      },
      "tasks": [
        {
          "task_id": "T1",
          "title": "Data extraction and cleaning",
          "description": "Detailed task description",
          "estimated_hours": 16,
          "required_skills": ["sql", "python"],
          "dependencies": [],
          "deliverables": ["clean_dataset.csv", "data_quality_report.md"]
        }
      ],
      "milestones": [
        {
          "milestone_id": "M1",
          "title": "Data preparation complete",
          "description": "All required data extracted, cleaned, and validated",
          "target_date": "2024-02-15",
          "success_criteria": ["Data quality >95%", "All variables present"],
          "dependencies": ["T1", "T2"]
        }
      ],
      "timeline": {
        "estimated_duration_days": 14,
        "start_date": "2024-02-01",
        "end_date": "2024-02-15",
        "critical_path": ["T1", "T3", "T5"]
      },
      "resources": {
        "primary_analyst": "data_scientist",
        "supporting_roles": ["sql_developer", "business_analyst"],
        "total_person_hours": 40,
        "tools_budget": 500,
        "external_dependencies": ["IT for database access"]
      },
      "deliverables": [
        {
          "name": "Analysis Report",
          "format": "jupyter_notebook|pdf|powerpoint",
          "audience": "business_stakeholders",
          "key_components": ["executive_summary", "findings", "recommendations"]
        }
      ],
      "risks": [
        {
          "risk": "Data quality issues",
          "probability": "medium",
          "impact": "high", 
          "mitigation": "Early data profiling and validation"
        }
      ],
      "quality_assurance": {
        "peer_review_required": true,
        "code_review_checklist": ["documentation", "testing", "reproducibility"],
        "stakeholder_validation": "weekly check-ins"
      }
    }
  ],
  "planning_summary": {
    "total_estimated_duration": "6 weeks",
    "parallel_opportunities": [["1.1", "1.2"], ["2.1", "2.2"]],
    "resource_bottlenecks": ["Data access permissions", "Senior analyst availability"],
    "critical_dependencies": ["External data source setup"]
  }
}

REQUIREMENTS:
1. Plans must be executable with specific, actionable tasks
2. Include realistic time estimates based on complexity
3. Specify exact data requirements and access methods
4. Define clear success criteria and deliverables
5. Identify risks and mitigation strategies
6. Consider dependencies and parallel execution opportunities
7. Include quality assurance and peer review processes"""
        
        return prompt
    
    def _get_additional_context(self, input_data: PlannerInput) -> str:
        """Generate additional context string."""
        context_parts = []
        
        # Add resource context
        if input_data.available_resources:
            resources = input_data.available_resources
            if resources.get("team_skills"):
                context_parts.append(f"Available Skills: {', '.join(resources['team_skills'])}")
            if resources.get("tools"):
                context_parts.append(f"Available Tools: {', '.join(resources['tools'])}")
        
        context_parts.append(f"Planning Horizon: {input_data.planning_horizon}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _create_analysis_plans(
        self, 
        parsed_response: Dict[str, Any], 
        selected_nodes: List[Dict[str, Any]], 
        input_data: PlannerInput
    ) -> List[AnalysisPlan]:
        """Create structured analysis plans from parsed response."""
        plans_data = parsed_response.get("analysis_plans", [])
        analysis_plans = []
        
        for plan_data in plans_data:
            # Process data requirements
            data_requirements = []
            for dr_data in plan_data.get("data_requirements", []):
                data_req = DataRequirement(
                    source_name=dr_data.get("source_name", ""),
                    source_type=dr_data.get("source_type", "database_table"),
                    description=dr_data.get("description", ""),
                    sample_size=dr_data.get("sample_size"),
                    time_period=dr_data.get("time_period", ""),
                    access_method=dr_data.get("access_method", ""),
                    quality_requirements=dr_data.get("quality_requirements", ""),
                    estimated_prep_hours=dr_data.get("estimated_prep_hours", 8)
                )
                data_requirements.append(data_req)
            
            # Process milestones
            milestones = []
            for ms_data in plan_data.get("milestones", []):
                milestone = Milestone(
                    milestone_id=ms_data.get("milestone_id", ""),
                    title=ms_data.get("title", ""),
                    description=ms_data.get("description", ""),
                    target_date=ms_data.get("target_date", ""),
                    success_criteria=ms_data.get("success_criteria", []),
                    dependencies=ms_data.get("dependencies", [])
                )
                milestones.append(milestone)
            
            # Create analysis plan
            analysis_plan = AnalysisPlan(
                node_id=plan_data.get("node_id", ""),
                plan_title=plan_data.get("plan_title", ""),
                objective=plan_data.get("objective", ""),
                success_criteria=plan_data.get("success_criteria", []),
                data_requirements=data_requirements,
                methodology=plan_data.get("methodology", {}),
                tasks=plan_data.get("tasks", []),
                milestones=milestones,
                timeline=plan_data.get("timeline", {}),
                resources=plan_data.get("resources", {}),
                deliverables=plan_data.get("deliverables", []),
                risks=plan_data.get("risks", []),
                quality_assurance=plan_data.get("quality_assurance", {})
            )
            analysis_plans.append(analysis_plan)
        
        return analysis_plans
    
    def _create_execution_timeline(
        self, 
        analysis_plans: List[AnalysisPlan], 
        input_data: PlannerInput
    ) -> Dict[str, Any]:
        """Create overall execution timeline for all plans."""
        # Calculate total duration and identify parallel opportunities
        total_days = 0
        parallel_groups = []
        critical_path = []
        
        # Sort plans by dependencies and priority
        independent_plans = []
        dependent_plans = []
        
        for plan in analysis_plans:
            has_dependencies = any(
                dep in [p.node_id for p in analysis_plans]
                for dep in plan.timeline.get("dependencies", [])
            )
            
            if has_dependencies:
                dependent_plans.append(plan)
            else:
                independent_plans.append(plan)
        
        # Calculate timeline
        if independent_plans:
            # Independent plans can run in parallel
            max_independent_duration = max(
                plan.timeline.get("estimated_duration_days", 14) 
                for plan in independent_plans
            )
            total_days = max_independent_duration
            
            # Group independent plans for parallel execution
            parallel_groups.append([plan.node_id for plan in independent_plans])
        
        if dependent_plans:
            # Dependent plans run sequentially after independent ones
            sequential_duration = sum(
                plan.timeline.get("estimated_duration_days", 14)
                for plan in dependent_plans
            )
            total_days += sequential_duration
            
            # Add to critical path
            critical_path.extend([plan.node_id for plan in dependent_plans])
        
        # Identify resource bottlenecks
        resource_demand = {}
        for plan in analysis_plans:
            resources = plan.resources
            primary_analyst = resources.get("primary_analyst", "analyst")
            person_hours = resources.get("total_person_hours", 40)
            
            if primary_analyst not in resource_demand:
                resource_demand[primary_analyst] = 0
            resource_demand[primary_analyst] += person_hours
        
        return {
            "total_estimated_days": total_days,
            "total_estimated_weeks": (total_days + 6) // 7,  # Round up
            "parallel_execution_groups": parallel_groups,
            "critical_path": critical_path,
            "resource_demand_by_role": resource_demand,
            "timeline_by_plan": {
                plan.node_id: {
                    "duration_days": plan.timeline.get("estimated_duration_days", 14),
                    "start_date": plan.timeline.get("start_date", ""),
                    "end_date": plan.timeline.get("end_date", ""),
                    "dependencies": plan.timeline.get("dependencies", [])
                }
                for plan in analysis_plans
            }
        }
    
    def _calculate_detailed_resource_allocation(
        self, 
        analysis_plans: List[AnalysisPlan], 
        input_data: PlannerInput
    ) -> Dict[str, Any]:
        """Calculate detailed resource allocation for all plans."""
        # Aggregate resource requirements
        total_person_hours = sum(plan.resources.get("total_person_hours", 40) for plan in analysis_plans)
        
        # Aggregate skills needed
        all_skills = set()
        skill_hours = {}
        for plan in analysis_plans:
            for task in plan.tasks:
                for skill in task.get("required_skills", []):
                    all_skills.add(skill)
                    if skill not in skill_hours:
                        skill_hours[skill] = 0
                    skill_hours[skill] += task.get("estimated_hours", 8)
        
        # Aggregate tools and budget
        total_tools_budget = sum(plan.resources.get("tools_budget", 0) for plan in analysis_plans)
        
        # Required roles
        roles_needed = set()
        role_hours = {}
        for plan in analysis_plans:
            primary_role = plan.resources.get("primary_analyst", "analyst")
            supporting_roles = plan.resources.get("supporting_roles", [])
            
            roles_needed.add(primary_role)
            roles_needed.update(supporting_roles)
            
            # Estimate hours per role (primary gets 70%, supporting split remaining 30%)
            plan_hours = plan.resources.get("total_person_hours", 40)
            if primary_role not in role_hours:
                role_hours[primary_role] = 0
            role_hours[primary_role] += plan_hours * 0.7
            
            if supporting_roles:
                supporting_hours_each = (plan_hours * 0.3) / len(supporting_roles)
                for role in supporting_roles:
                    if role not in role_hours:
                        role_hours[role] = 0
                    role_hours[role] += supporting_hours_each
        
        # External dependencies
        external_deps = set()
        for plan in analysis_plans:
            external_deps.update(plan.resources.get("external_dependencies", []))
        
        return {
            "total_person_hours": total_person_hours,
            "total_person_weeks": total_person_hours / 40,  # Assuming 40 hour work weeks
            "required_skills": list(all_skills),
            "skill_demand_hours": skill_hours,
            "required_roles": list(roles_needed),
            "role_demand_hours": role_hours,
            "total_tools_budget": total_tools_budget,
            "external_dependencies": list(external_deps),
            "peak_resource_demand": self._calculate_peak_demand(analysis_plans),
            "resource_constraints_impact": input_data.available_resources or {}
        }
    
    def _calculate_peak_demand(self, analysis_plans: List[AnalysisPlan]) -> Dict[str, Any]:
        """Calculate peak resource demand periods."""
        # This is a simplified calculation - in reality would need detailed scheduling
        # For now, assume all independent plans start simultaneously
        
        independent_plans = [
            plan for plan in analysis_plans 
            if not plan.timeline.get("dependencies", [])
        ]
        
        if not independent_plans:
            return {"peak_concurrent_analysts": 1, "peak_period": "start"}
        
        return {
            "peak_concurrent_analysts": len(independent_plans),
            "peak_period": "first_2_weeks",
            "peak_skills_needed": list(set().union(
                *[
                    set().union(*[task.get("required_skills", []) for task in plan.tasks])
                    for plan in independent_plans
                ]
            ))
        }
    
    def _calculate_confidence(
        self, 
        parsed_response: Dict[str, Any], 
        input_data: PlannerInput
    ) -> float:
        """Calculate confidence score based on plan quality."""
        score = 0.5  # Base score
        
        plans = parsed_response.get("analysis_plans", [])
        
        if not plans:
            return 0.2
        
        # Check for detailed task breakdown
        avg_tasks_per_plan = sum(len(plan.get("tasks", [])) for plan in plans) / len(plans)
        if avg_tasks_per_plan >= 5:
            score += 0.1
        
        # Check for realistic time estimates
        has_reasonable_estimates = all(
            10 <= plan.get("timeline", {}).get("estimated_duration_days", 0) <= 60
            for plan in plans
        )
        if has_reasonable_estimates:
            score += 0.1
        
        # Check for specific data requirements
        has_detailed_data_reqs = all(
            len(plan.get("data_requirements", [])) > 0 and
            all(len(dr.get("description", "")) > 20 for dr in plan.get("data_requirements", []))
            for plan in plans
        )
        if has_detailed_data_reqs:
            score += 0.15
        
        # Check for risk assessment
        has_risk_assessment = all(
            len(plan.get("risks", [])) > 0
            for plan in plans
        )
        if has_risk_assessment:
            score += 0.1
        
        # Check for quality assurance process
        has_qa_process = all(
            plan.get("quality_assurance", {}).get("peer_review_required", False)
            for plan in plans
        )
        if has_qa_process:
            score += 0.05
        
        return min(1.0, max(0.0, score))