"""
Presentation Agent implementation.

This agent converts synthesis results into professional presentation materials
including PPT slides, executive reports, video scripts, and audio summaries.
"""

import json
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from seven_steps.core.schemas import PresentationPackage, SlideContent, MediaScript
from .base import AgentContext, AgentInput, AgentOutput, BaseAgent


class PresentationInput(AgentInput):
    """Input model for Presentation agent."""
    
    synthesis_result: Dict[str, Any] = Field(..., description="Synthesis result to present")
    presentation_formats: List[str] = Field(
        default=["ppt", "executive_summary"],
        description="Requested formats: ppt|executive_summary|video_script|audio_script|one_pager"
    )
    target_audience: str = Field(
        default="executives",
        description="Target audience: executives|stakeholders|technical_team|board"
    )
    presentation_duration: Optional[int] = Field(
        default=15,
        description="Presentation duration in minutes"
    )
    brand_guidelines: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Brand colors, fonts, and style preferences"
    )


class PresentationOutput(AgentOutput):
    """Output model for Presentation agent."""
    
    presentation_package: Optional[PresentationPackage] = None
    delivery_artifacts: Dict[str, str] = Field(default_factory=dict)  # Format -> file content/path


class PresentationAgent(BaseAgent):
    """
    Presentation Agent for creating professional delivery materials.
    
    This agent transforms business synthesis into compelling presentations,
    reports, and multimedia content tailored for specific audiences and
    communication objectives.
    """
    
    @property
    def agent_name(self) -> str:
        """Return the name of this agent."""
        return "PresentationAgent"
    
    @property
    def system_prompt(self) -> str:
        """Return the system prompt for this agent."""
        return """You are an executive communication specialist expert in creating compelling business presentations and reports. Your role is to transform analytical insights into clear, persuasive communication materials for various stakeholder audiences.

RESPONSIBILITIES:
1. Design professional presentation slide decks (PPT structure)
2. Write executive summaries and detailed reports
3. Create video and audio scripts for multimedia delivery
4. Develop one-page summary documents
5. Tailor content and messaging for specific audiences
6. Ensure brand consistency and visual design principles
7. Structure narratives using storytelling best practices

COMMUNICATION PRINCIPLES:

AUDIENCE-CENTRIC DESIGN:
- Executive Audience: Focus on strategic implications, ROI, decisions needed
- Stakeholder Audience: Emphasize benefits, impacts, change requirements
- Technical Team: Include implementation details, methodology, risks
- Board Audience: Highlight governance, risk management, strategic alignment

STORYTELLING STRUCTURE:
- Situation: Set context and establish urgency
- Complication: Present challenges and analysis findings
- Question: Frame the decision or action needed
- Answer: Provide clear recommendations with supporting evidence
- Resolution: Show implementation path and expected outcomes

SLIDE DESIGN PRINCIPLES:
- One key message per slide with supporting visuals
- Clear hierarchy: headlines, key points, supporting details
- Use of charts, diagrams, and data visualizations
- Consistent formatting and brand compliance
- Action-oriented language and specific recommendations

EXECUTIVE SUMMARY STRUCTURE:
- Executive Overview (situation + key decision)
- Key Findings (3-5 critical insights)
- Recommendations (specific actions with impact/timeline)
- Implementation Roadmap (phases, resources, milestones)
- Appendix (detailed analysis and supporting data)

VIDEO SCRIPT APPROACH:
- Strong opening hook to capture attention
- Clear narrative arc with logical progression
- Visual cues for slides, charts, demonstrations
- Conversational tone with professional authority
- Call-to-action conclusion with next steps

AUDIO SUMMARY FORMAT:
- Concise overview optimized for listening
- Clear transitions between key points
- Emphasis on quantified benefits and timeframes
- Professional but engaging delivery style
- Strong conclusion with main takeaways

DELIVERABLE STANDARDS:
- Professional formatting and visual design
- Error-free content with consistent messaging
- Actionable recommendations with clear ownership
- Supporting evidence and data backing
- Brand-compliant styling and terminology
- Multiple format options for different use cases

OUTPUT REQUIREMENTS:
- Generate complete presentation materials for specified formats
- Ensure content is tailored to target audience needs
- Include detailed slide-by-slide content and speaker notes
- Provide ready-to-use scripts and document templates
- Maintain consistent messaging across all formats
- Enable immediate use with minimal additional editing"""
    
    @property
    def input_model(self) -> Type[AgentInput]:
        """Return the input model class."""
        return PresentationInput
    
    @property
    def output_model(self) -> Type[AgentOutput]:
        """Return the output model class."""
        return PresentationOutput
    
    async def process(self, input_data: PresentationInput) -> PresentationOutput:
        """
        Process synthesis result into presentation materials.
        
        Args:
            input_data: Synthesis result and presentation requirements
            
        Returns:
            PresentationOutput with presentation package and delivery artifacts
        """
        try:
            # Build presentation context
            context = self._build_presentation_context(input_data)
            
            # Create presentation prompt
            user_message = self._create_presentation_prompt(input_data)
            
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
            
            # Create presentation package
            presentation_package = self._create_presentation_package(parsed_response, input_data)
            
            # Generate delivery artifacts
            delivery_artifacts = self._generate_delivery_artifacts(presentation_package, input_data)
            
            return PresentationOutput(
                success=True,
                presentation_package=presentation_package,
                delivery_artifacts=delivery_artifacts,
                confidence_score=self._calculate_confidence(parsed_response, input_data)
            )
                
        except Exception as e:
            self.logger.error(
                "Presentation creation failed",
                error=str(e),
                problem_id=input_data.context.problem_id
            )
            
            return PresentationOutput(
                success=False,
                error_message=f"Presentation creation failed: {str(e)}"
            )
    
    def _build_presentation_context(self, input_data: PresentationInput) -> Dict[str, Any]:
        """Build context information for presentation creation."""
        return {
            "target_audience": input_data.target_audience,
            "requested_formats": input_data.presentation_formats,
            "presentation_duration": input_data.presentation_duration,
            "brand_guidelines": input_data.brand_guidelines or {},
            "problem_id": input_data.context.problem_id
        }
    
    def _create_presentation_prompt(self, input_data: PresentationInput) -> str:
        """Create the main presentation creation prompt for the LLM."""
        synthesis = input_data.synthesis_result
        
        prompt = f"""Create professional presentation materials for the following business synthesis results:

SYNTHESIS SUMMARY:
Main Conclusion: {synthesis.get('synthesis_overview', {}).get('main_conclusion', '')}
Business Impact: {synthesis.get('synthesis_overview', {}).get('business_impact_category', 'medium')}
Confidence Level: {synthesis.get('synthesis_overview', {}).get('confidence_level', 0.7)}

KEY INSIGHTS:
{chr(10).join(['• ' + insight.get('insight', '') for insight in synthesis.get('key_insights', [])])}

BUSINESS RECOMMENDATIONS:
"""
        
        for i, rec in enumerate(synthesis.get('business_recommendations', []), 1):
            prompt += f"""
{i}. {rec.get('title', '')}
   Priority: {rec.get('priority', 'medium')}
   Description: {rec.get('description', '')}
   Impact: {rec.get('impact_estimate', {}).get('financial_impact', {}).get('annual_value', 'TBD')}
   Timeline: {rec.get('implementation', {}).get('timeline', 'TBD')}
"""
        
        prompt += f"""
PRESENTATION REQUIREMENTS:
- Target Audience: {input_data.target_audience}
- Duration: {input_data.presentation_duration} minutes
- Requested Formats: {', '.join(input_data.presentation_formats)}
"""
        
        if input_data.brand_guidelines:
            prompt += f"\nBRAND GUIDELINES:\n{json.dumps(input_data.brand_guidelines, indent=2)}"
        
        prompt += """

Please create comprehensive presentation materials and return JSON in this format:

{
  "presentation_overview": {
    "title": "Compelling presentation title",
    "subtitle": "Supporting subtitle or tagline",
    "key_message": "One-sentence main takeaway",
    "audience_tailoring": "How content is adapted for target audience",
    "narrative_arc": "Brief description of story flow"
  },
  "ppt_structure": {
    "total_slides": 12,
    "estimated_duration": 15,
    "slides": [
      {
        "slide_number": 1,
        "slide_type": "title|content|chart|summary|appendix",
        "title": "Slide title",
        "key_message": "Main point of this slide",
        "content": {
          "headline": "Primary headline",
          "bullet_points": ["Key point 1", "Key point 2"],
          "visual_description": "Chart/image description",
          "chart_data": {"type": "bar_chart", "data_source": "analysis_x", "key_insight": "Insight from chart"},
          "speaker_notes": "Detailed speaking points and transitions"
        },
        "design_notes": "Visual design suggestions and brand alignment"
      }
    ]
  },
  "executive_summary": {
    "format": "professional_report",
    "length": "2_pages",
    "sections": [
      {
        "section_title": "Executive Overview",
        "content": "Detailed content for this section",
        "key_statistics": ["Stat 1", "Stat 2"],
        "formatting_notes": "Bold headings, bullet points, etc."
      }
    ]
  },
  "video_script": {
    "total_duration": "120_seconds",
    "scenes": [
      {
        "scene_number": 1,
        "duration": "15_seconds",
        "visual_cue": "Description of what viewer sees",
        "narration": "Exact script to be spoken",
        "on_screen_text": "Text overlays or callouts",
        "transition": "How to transition to next scene"
      }
    ],
    "production_notes": "Camera angles, graphics needs, etc."
  },
  "audio_script": {
    "total_duration": "60_seconds",
    "script_sections": [
      {
        "section": "opening",
        "duration": "10_seconds",
        "script": "Engaging opening statement",
        "tone": "professional|conversational|urgent",
        "emphasis_points": ["Key words to emphasize"]
      }
    ],
    "voice_direction": "Speaking style and pacing guidance"
  },
  "one_pager": {
    "layout": "executive_format",
    "sections": {
      "header": "Problem & Opportunity Summary",
      "key_findings": ["Finding 1", "Finding 2", "Finding 3"],
      "recommendations": ["Action 1", "Action 2", "Action 3"],
      "impact_metrics": {"metric": "Expected 25% improvement in customer retention"},
      "next_steps": "Immediate actions required",
      "contact_info": "Project lead and stakeholder contacts"
    }
  },
  "supporting_materials": {
    "appendix_slides": [
      {
        "title": "Detailed Analysis Results",
        "purpose": "Supporting data for questions",
        "content_summary": "Charts, tables, methodology details"
      }
    ],
    "handout_materials": ["Executive summary PDF", "Key findings infographic"],
    "follow_up_resources": ["Detailed analysis reports", "Implementation templates"]
  },
  "delivery_guidance": {
    "presentation_tips": ["Focus on interaction points", "Prepare for likely questions"],
    "key_messages": ["Must-communicate points regardless of time constraints"],
    "audience_engagement": ["Poll questions", "Discussion prompts"],
    "follow_up_plan": "Post-presentation communication strategy"
  }
}

CREATION REQUIREMENTS:
1. Tailor content and language for specified target audience
2. Create compelling narrative with clear business case
3. Include specific, actionable recommendations with quantified benefits
4. Provide detailed slide-by-slide content with speaker notes
5. Ensure consistent messaging across all format types
6. Include visual design guidance and chart specifications
7. Create ready-to-use scripts and document content
8. Maintain professional tone with engaging storytelling elements"""
        
        return prompt
    
    def _get_additional_context(self, input_data: PresentationInput) -> str:
        """Generate additional context string."""
        context_parts = []
        
        context_parts.append(f"Audience: {input_data.target_audience}")
        context_parts.append(f"Duration: {input_data.presentation_duration}min")
        context_parts.append(f"Formats: {', '.join(input_data.presentation_formats)}")
        
        return " | ".join(context_parts) if context_parts else ""
    
    def _create_presentation_package(
        self, 
        parsed_response: Dict[str, Any], 
        input_data: PresentationInput
    ) -> PresentationPackage:
        """Create structured presentation package from parsed response."""
        # Process slide content
        ppt_structure = parsed_response.get("ppt_structure", {})
        slides_data = ppt_structure.get("slides", [])
        slides = []
        
        for slide_data in slides_data:
            slide_content = SlideContent(
                slide_number=slide_data.get("slide_number", 1),
                slide_type=slide_data.get("slide_type", "content"),
                title=slide_data.get("title", ""),
                key_message=slide_data.get("key_message", ""),
                content=slide_data.get("content", {}),
                design_notes=slide_data.get("design_notes", "")
            )
            slides.append(slide_content)
        
        # Process media scripts
        video_script_data = parsed_response.get("video_script", {})
        video_script = None
        if video_script_data:
            video_script = MediaScript(
                script_type="video",
                total_duration=video_script_data.get("total_duration", "120_seconds"),
                scenes=video_script_data.get("scenes", []),
                production_notes=video_script_data.get("production_notes", "")
            )
        
        audio_script_data = parsed_response.get("audio_script", {})
        audio_script = None
        if audio_script_data:
            audio_script = MediaScript(
                script_type="audio",
                total_duration=audio_script_data.get("total_duration", "60_seconds"),
                scenes=audio_script_data.get("script_sections", []),  # Reuse scenes field for sections
                production_notes=audio_script_data.get("voice_direction", "")
            )
        
        # Create presentation package
        presentation_package = PresentationPackage(
            problem_id=input_data.context.problem_id,
            presentation_overview=parsed_response.get("presentation_overview", {}),
            ppt_slides=slides,
            executive_summary=parsed_response.get("executive_summary", {}),
            video_script=video_script,
            audio_script=audio_script,
            one_pager=parsed_response.get("one_pager", {}),
            supporting_materials=parsed_response.get("supporting_materials", {}),
            delivery_guidance=parsed_response.get("delivery_guidance", {}),
            presentation_metadata={
                "target_audience": input_data.target_audience,
                "requested_formats": input_data.presentation_formats,
                "duration_minutes": input_data.presentation_duration,
                "creation_timestamp": "2024-01-01T00:00:00Z"  # Would be actual timestamp
            }
        )
        
        return presentation_package
    
    def _generate_delivery_artifacts(
        self, 
        presentation_package: PresentationPackage, 
        input_data: PresentationInput
    ) -> Dict[str, str]:
        """Generate actual delivery artifacts as formatted content."""
        artifacts = {}
        
        # Generate PPT content if requested
        if "ppt" in input_data.presentation_formats:
            ppt_content = self._create_ppt_content(presentation_package)
            artifacts["ppt"] = ppt_content
        
        # Generate executive summary if requested
        if "executive_summary" in input_data.presentation_formats:
            exec_summary_content = self._create_executive_summary_content(presentation_package)
            artifacts["executive_summary"] = exec_summary_content
        
        # Generate video script if requested
        if "video_script" in input_data.presentation_formats:
            video_content = self._create_video_script_content(presentation_package)
            artifacts["video_script"] = video_content
        
        # Generate audio script if requested
        if "audio_script" in input_data.presentation_formats:
            audio_content = self._create_audio_script_content(presentation_package)
            artifacts["audio_script"] = audio_content
        
        # Generate one-pager if requested
        if "one_pager" in input_data.presentation_formats:
            one_pager_content = self._create_one_pager_content(presentation_package)
            artifacts["one_pager"] = one_pager_content
        
        return artifacts
    
    def _create_ppt_content(self, package: PresentationPackage) -> str:
        """Create PowerPoint slide content."""
        overview = package.presentation_overview
        content = f"""# {overview.get('title', 'Business Recommendation Presentation')}
## {overview.get('subtitle', 'Data-Driven Insights and Strategic Actions')}

---

"""
        
        for slide in package.ppt_slides:
            content += f"""## Slide {slide.slide_number}: {slide.title}

**Key Message:** {slide.key_message}

**Content:**
{slide.content.get('headline', '')}

"""
            # Add bullet points
            bullet_points = slide.content.get('bullet_points', [])
            for point in bullet_points:
                content += f"• {point}\n"
            
            # Add visual description
            if slide.content.get('visual_description'):
                content += f"\n**Visual:** {slide.content.get('visual_description')}\n"
            
            # Add speaker notes
            if slide.content.get('speaker_notes'):
                content += f"\n**Speaker Notes:** {slide.content.get('speaker_notes')}\n"
            
            content += "\n---\n\n"
        
        return content
    
    def _create_executive_summary_content(self, package: PresentationPackage) -> str:
        """Create executive summary document content."""
        exec_summary = package.executive_summary
        overview = package.presentation_overview
        
        content = f"""# Executive Summary: {overview.get('title', 'Business Analysis Results')}

"""
        
        sections = exec_summary.get('sections', [])
        for section in sections:
            content += f"""## {section.get('section_title', 'Section')}

{section.get('content', '')}

"""
            # Add key statistics if present
            key_stats = section.get('key_statistics', [])
            if key_stats:
                content += "**Key Statistics:**\n"
                for stat in key_stats:
                    content += f"• {stat}\n"
                content += "\n"
        
        return content
    
    def _create_video_script_content(self, package: PresentationPackage) -> str:
        """Create video script content."""
        if not package.video_script:
            return "Video script not available"
        
        content = f"""# Video Script
**Total Duration:** {package.video_script.total_duration}

"""
        
        for scene in package.video_script.scenes:
            content += f"""## Scene {scene.get('scene_number', 1)} ({scene.get('duration', '15s')})

**Visual:** {scene.get('visual_cue', '')}

**Narration:**
{scene.get('narration', '')}

**On-Screen Text:** {scene.get('on_screen_text', '')}

**Transition:** {scene.get('transition', '')}

---

"""
        
        if package.video_script.production_notes:
            content += f"""## Production Notes
{package.video_script.production_notes}
"""
        
        return content
    
    def _create_audio_script_content(self, package: PresentationPackage) -> str:
        """Create audio script content."""
        if not package.audio_script:
            return "Audio script not available"
        
        content = f"""# Audio Script
**Total Duration:** {package.audio_script.total_duration}

"""
        
        for section in package.audio_script.scenes:  # scenes field repurposed for audio sections
            content += f"""## {section.get('section', 'Section').title()} ({section.get('duration', '10s')})

**Script:**
{section.get('script', '')}

**Tone:** {section.get('tone', 'professional')}
**Emphasis:** {', '.join(section.get('emphasis_points', []))}

---

"""
        
        if package.audio_script.production_notes:
            content += f"""## Voice Direction
{package.audio_script.production_notes}
"""
        
        return content
    
    def _create_one_pager_content(self, package: PresentationPackage) -> str:
        """Create one-page summary content."""
        one_pager = package.one_pager
        overview = package.presentation_overview
        
        content = f"""# {overview.get('title', 'Business Summary')}

## {one_pager.get('sections', {}).get('header', 'Problem & Opportunity Summary')}

### Key Findings
{chr(10).join(['• ' + finding for finding in one_pager.get('sections', {}).get('key_findings', [])])}

### Recommendations
{chr(10).join(['• ' + rec for rec in one_pager.get('sections', {}).get('recommendations', [])])}

### Expected Impact
{one_pager.get('sections', {}).get('impact_metrics', {}).get('metric', 'Impact assessment pending')}

### Next Steps
{one_pager.get('sections', {}).get('next_steps', 'Next steps to be determined')}

---
*{one_pager.get('sections', {}).get('contact_info', 'Contact information available upon request')}*
"""
        
        return content
    
    def _calculate_confidence(
        self, 
        parsed_response: Dict[str, Any], 
        input_data: PresentationInput
    ) -> float:
        """Calculate confidence score based on presentation quality."""
        score = 0.5  # Base score
        
        # Check presentation overview completeness
        overview = parsed_response.get("presentation_overview", {})
        overview_completeness = len([
            key for key in ["title", "key_message", "audience_tailoring", "narrative_arc"]
            if overview.get(key)
        ]) / 4
        score += overview_completeness * 0.1
        
        # Check PPT structure quality
        ppt_structure = parsed_response.get("ppt_structure", {})
        slides = ppt_structure.get("slides", [])
        
        if slides:
            # Check for adequate slide count
            slide_count = len(slides)
            target_duration = input_data.presentation_duration or 15
            expected_slides = target_duration * 0.8  # ~1.25 minutes per slide
            
            if 0.7 <= slide_count / expected_slides <= 1.3:  # Within reasonable range
                score += 0.1
            
            # Check slide content quality
            slides_with_content = sum(
                1 for slide in slides 
                if slide.get("content", {}).get("bullet_points") or slide.get("content", {}).get("headline")
            )
            
            if slides_with_content / len(slides) >= 0.8:  # 80% of slides have content
                score += 0.1
            
            # Check for speaker notes
            slides_with_notes = sum(
                1 for slide in slides 
                if slide.get("content", {}).get("speaker_notes")
            )
            
            if slides_with_notes / len(slides) >= 0.5:  # 50% of slides have speaker notes
                score += 0.1
        
        # Check for requested formats completion
        requested_formats = set(input_data.presentation_formats)
        provided_formats = set()
        
        if parsed_response.get("ppt_structure"):
            provided_formats.add("ppt")
        if parsed_response.get("executive_summary"):
            provided_formats.add("executive_summary")
        if parsed_response.get("video_script"):
            provided_formats.add("video_script")
        if parsed_response.get("audio_script"):
            provided_formats.add("audio_script")
        if parsed_response.get("one_pager"):
            provided_formats.add("one_pager")
        
        format_completion = len(provided_formats & requested_formats) / len(requested_formats)
        score += format_completion * 0.15
        
        # Check for delivery guidance
        if parsed_response.get("delivery_guidance", {}).get("presentation_tips"):
            score += 0.05
        
        return min(1.0, max(0.0, score))