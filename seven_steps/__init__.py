"""
Seven Steps to Poem - AI Agent System for Business Problem Solving

This package implements McKinsey's 7-step problem-solving methodology
as an orchestrated AI agent system that automatically generates
comprehensive business solutions.
"""

__version__ = "1.0.0"
__author__ = "Seven Steps Team"
__email__ = "team@sevenstepstopoem.com"

from .core.config import Settings
from .core.logging import get_logger

__all__ = ["Settings", "get_logger"]