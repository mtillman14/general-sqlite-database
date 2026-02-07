"""Path generation utilities for data pipelines.

This package provides template-based path generation for discovering and
iterating over files organized by metadata (subject, trial, session, etc.).
"""

from pathgen.generator import PathGenerator

__all__ = ["PathGenerator"]
__version__ = "0.1.0"
