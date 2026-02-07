"""Path generation utilities for loading raw data files.

This module re-exports the PathGenerator from the pathgen package
for backwards compatibility within scidb.
"""

from pathgen import PathGenerator

__all__ = ["PathGenerator"]
