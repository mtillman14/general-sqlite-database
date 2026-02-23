"""Path generation utilities for loading raw data files.

This module re-exports the PathGenerator from the pathgen package
for convenience within scidb.
"""

from scipathgen import PathGenerator

__all__ = ["PathGenerator"]
