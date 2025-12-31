from .thunk import Thunk
from typing import Any

def evaluate():
    """Evaluate all thunks in the current pipeline."""
    # Get all Thunk instances in the global namespace
    import inspect
    import sys
    current_module = sys.modules[__name__]
    thunks = [obj for name, obj in inspect.getmembers(current_module) if isinstance(obj, Thunk)]
    