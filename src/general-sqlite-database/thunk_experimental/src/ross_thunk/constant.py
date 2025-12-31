from hashlib import sha256
from typing import Any

from ross_thunk.thunk import Thunk

class Constant:
    """A simple class to represent a constant value in the thunking system."""
    def __init__(self, value: Any):
        self.value = value
        self.hash = sha256(repr(value).encode()).hexdigest()         
        self._is_missing = value is Thunk

    def __repr__(self):
        value = repr(self.value) if not self._is_missing else "MISSING"
        return f"Constant(value={value}, hash={self.hash})"