from typing import Any, Dict

from .ThunkStorageHandler import ThunkStorageHandler

class ThunkStorageRegistry:
    """Registry to map thunk types to their storage handlers."""
    
    def __init__(self):
        self._handlers: Dict[str, ThunkStorageHandler] = {}
    
    def register_handler(self, thunk_type: str, handler: ThunkStorageHandler):
        """Register a storage handler for a thunk type."""
        self._handlers[thunk_type] = handler
    
    def get_handler(self, thunk_type: str) -> ThunkStorageHandler:
        """Get the storage handler for a thunk type."""
        if thunk_type not in self._handlers:
            raise ValueError(f"No storage handler registered for thunk type: {thunk_type}")
        return self._handlers[thunk_type]
    
    def get_all_handlers(self) -> Dict[str, ThunkStorageHandler]:
        """Get all registered handlers."""
        return self._handlers.copy()