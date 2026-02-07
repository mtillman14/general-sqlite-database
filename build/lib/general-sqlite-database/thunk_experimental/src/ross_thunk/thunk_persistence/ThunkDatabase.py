from typing import Any, Dict, Optional, Union
from pathlib import Path

from .ThunkWriter import ThunkWriter
from .ThunkReader import ThunkReader

class ThunkDatabase:
    """Facade providing both read and write operations for different thunk types."""
    
    def __init__(self, connection_spec: Union[str, Path, Dict[str, Any]] = "thunks.db"):
        self.writer = ThunkWriter(connection_spec)
        self.reader = ThunkReader(connection_spec)
    
    def write_pipeline_thunk(self, thunk_data: Dict[str, Any]) -> None:
        """Write a PipelineThunk."""
        self.writer.write_thunk('pipeline', thunk_data)
    
    def write_thunk_output(self, thunk_data: Dict[str, Any]) -> None:
        """Write an ThunkOutput."""
        self.writer.write_thunk('output', thunk_data)
    
    def get_pipeline_thunk(self, thunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a PipelineThunk by ID."""
        return self.reader.get_thunk_by_id('pipeline', thunk_id)
    
    def get_thunk_output(self, thunk_id: str) -> Optional[Dict[str, Any]]:
        """Get an ThunkOutput by ID."""
        return self.reader.get_thunk_by_id('output', thunk_id)