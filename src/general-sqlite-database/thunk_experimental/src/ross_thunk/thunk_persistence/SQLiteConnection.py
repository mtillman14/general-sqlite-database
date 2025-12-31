from contextlib import contextmanager
from typing import Any, Union
from pathlib import Path
import sqlite3
import json

from .DatabaseConnection import DatabaseConnection

class SQLiteConnection(DatabaseConnection):
    """SQLite-specific database connection."""
    
    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path)
    
    @contextmanager
    def get_connection(self):
        """Context manager for SQLite connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()
    
    def get_parameter_placeholder(self) -> str:
        return "?"
    
    def serialize_json(self, data: Any) -> str:
        """SQLite stores JSON as TEXT."""
        try:
            return json.dumps(data, default=str)
        except (TypeError, ValueError):
            return json.dumps(str(data))
    
    def deserialize_json(self, data: str) -> Any:
        """Deserialize from SQLite TEXT field."""
        if not data:
            return None
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data