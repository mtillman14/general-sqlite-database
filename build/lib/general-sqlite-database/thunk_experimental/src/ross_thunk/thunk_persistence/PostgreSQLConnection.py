from contextlib import contextmanager
from typing import Any
from typing import Union
from pathlib import Path

try:
    import psycopg2
except ImportError:
    psycopg2 = None
    print("psycopg2 is not installed. PostgreSQL support will not be available.")

from .DatabaseConnection import DatabaseConnection

class PostgreSQLConnection(DatabaseConnection):
    """PostgreSQL-specific database connection."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    @contextmanager
    def get_connection(self):
        """Context manager for PostgreSQL connections."""        
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
        finally:
            conn.close()
    
    def get_parameter_placeholder(self) -> str:
        return "%s"
    
    def serialize_json(self, data: Any) -> Any:
        """PostgreSQL can handle Python objects directly with JSONB."""
        return data  # psycopg2 handles the JSON serialization
    
    def deserialize_json(self, data: Any) -> Any:
        """PostgreSQL JSONB returns Python objects directly."""
        return data