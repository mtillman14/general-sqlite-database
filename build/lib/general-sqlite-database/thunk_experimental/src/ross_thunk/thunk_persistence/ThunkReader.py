from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import threading

from .DatabaseConnection import DatabaseConnection
from .SQLiteConnection import SQLiteConnection
from .PostgreSQLConnection import PostgreSQLConnection
from .ThunkStorageHandler import ThunkStorageHandler
from .PipelineThunkStorageHandler import PipelineThunkStorageHandler
from .OutputThunkStorageHandler import OutputThunkStorageHandler
from .ThunkStorageRegistry import ThunkStorageRegistry

class ThunkReader:
    """Unified reader that handles different thunk types through storage handlers."""
    
    def __init__(self, connection_spec: Union[str, Path, Dict[str, Any]] = "thunks.db"):
        self.db_connection = self._create_connection(connection_spec)
        self.storage_registry = ThunkStorageRegistry()
        self._initialize_default_handlers()
    
    def _create_connection(self, connection_spec) -> DatabaseConnection:
        """Create appropriate database connection."""
        if isinstance(connection_spec, (str, Path)):
            connection_str = str(connection_spec)
            if connection_str.startswith(('postgresql://', 'postgres://')):
                return PostgreSQLConnection(connection_str)
            else:
                return SQLiteConnection(connection_spec)
        elif isinstance(connection_spec, dict):
            db_type = connection_spec.get('type', '').lower()
            if db_type == 'sqlite':
                return SQLiteConnection(connection_spec['path'])
            elif db_type in ('postgresql', 'postgres'):
                return PostgreSQLConnection(connection_spec['connection_string'])
        raise ValueError(f"Invalid connection specification: {connection_spec}")
    
    def _initialize_default_handlers(self):
        """Register default storage handlers."""
        self.storage_registry.register_handler('pipeline', PipelineThunkStorageHandler())
        self.storage_registry.register_handler('output', OutputThunkStorageHandler())
    
    def get_thunk_by_id(self, thunk_type: str, thunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a thunk by ID using the appropriate storage handler."""
        handler = self.storage_registry.get_handler(thunk_type)
        
        try:
            with self.db_connection.get_connection() as conn:
                select_sql = handler.get_select_by_id_sql(self.db_connection)
                cursor = conn.execute(select_sql, (thunk_id,))
                row = cursor.fetchone()
                
                if row:
                    raw_data = dict(row)
                    return handler.deserialize_thunk_data(raw_data, self.db_connection)
                return None
        except Exception as e:
            print(f"Database read error: {e}")
            return None