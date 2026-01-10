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

class ThunkWriter:
    """Unified writer that handles different thunk types through storage handlers."""
    
    def __init__(self, connection_spec: Union[str, Path, Dict[str, Any]] = "thunks.db"):
        self.db_connection = self._create_connection(connection_spec)
        self.storage_registry = ThunkStorageRegistry()
        self._lock = threading.Lock()
        self._initialize_default_handlers()
        self._initialize_databases()
    
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
    
    def _initialize_databases(self):
        """Initialize all tables for all registered handlers."""
        with self._lock:
            for handler in self.storage_registry.get_all_handlers().values():
                with self.db_connection.get_connection() as conn:
                    # Create table
                    conn.execute(handler.get_create_table_sql(self.db_connection))
                    
                    # Create indexes
                    for index_sql in handler.get_create_indexes_sql(self.db_connection):
                        conn.execute(index_sql)
                    
                    conn.commit()
    
    def write_thunk(self, thunk_type: str, thunk_data: Dict[str, Any]) -> None:
        """Write a thunk using the appropriate storage handler."""
        handler = self.storage_registry.get_handler(thunk_type)
        serialized_data = handler.serialize_thunk_data(thunk_data, self.db_connection)
        
        with self._lock:
            try:
                with self.db_connection.get_connection() as conn:
                    insert_sql = handler.get_insert_sql(self.db_connection)
                    
                    # Build parameter list based on the specific handler's needs
                    if thunk_type == 'pipeline':
                        params = (
                            serialized_data['id'],
                            serialized_data['func_name'],
                            serialized_data['args'],
                            serialized_data['kwargs'],
                            serialized_data['result'],
                            serialized_data['execution_time'],
                            serialized_data['timestamp']
                        )
                    elif thunk_type == 'output':
                        if isinstance(self.db_connection, SQLiteConnection):
                            params = (
                                serialized_data['id'],
                                serialized_data['func_name'],
                                serialized_data['args'],
                                serialized_data['kwargs'],
                                serialized_data['result'],
                                serialized_data['value_type'],
                                serialized_data['value_scalar'],
                                serialized_data['value_binary'],
                                serialized_data['value_json'],
                                serialized_data['execution_time'],
                                serialized_data['timestamp']
                            )
                        else:  # PostgreSQL
                            params = (
                                serialized_data['id'],
                                serialized_data['func_name'],
                                serialized_data['args'],
                                serialized_data['kwargs'],
                                serialized_data['result'],
                                serialized_data['value'],
                                serialized_data['value_binary'],
                                serialized_data['execution_time'],
                                serialized_data['timestamp']
                            )
                    
                    conn.execute(insert_sql, params)
                    conn.commit()
            except Exception as e:
                print(f"Database write error: {e}")