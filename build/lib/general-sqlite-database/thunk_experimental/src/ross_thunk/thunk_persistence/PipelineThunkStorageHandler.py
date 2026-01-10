from datetime import datetime
from typing import Any, Dict, List

from .DatabaseConnection import DatabaseConnection
from .SQLiteConnection import SQLiteConnection
from .PostgreSQLConnection import PostgreSQLConnection
from .ThunkStorageHandler import ThunkStorageHandler

class PipelineThunkStorageHandler(ThunkStorageHandler):
    """Storage handler for PipelineThunk - simpler storage requirements."""
    
    def get_table_name(self) -> str:
        return "pipeline_thunks"
    
    def get_create_table_sql(self, db_connection: DatabaseConnection) -> str:
        """Create table SQL that varies by database for PipelineThunk."""
        if isinstance(db_connection, SQLiteConnection):
            return f"""
                CREATE TABLE IF NOT EXISTS {self.get_table_name()} (
                    id TEXT PRIMARY KEY,
                    func_name TEXT NOT NULL,
                    args TEXT,  -- JSON as TEXT in SQLite
                    kwargs TEXT,
                    result TEXT,
                    execution_time REAL,
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
        elif isinstance(db_connection, PostgreSQLConnection):
            return f"""
                CREATE TABLE IF NOT EXISTS {self.get_table_name()} (
                    id TEXT PRIMARY KEY,
                    func_name TEXT NOT NULL,
                    args JSONB,  -- Native JSON in PostgreSQL
                    kwargs JSONB,
                    result JSONB,
                    execution_time REAL,
                    timestamp TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        else:
            raise ValueError(f"Unsupported database type: {type(db_connection)}")
    
    def get_create_indexes_sql(self, db_connection: DatabaseConnection) -> List[str]:
        table_name = self.get_table_name()
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_func_name ON {table_name}(func_name)",
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_timestamp ON {table_name}(timestamp)",
        ]
        
        # PostgreSQL can have GIN indexes on JSONB
        if isinstance(db_connection, PostgreSQLConnection):
            indexes.extend([
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_args_gin ON {table_name} USING GIN(args)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_kwargs_gin ON {table_name} USING GIN(kwargs)"
            ])
        
        return indexes
    
    def serialize_thunk_data(self, thunk_data: Dict[str, Any], db_connection: DatabaseConnection) -> Dict[str, Any]:
        """Transform data for storage - PipelineThunk has standard serialization."""
        return {
            'id': thunk_data['id'],
            'func_name': thunk_data['func_name'],
            'args': db_connection.serialize_json(thunk_data['args']),
            'kwargs': db_connection.serialize_json(thunk_data['kwargs']),
            'result': db_connection.serialize_json(thunk_data['result']),
            'execution_time': thunk_data['execution_time'],
            'timestamp': thunk_data['timestamp'].isoformat() if thunk_data['timestamp'] else None
        }
    
    def deserialize_thunk_data(self, raw_data: Dict[str, Any], db_connection: DatabaseConnection) -> Dict[str, Any]:
        """Transform stored data back to thunk format."""
        return {
            'id': raw_data['id'],
            'func_name': raw_data['func_name'],
            'args': db_connection.deserialize_json(raw_data['args']),
            'kwargs': db_connection.deserialize_json(raw_data['kwargs']),
            'result': db_connection.deserialize_json(raw_data['result']),
            'execution_time': raw_data['execution_time'],
            'timestamp': datetime.fromisoformat(raw_data['timestamp']) if raw_data['timestamp'] else None,
            'created_at': raw_data['created_at']
        }
    
    def get_insert_sql(self, db_connection: DatabaseConnection) -> str:
        """Get database-specific INSERT SQL."""
        table_name = self.get_table_name()
        placeholder = db_connection.get_parameter_placeholder()
        
        if isinstance(db_connection, SQLiteConnection):
            return f"""
                INSERT OR REPLACE INTO {table_name} 
                (id, func_name, args, kwargs, result, execution_time, timestamp)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """
        elif isinstance(db_connection, PostgreSQLConnection):
            return f"""
                INSERT INTO {table_name} 
                (id, func_name, args, kwargs, result, execution_time, timestamp)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                ON CONFLICT (id) DO UPDATE SET
                    func_name = EXCLUDED.func_name,
                    args = EXCLUDED.args,
                    kwargs = EXCLUDED.kwargs,
                    result = EXCLUDED.result,
                    execution_time = EXCLUDED.execution_time,
                    timestamp = EXCLUDED.timestamp
            """
    
    def get_select_by_id_sql(self, db_connection: DatabaseConnection) -> str:
        """Get database-specific SELECT by ID SQL."""
        table_name = self.get_table_name()
        placeholder = db_connection.get_parameter_placeholder()
        return f"SELECT * FROM {table_name} WHERE id = {placeholder}"