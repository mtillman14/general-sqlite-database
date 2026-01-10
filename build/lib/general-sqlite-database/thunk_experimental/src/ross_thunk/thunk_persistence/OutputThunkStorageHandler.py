from datetime import datetime
from typing import Any, Dict, List

from .ThunkStorageHandler import ThunkStorageHandler
from .DatabaseConnection import DatabaseConnection
from .SQLiteConnection import SQLiteConnection
from .PostgreSQLConnection import PostgreSQLConnection

class OutputThunkStorageHandler(ThunkStorageHandler):
    """Storage handler for OutputThunk - complex value storage requirements."""
    
    def get_table_name(self) -> str:
        return "output_thunks"
    
    def get_create_table_sql(self, db_connection: DatabaseConnection) -> str:
        """Create table SQL that varies significantly by database for OutputThunk."""
        if isinstance(db_connection, SQLiteConnection):
            # SQLite: Store complex values as separate columns
            return f"""
                CREATE TABLE IF NOT EXISTS {self.get_table_name()} (
                    id TEXT PRIMARY KEY,
                    func_name TEXT NOT NULL,
                    args TEXT,
                    kwargs TEXT,
                    result TEXT,
                    value_type TEXT,  -- 'scalar', 'array', 'object', etc.
                    value_scalar TEXT,  -- Simple values as text
                    value_binary BLOB,  -- Binary data like images, pickled objects
                    value_json TEXT,  -- Complex structures as JSON text
                    execution_time REAL,
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
        elif isinstance(db_connection, PostgreSQLConnection):
            # PostgreSQL: Use advanced column types
            return f"""
                CREATE TABLE IF NOT EXISTS {self.get_table_name()} (
                    id TEXT PRIMARY KEY,
                    func_name TEXT NOT NULL,
                    args JSONB,
                    kwargs JSONB,
                    result JSONB,
                    value JSONB,  -- PostgreSQL can handle complex values natively
                    value_binary BYTEA,  -- PostgreSQL binary type
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
        
        if isinstance(db_connection, SQLiteConnection):
            # SQLite-specific indexes
            indexes.append(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_value_type ON {table_name}(value_type)")
        elif isinstance(db_connection, PostgreSQLConnection):
            # PostgreSQL can index JSONB values
            indexes.extend([
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_value_gin ON {table_name} USING GIN(value)",
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_args_gin ON {table_name} USING GIN(args)"
            ])
        
        return indexes
    
    def serialize_thunk_data(self, thunk_data: Dict[str, Any], db_connection: DatabaseConnection) -> Dict[str, Any]:
        """Transform data for storage - OutputThunk has complex value handling."""
        value = thunk_data.get('value')
        
        if isinstance(db_connection, SQLiteConnection):
            # SQLite: Break down the value into type-specific columns
            serialized = {
                'id': thunk_data['id'],
                'func_name': thunk_data['func_name'],
                'args': db_connection.serialize_json(thunk_data['args']),
                'kwargs': db_connection.serialize_json(thunk_data['kwargs']),
                'result': db_connection.serialize_json(thunk_data['result']),
                'execution_time': thunk_data['execution_time'],
                'timestamp': thunk_data['timestamp'].isoformat() if thunk_data['timestamp'] else None,
                'value_type': None,
                'value_scalar': None,
                'value_binary': None,
                'value_json': None
            }
            
            # Determine value type and store appropriately
            if value is None:
                serialized['value_type'] = 'null'
            elif isinstance(value, (str, int, float, bool)):
                serialized['value_type'] = 'scalar'
                serialized['value_scalar'] = str(value)
            elif isinstance(value, bytes):
                serialized['value_type'] = 'binary'
                serialized['value_binary'] = value
            else:
                serialized['value_type'] = 'json'
                serialized['value_json'] = db_connection.serialize_json(value)
            
            return serialized
            
        elif isinstance(db_connection, PostgreSQLConnection):
            # PostgreSQL: Store value directly as JSONB
            return {
                'id': thunk_data['id'],
                'func_name': thunk_data['func_name'],
                'args': thunk_data['args'],  # PostgreSQL handles JSON natively
                'kwargs': thunk_data['kwargs'],
                'result': thunk_data['result'],
                'value': value if not isinstance(value, bytes) else None,
                'value_binary': value if isinstance(value, bytes) else None,
                'execution_time': thunk_data['execution_time'],
                'timestamp': thunk_data['timestamp']
            }
    
    def deserialize_thunk_data(self, raw_data: Dict[str, Any], db_connection: DatabaseConnection) -> Dict[str, Any]:
        """Transform stored data back to thunk format."""
        if isinstance(db_connection, SQLiteConnection):
            # SQLite: Reconstruct value from type-specific columns
            value_type = raw_data.get('value_type')
            if value_type == 'null':
                value = None
            elif value_type == 'scalar':
                value_str = raw_data['value_scalar']
                # Try to convert back to original type
                try:
                    if value_str in ('True', 'False'):
                        value = value_str == 'True'
                    elif '.' in value_str:
                        value = float(value_str)
                    else:
                        value = int(value_str)
                except ValueError:
                    value = value_str  # Keep as string
            elif value_type == 'binary':
                value = raw_data['value_binary']
            elif value_type == 'json':
                value = db_connection.deserialize_json(raw_data['value_json'])
            else:
                value = None
            
            return {
                'id': raw_data['id'],
                'func_name': raw_data['func_name'],
                'args': db_connection.deserialize_json(raw_data['args']),
                'kwargs': db_connection.deserialize_json(raw_data['kwargs']),
                'result': db_connection.deserialize_json(raw_data['result']),
                'value': value,
                'execution_time': raw_data['execution_time'],
                'timestamp': datetime.fromisoformat(raw_data['timestamp']) if raw_data['timestamp'] else None,
                'created_at': raw_data['created_at']
            }
            
        elif isinstance(db_connection, PostgreSQLConnection):
            # PostgreSQL: Value stored directly
            value = raw_data['value'] if raw_data['value'] is not None else raw_data['value_binary']
            return {
                'id': raw_data['id'],
                'func_name': raw_data['func_name'],
                'args': raw_data['args'],
                'kwargs': raw_data['kwargs'],
                'result': raw_data['result'],
                'value': value,
                'execution_time': raw_data['execution_time'],
                'timestamp': raw_data['timestamp'],
                'created_at': raw_data['created_at']
            }
    
    def get_insert_sql(self, db_connection: DatabaseConnection) -> str:
        """Get database-specific INSERT SQL for OutputThunk."""
        table_name = self.get_table_name()
        placeholder = db_connection.get_parameter_placeholder()
        
        if isinstance(db_connection, SQLiteConnection):
            return f"""
                INSERT OR REPLACE INTO {table_name} 
                (id, func_name, args, kwargs, result, value_type, value_scalar, value_binary, value_json, execution_time, timestamp)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
            """
        elif isinstance(db_connection, PostgreSQLConnection):
            return f"""
                INSERT INTO {table_name} 
                (id, func_name, args, kwargs, result, value, value_binary, execution_time, timestamp)
                VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})
                ON CONFLICT (id) DO UPDATE SET
                    func_name = EXCLUDED.func_name,
                    args = EXCLUDED.args,
                    kwargs = EXCLUDED.kwargs,
                    result = EXCLUDED.result,
                    value = EXCLUDED.value,
                    value_binary = EXCLUDED.value_binary,
                    execution_time = EXCLUDED.execution_time,
                    timestamp = EXCLUDED.timestamp
            """
    
    def get_select_by_id_sql(self, db_connection: DatabaseConnection) -> str:
        """Get database-specific SELECT by ID SQL."""
        table_name = self.get_table_name()
        placeholder = db_connection.get_parameter_placeholder()
        return f"SELECT * FROM {table_name} WHERE id = {placeholder}"