from abc import ABC, abstractmethod
from typing import Any, Dict, List

from .DatabaseConnection import DatabaseConnection

class ThunkStorageHandler(ABC):
    """Abstract handler for different Thunk type storage requirements."""
    
    @abstractmethod
    def get_table_name(self) -> str:
        """Get the table name for this thunk type."""
        pass
    
    @abstractmethod
    def get_create_table_sql(self, db_connection: DatabaseConnection) -> str:
        """Get database-specific table creation SQL."""
        pass
    
    @abstractmethod
    def get_create_indexes_sql(self, db_connection: DatabaseConnection) -> List[str]:
        """Get database-specific index creation SQL."""
        pass
    
    @abstractmethod
    def serialize_thunk_data(self, thunk_data: Dict[str, Any], db_connection: DatabaseConnection) -> Dict[str, Any]:
        """Transform thunk data for storage in this database."""
        pass
    
    @abstractmethod
    def deserialize_thunk_data(self, raw_data: Dict[str, Any], db_connection: DatabaseConnection) -> Dict[str, Any]:
        """Transform stored data back to thunk format."""
        pass
    
    @abstractmethod
    def get_insert_sql(self, db_connection: DatabaseConnection) -> str:
        """Get database-specific INSERT SQL."""
        pass
    
    @abstractmethod
    def get_select_by_id_sql(self, db_connection: DatabaseConnection) -> str:
        """Get database-specific SELECT by ID SQL."""
        pass