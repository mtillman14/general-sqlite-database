from abc import ABC, abstractmethod
from typing import Any

class DatabaseConnection(ABC):
    """Abstract base class for database connections."""
    
    @abstractmethod
    def get_connection(self):
        """Get a database connection (context manager)."""
        pass
    
    @abstractmethod
    def get_parameter_placeholder(self) -> str:
        """Get the parameter placeholder for this database (? vs %s)."""
        pass
    
    @abstractmethod
    def serialize_json(self, data: Any) -> Any:
        """Serialize data for JSON storage in this database."""
        pass
    
    @abstractmethod
    def deserialize_json(self, data: Any) -> Any:
        """Deserialize JSON data from this database."""
        pass