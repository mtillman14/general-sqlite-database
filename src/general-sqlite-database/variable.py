from typing import Any

class Variable:

    def __init__(self, data: Any, name: str, data_object: 'DataObject', hash: str = None):
        """
        Initialize the variable with a name, data object, and optional hash.
        """
        self.data = data
        self.name = name
        self.data_object = data_object
        self.hash = hash


    @classmethod
    def load(self, name: str, 
             data_object: DataObject,
             hash: str = None
             ) -> 'Variable':
        """
        Load the variable from the database.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    

    def save(self, name: str,
             data_object: DataObject
             ) -> None:
        """
        Save the variable to the database.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")