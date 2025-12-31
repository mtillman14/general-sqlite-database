import sqlite3
import hashlib

from variable import Variable

class Database:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.cursor = self.connection.cursor()


    def load(self, name: str, data_object, hash: str = None):
        """
        Load the variable from the database.
        """        
    

    def save(self, variable: Variable) -> None:
        """
        Save the variable to the database.
        """
        self.create_table_if_not_exists(variable)


    def create_table_if_not_exists(self, variable: Variable) -> None:
        """
        Create a table for the variable if it does not already exist.
        """
        name = variable.name
        data = variable.data
        hashed_str = hash_data(data)
        if is_numeric(data):
            col_type = 'INTEGER'
        else:
            col_type = 'TEXT'
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data {col_type},
                hash TEXT
            )
        """)
        self.connection.commit()


def is_numeric(value: str) -> bool:
    """
    Check if the value is numeric.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False
    

def hash_data(data: str) -> str:
    """
    Hash the data using a simple hash function.
    """    
    return hashlib.sha256(data.encode()).hexdigest()