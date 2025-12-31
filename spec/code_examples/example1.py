"""A code example"""

from typing import Union, Any

import numpy as np
import pandas as pd

from . import RotationMatrix

a = np.ndarray([1 0 0; 0 1 0; 0 0 1])
a_var = RotationMatrix(a)
a_var.save(subject = 1, trial=1)

a_var_loaded = RotationMatrix.load(subject=1, trial=1)

class RotationMatrix(ParentClass):

    def __init__(self, data: Any):
        self.data = data

    def to_db() -> pd.DataFrame:
        """Define the conversion from the Python datatype to pandas DataFrame for storage in a SQL database table
        """
        data = self.data

        df = pd.DataFrame({x: [data[0,0], data[0,1], data[0,2]], y: [data[1,0], data[1,1], data[2,1]], z: [data[2,0], data[2,1], data[2,2]]})
        return df

    
    @staticmethod
    def from_db(df: pd.DataFrame) -> Union[list, np.ndarray]:
        """Define the conversion from pandas DataFrame (the format that the data is extracted from the database) to its native Python datatype

        Returns:
            Union[list, np.ndarray]: The native Python data type
        """

        a_var = [df.x[0], df.y[0], df.z[0]; df.x[1], df.y[1], df.z[1]; df.x[2], df.y[2], df.z[2]]
        return a_var