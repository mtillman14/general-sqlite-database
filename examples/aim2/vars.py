

import pandas as pd
import numpy as np

from scidb import BaseVariable


class ScalarList(BaseVariable):
    """Base class for scalar measurements."""

    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": [self.data]})

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return np.ndarray([v for v in df["value"]])


class StepLength(ScalarList):
    """Step length measurement in meters."""

    pass


class StepWidth(ScalarList):
    """Step width measurement in meters."""

    pass


class Side(ScalarList):
    """Left or right side"""

    pass