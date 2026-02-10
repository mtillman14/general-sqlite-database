
import pandas as pd

def load_gaitrite_file(path: str) -> pd.DataFrame:
    """Load raw GaitRite .xlsx file."""
    return pd.read_excel(path)