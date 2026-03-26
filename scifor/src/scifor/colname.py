"""ColName wrapper — resolves to the single non-schema data column name of a DataFrame."""

from typing import Any


class ColName:
    """
    Marker that resolves to the single non-schema data column name.

    Use this when a function needs to know the name of the data column
    in a DataFrame, but the function itself should stay framework-agnostic.

    Works with plain DataFrames (standalone mode). At for_each time,
    ColName(df) is replaced by the string name of the single data column
    (i.e., the one column that is not a schema key).

    Example:
        set_schema(["subject", "session"])
        result = for_each(
            analyze,
            inputs={"table": raw_df, "col_name": ColName(raw_df)},
            subject=[1, 2],
            session=["pre", "post"],
        )

        # The function is pure — no framework imports:
        def analyze(table, col_name):
            return table[col_name].mean()

    Raises ValueError if the DataFrame has 0 or 2+ non-schema data columns.
    """

    def __init__(self, data: Any):
        """
        Args:
            data: A pandas DataFrame whose single data column name will be resolved.
        """
        self.data = data
