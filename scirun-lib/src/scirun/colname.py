"""ColName wrapper — resolves to the single data column name of a Variable type."""

from typing import Any


class ColName:
    """
    Marker that resolves to the single data column name of a DB-backed variable.

    Use this when a function needs to know the name of the data column
    but the function itself should stay framework-agnostic.

    At for_each time, ColName(MyVar) is replaced by the string name of
    the single data column for that variable type.

    Example:
        for_each(
            analyze,
            inputs={"table": MyVar, "col_name": ColName(MyVar)},
            outputs=[Result],
            subject=[1, 2, 3],
        )

        # The function is pure — no framework imports:
        def analyze(table, col_name):
            return table[col_name].mean()

    Raises ValueError if the variable has 0 or 2+ data columns.
    """

    def __init__(self, var_type: Any):
        """
        Args:
            var_type: The variable type (class) whose data column name will be resolved.
        """
        self.var_type = var_type
