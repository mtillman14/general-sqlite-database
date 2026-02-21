"""Column selection wrapper for BaseVariable types in for_each() inputs."""

from typing import Any


class ColumnSelection:
    """
    Wraps a BaseVariable class with column selection for use in for_each() inputs.

    Created automatically by BaseVariable.__class_getitem__ when using bracket
    syntax:

        MyVar["col_name"]           # single column -> numpy array
        MyVar[["col_a", "col_b"]]   # multiple columns -> DataFrame subset

    After loading, only the specified columns are extracted from the loaded
    DataFrame. Single column selection returns a numpy array; multiple columns
    return a DataFrame subset.
    """

    def __init__(self, var_type: type, columns: list[str]):
        """
        Args:
            var_type: The BaseVariable subclass to load
            columns: List of column names to extract after loading
        """
        self.var_type = var_type
        self.columns = columns

    @property
    def __name__(self) -> str:
        """Return a display name for format_inputs and error messages."""
        var_name = getattr(self.var_type, '__name__', type(self.var_type).__name__)
        if len(self.columns) == 1:
            return f'{var_name}["{self.columns[0]}"]'
        cols = ", ".join(f'"{c}"' for c in self.columns)
        return f'{var_name}[{cols}]'

    def load(self, **metadata) -> Any:
        """Load from the underlying var_type, then apply column selection."""
        return self.var_type.load(**metadata)

    # --- Comparison operators that produce ColumnFilter objects ---

    def __eq__(self, other):
        from scidb.filters import ColumnFilter
        return ColumnFilter(self.var_type, self.columns[0], "==", other)

    def __ne__(self, other):
        from scidb.filters import ColumnFilter
        return ColumnFilter(self.var_type, self.columns[0], "!=", other)

    def __lt__(self, other):
        from scidb.filters import ColumnFilter
        return ColumnFilter(self.var_type, self.columns[0], "<", other)

    def __le__(self, other):
        from scidb.filters import ColumnFilter
        return ColumnFilter(self.var_type, self.columns[0], "<=", other)

    def __gt__(self, other):
        from scidb.filters import ColumnFilter
        return ColumnFilter(self.var_type, self.columns[0], ">", other)

    def __ge__(self, other):
        from scidb.filters import ColumnFilter
        return ColumnFilter(self.var_type, self.columns[0], ">=", other)

    def isin(self, values) -> "InFilter":
        """Create an InFilter for set membership testing.

        Example:
            MyVar["Side"].isin(["L", "R"])
        """
        from scidb.filters import InFilter
        return InFilter(self.var_type, self.columns[0], list(values))

    def to_key(self) -> str:
        """Return a canonical string for use as a version key."""
        return f"{self.var_type.__name__}[{self.columns!r}]"

    def __hash__(self):
        return hash((self.var_type, tuple(self.columns)))
