"""DuckDB query interface for analytical queries on Parquet BLOB data."""

import io
from typing import TYPE_CHECKING

import pandas as pd

try:
    import duckdb
except ImportError:
    duckdb = None  # Optional dependency

if TYPE_CHECKING:
    from .database import DatabaseManager


def _require_duckdb():
    """Raise ImportError if DuckDB is not available."""
    if duckdb is None:
        raise ImportError(
            "DuckDB is required for query functionality. "
            "Install it with: pip install duckdb"
        )


class QueryInterface:
    """
    DuckDB-powered query interface for SciDB.

    Provides analytical SQL queries across variable data stored as Parquet BLOBs.
    Data remains in SQLite as the source of truth; DuckDB is used as a query engine.

    Example:
        db = configure_database("experiment.db")
        qi = QueryInterface(db)

        # Query a single variable type
        df = qi.query("SELECT * FROM sensor_reading WHERE value > 100")

        # Query with metadata filter
        df = qi.query(
            "SELECT * FROM sensor_reading WHERE value > 100",
            metadata_filter={"experiment": "exp1"}
        )

        # Join across variable types
        df = qi.query('''
            SELECT s.timestamp, s.value, p.result
            FROM sensor_reading s
            JOIN processed_data p ON s.timestamp = p.timestamp
        ''')

        # Get available tables
        print(qi.tables())
    """

    def __init__(self, db: "DatabaseManager"):
        """
        Initialize query interface.

        Args:
            db: The DatabaseManager instance to query
        """
        _require_duckdb()
        self.db = db
        self._duck = duckdb.connect(":memory:")

    def tables(self) -> list[str]:
        """
        List all queryable variable types.

        Returns:
            List of table names (variable types that have been registered)
        """
        cursor = self.db.connection.execute(
            "SELECT table_name FROM _registered_types ORDER BY table_name"
        )
        return [row["table_name"] for row in cursor.fetchall()]

    def schema(self, table_name: str) -> pd.DataFrame:
        """
        Get the schema (columns and types) for a variable type.

        Samples the first record to infer the DataFrame schema.

        Args:
            table_name: The variable type table name

        Returns:
            DataFrame with column names and types
        """
        # Get one blob to infer schema
        cursor = self.db.connection.execute(
            f"""
            SELECT d.data
            FROM {table_name} v
            JOIN _data d ON v.content_hash = d.content_hash
            LIMIT 1
            """
        )
        row = cursor.fetchone()

        if row is None:
            return pd.DataFrame(columns=["column", "dtype"])

        df = pd.read_parquet(io.BytesIO(row["data"]))

        return pd.DataFrame({
            "column": df.columns.tolist(),
            "dtype": [str(dt) for dt in df.dtypes.tolist()]
        })

    def query(
        self,
        sql: str,
        metadata_filter: dict | None = None,
        include_metadata: bool = False,
        include_vhash: bool = False,
    ) -> pd.DataFrame:
        """
        Execute an analytical SQL query across variable data.

        The query runs against DuckDB with variable types registered as tables.
        Each table contains the DataFrame columns from to_db() plus optional
        metadata columns.

        Args:
            sql: SQL query string. Table names are variable type table names
                 (e.g., "sensor_reading", "processed_data")
            metadata_filter: Optional dict to filter which records are included.
                           Applied before query execution.
            include_metadata: If True, include _meta_* columns for each metadata key
            include_vhash: If True, include _vhash column to identify source records

        Returns:
            Query results as a pandas DataFrame

        Example:
            # Simple query
            df = qi.query("SELECT * FROM sensor_reading WHERE value > 100")

            # With metadata filter (only include experiment='exp1' records)
            df = qi.query(
                "SELECT AVG(value) FROM sensor_reading",
                metadata_filter={"experiment": "exp1"}
            )

            # Include source tracking
            df = qi.query(
                "SELECT * FROM sensor_reading",
                include_vhash=True,
                include_metadata=True
            )
        """
        # Parse referenced tables from SQL (simple approach)
        referenced_tables = self._extract_table_names(sql)

        # Register each referenced table with DuckDB
        for table_name in referenced_tables:
            self._register_table(
                table_name,
                metadata_filter=metadata_filter,
                include_metadata=include_metadata,
                include_vhash=include_vhash,
            )

        # Execute query
        result = self._duck.execute(sql).fetchdf()

        return result

    def _extract_table_names(self, sql: str) -> set[str]:
        """
        Extract table names referenced in SQL query.

        Simple approach: find all registered table names that appear in the SQL.
        """
        available_tables = set(self.tables())
        sql_lower = sql.lower()

        referenced = set()
        for table in available_tables:
            # Check if table name appears in SQL (with word boundaries)
            # This is a simple heuristic; could be improved with proper parsing
            if table.lower() in sql_lower:
                referenced.add(table)

        return referenced

    def _register_table(
        self,
        table_name: str,
        metadata_filter: dict | None = None,
        include_metadata: bool = False,
        include_vhash: bool = False,
    ) -> None:
        """
        Register a variable type as a DuckDB table.

        Loads all matching Parquet BLOBs and combines them into a single
        table that DuckDB can query.
        """
        import json

        # Build SQLite query to get blobs
        conditions = []
        params = []

        if metadata_filter:
            for key, value in metadata_filter.items():
                conditions.append(f"json_extract(v.metadata, '$.{key}') = ?")
                if isinstance(value, (dict, list)):
                    params.append(json.dumps(value))
                else:
                    params.append(value)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        cursor = self.db.connection.execute(
            f"""
            SELECT v.vhash, v.metadata, d.data
            FROM {table_name} v
            JOIN _data d ON v.content_hash = d.content_hash
            WHERE {where_clause}
            """,
            params,
        )
        rows = cursor.fetchall()

        if not rows:
            # Register empty table with expected schema if possible
            self._duck.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT 1 WHERE FALSE")
            return

        # Combine all DataFrames
        all_dfs = []
        for row in rows:
            df = pd.read_parquet(io.BytesIO(row["data"]))

            if include_vhash:
                df["_vhash"] = row["vhash"]

            if include_metadata:
                meta = json.loads(row["metadata"])
                for key, value in meta.items():
                    df[f"_meta_{key}"] = value

            all_dfs.append(df)

        combined = pd.concat(all_dfs, ignore_index=True)

        # Register with DuckDB
        self._duck.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM combined")

    def query_raw(self, sql: str) -> pd.DataFrame:
        """
        Execute a raw DuckDB SQL query without automatic table registration.

        Use this after manually registering tables or for DuckDB-specific
        operations.

        Args:
            sql: SQL query string

        Returns:
            Query results as a pandas DataFrame
        """
        return self._duck.execute(sql).fetchdf()

    def register_dataframe(self, name: str, df: pd.DataFrame) -> None:
        """
        Register a pandas DataFrame as a queryable table.

        Useful for joining database data with in-memory data.

        Args:
            name: Table name to use in queries
            df: DataFrame to register
        """
        self._duck.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")

    def explain(self, sql: str) -> str:
        """
        Get the query execution plan.

        Args:
            sql: SQL query string

        Returns:
            Query plan as a string
        """
        result = self._duck.execute(f"EXPLAIN {sql}").fetchdf()
        return result.to_string()

    def close(self):
        """Close the DuckDB connection."""
        self._duck.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Convenience function for one-off queries
def query(
    db: "DatabaseManager",
    sql: str,
    metadata_filter: dict | None = None,
    include_metadata: bool = False,
    include_vhash: bool = False,
) -> pd.DataFrame:
    """
    Execute a one-off analytical query.

    Convenience wrapper that creates a QueryInterface, runs the query,
    and returns results.

    Args:
        db: The DatabaseManager instance
        sql: SQL query string
        metadata_filter: Optional metadata filter
        include_metadata: Include _meta_* columns
        include_vhash: Include _vhash column

    Returns:
        Query results as DataFrame

    Example:
        from scidb.query import query

        db = configure_database("experiment.db")
        df = query(db, "SELECT * FROM sensor_reading WHERE value > 100")
    """
    with QueryInterface(db) as qi:
        return qi.query(
            sql,
            metadata_filter=metadata_filter,
            include_metadata=include_metadata,
            include_vhash=include_vhash,
        )
