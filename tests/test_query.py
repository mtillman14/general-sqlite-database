"""Tests for DuckDB query interface."""

import numpy as np
import pandas as pd
import pytest

from scidb import BaseVariable, configure_database

# Skip all tests if DuckDB not installed
pytest.importorskip("duckdb")

from scidb.query import QueryInterface, query


class SensorReading(BaseVariable):
    """Test variable: time series sensor data."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": range(len(self.data)),
            "value": self.data,
        })

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df["value"].values


class ProcessedSignal(BaseVariable):
    """Test variable: processed signal data."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({
            "timestamp": range(len(self.data)),
            "filtered": self.data,
        })

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df["filtered"].values


@pytest.fixture
def db_with_data(tmp_path):
    """Create a database with test data."""
    db = configure_database(tmp_path / "test.db")

    # Save some sensor readings
    for exp in ["exp1", "exp2"]:
        for trial in [1, 2]:
            data = np.random.randn(100) * (10 if exp == "exp1" else 5)
            reading = SensorReading(data)
            reading.save(experiment=exp, trial=trial)

    # Save some processed signals
    for exp in ["exp1", "exp2"]:
        data = np.random.randn(100) * 2
        signal = ProcessedSignal(data)
        signal.save(experiment=exp, stage="final")

    return db


class TestQueryInterface:
    """Tests for QueryInterface class."""

    def test_tables(self, db_with_data):
        """Test listing available tables."""
        qi = QueryInterface(db_with_data)
        tables = qi.tables()

        assert "sensor_reading" in tables
        assert "processed_signal" in tables

    def test_schema(self, db_with_data):
        """Test getting table schema."""
        qi = QueryInterface(db_with_data)
        schema = qi.schema("sensor_reading")

        assert "timestamp" in schema["column"].values
        assert "value" in schema["column"].values

    def test_simple_query(self, db_with_data):
        """Test basic SELECT query."""
        qi = QueryInterface(db_with_data)
        df = qi.query("SELECT * FROM sensor_reading LIMIT 10")

        assert len(df) == 10
        assert "timestamp" in df.columns
        assert "value" in df.columns

    def test_query_with_filter(self, db_with_data):
        """Test query with WHERE clause."""
        qi = QueryInterface(db_with_data)
        df = qi.query("SELECT * FROM sensor_reading WHERE value > 5")

        assert len(df) > 0
        assert (df["value"] > 5).all()

    def test_aggregation_query(self, db_with_data):
        """Test aggregation query."""
        qi = QueryInterface(db_with_data)
        df = qi.query("""
            SELECT
                AVG(value) as avg_value,
                MIN(value) as min_value,
                MAX(value) as max_value,
                COUNT(*) as n
            FROM sensor_reading
        """)

        assert len(df) == 1
        assert "avg_value" in df.columns
        assert df["n"].iloc[0] == 400  # 4 records * 100 rows each

    def test_metadata_filter(self, db_with_data):
        """Test filtering by metadata before query."""
        qi = QueryInterface(db_with_data)

        # Only exp1 data
        df = qi.query(
            "SELECT COUNT(*) as n FROM sensor_reading",
            metadata_filter={"experiment": "exp1"}
        )
        assert df["n"].iloc[0] == 200  # 2 trials * 100 rows

        # Only exp2 data
        df = qi.query(
            "SELECT COUNT(*) as n FROM sensor_reading",
            metadata_filter={"experiment": "exp2"}
        )
        assert df["n"].iloc[0] == 200

    def test_include_vhash(self, db_with_data):
        """Test including vhash in results."""
        qi = QueryInterface(db_with_data)
        df = qi.query(
            "SELECT * FROM sensor_reading LIMIT 10",
            include_vhash=True
        )

        assert "_vhash" in df.columns
        assert df["_vhash"].notna().all()

    def test_include_metadata(self, db_with_data):
        """Test including metadata columns in results."""
        qi = QueryInterface(db_with_data)
        df = qi.query(
            "SELECT * FROM sensor_reading LIMIT 10",
            include_metadata=True
        )

        assert "_meta_experiment" in df.columns
        assert "_meta_trial" in df.columns

    def test_join_across_tables(self, db_with_data):
        """Test joining across variable types."""
        qi = QueryInterface(db_with_data)

        # Join sensor_reading and processed_signal on timestamp
        df = qi.query("""
            SELECT
                s.timestamp,
                s.value as sensor_value,
                p.filtered as processed_value
            FROM sensor_reading s
            JOIN processed_signal p ON s.timestamp = p.timestamp
            WHERE s.timestamp < 10
        """)

        assert len(df) > 0
        assert "sensor_value" in df.columns
        assert "processed_value" in df.columns

    def test_group_by_metadata(self, db_with_data):
        """Test grouping by metadata columns."""
        qi = QueryInterface(db_with_data)
        df = qi.query(
            """
            SELECT
                _meta_experiment,
                AVG(value) as avg_value,
                COUNT(*) as n
            FROM sensor_reading
            GROUP BY _meta_experiment
            ORDER BY _meta_experiment
            """,
            include_metadata=True
        )

        assert len(df) == 2  # exp1 and exp2
        assert df["_meta_experiment"].tolist() == ["exp1", "exp2"]

    def test_register_dataframe(self, db_with_data):
        """Test registering external DataFrame."""
        qi = QueryInterface(db_with_data)

        # Register an external lookup table
        lookup = pd.DataFrame({
            "experiment": ["exp1", "exp2"],
            "description": ["First experiment", "Second experiment"]
        })
        qi.register_dataframe("experiment_info", lookup)

        # Join with registered table
        df = qi.query(
            """
            SELECT
                s._meta_experiment,
                e.description,
                AVG(s.value) as avg_value
            FROM sensor_reading s
            JOIN experiment_info e ON s._meta_experiment = e.experiment
            GROUP BY s._meta_experiment, e.description
            """,
            include_metadata=True
        )

        assert "description" in df.columns
        assert len(df) == 2


class TestQueryConvenienceFunction:
    """Tests for the query() convenience function."""

    def test_one_off_query(self, db_with_data):
        """Test one-off query function."""
        df = query(db_with_data, "SELECT COUNT(*) as n FROM sensor_reading")
        assert df["n"].iloc[0] == 400

    def test_with_all_options(self, db_with_data):
        """Test query with all options."""
        df = query(
            db_with_data,
            "SELECT * FROM sensor_reading LIMIT 5",
            metadata_filter={"experiment": "exp1"},
            include_metadata=True,
            include_vhash=True,
        )

        assert len(df) == 5
        assert "_vhash" in df.columns
        assert "_meta_experiment" in df.columns
        assert (df["_meta_experiment"] == "exp1").all()


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_table(self, tmp_path):
        """Test querying empty/unregistered table."""
        db = configure_database(tmp_path / "empty.db")
        db.register(SensorReading)

        qi = QueryInterface(db)
        df = qi.query("SELECT * FROM sensor_reading")

        assert len(df) == 0

    def test_missing_duckdb(self, monkeypatch):
        """Test graceful handling when DuckDB not installed."""
        import scidb.query as query_module

        # Simulate DuckDB not being installed
        monkeypatch.setattr(query_module, "duckdb", None)

        with pytest.raises(ImportError, match="DuckDB is required"):
            query_module._require_duckdb()
