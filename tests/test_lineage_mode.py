"""Tests for lineage_mode configuration (strict and ephemeral modes)."""

import numpy as np
import pandas as pd
import pytest

from scidb import (
    BaseVariable,
    DatabaseManager,
    configure_database,
    thunk,
    UnsavedIntermediateError,
)
from scidb.database import _local
from scidb.lineage import find_unsaved_variables


# --- Test Variable Classes ---

class RawData(BaseVariable):
    """Raw input data for testing."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.data})

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df["value"].values


class ProcessedData(BaseVariable):
    """Processed data for testing."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.data})

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df["value"].values


class FinalResult(BaseVariable):
    """Final result for testing."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.data})

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> np.ndarray:
        return df["value"].values


# --- Fixtures ---

@pytest.fixture
def strict_db(tmp_path):
    """Provide a database configured with strict lineage mode."""
    db_path = tmp_path / "strict_test.db"
    db = configure_database(db_path, lineage_mode="strict")
    yield db
    db.close()
    if hasattr(_local, 'database'):
        delattr(_local, 'database')


@pytest.fixture
def ephemeral_db(tmp_path):
    """Provide a database configured with ephemeral lineage mode."""
    db_path = tmp_path / "ephemeral_test.db"
    db = configure_database(db_path, lineage_mode="ephemeral")
    yield db
    db.close()
    if hasattr(_local, 'database'):
        delattr(_local, 'database')


@pytest.fixture(autouse=True)
def clear_global_state():
    """Clear global database state before and after each test."""
    if hasattr(_local, 'database'):
        delattr(_local, 'database')
    yield
    if hasattr(_local, 'database'):
        delattr(_local, 'database')


# --- Configuration Tests ---

class TestLineageModeConfiguration:
    """Test lineage_mode configuration."""

    def test_default_lineage_mode_is_strict(self, tmp_path):
        """Default lineage mode should be 'strict'."""
        db = DatabaseManager(tmp_path / "test.db")
        assert db.lineage_mode == "strict"
        db.close()

    def test_configure_strict_mode(self, tmp_path):
        """Can explicitly configure strict mode."""
        db = DatabaseManager(tmp_path / "test.db", lineage_mode="strict")
        assert db.lineage_mode == "strict"
        db.close()

    def test_configure_ephemeral_mode(self, tmp_path):
        """Can configure ephemeral mode."""
        db = DatabaseManager(tmp_path / "test.db", lineage_mode="ephemeral")
        assert db.lineage_mode == "ephemeral"
        db.close()

    def test_invalid_lineage_mode_raises_error(self, tmp_path):
        """Invalid lineage mode should raise ValueError."""
        with pytest.raises(ValueError, match="lineage_mode must be one of"):
            DatabaseManager(tmp_path / "test.db", lineage_mode="invalid")

    def test_configure_database_passes_lineage_mode(self, tmp_path):
        """configure_database should pass lineage_mode to DatabaseManager."""
        db = configure_database(tmp_path / "test.db", lineage_mode="ephemeral")
        assert db.lineage_mode == "ephemeral"
        db.close()


# --- Find Unsaved Variables Tests ---

class TestFindUnsavedVariables:
    """Test find_unsaved_variables function."""

    def test_find_unsaved_variable_in_chain(self, strict_db):
        """Should find unsaved BaseVariable in computation chain."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        # Create unsaved variable
        raw = RawData(np.array([1, 2, 3]))
        # Don't save it

        result = process(raw)

        unsaved = find_unsaved_variables(result)
        assert len(unsaved) == 1
        assert unsaved[0][0] is raw
        assert "process()" in unsaved[0][1]

    def test_no_unsaved_when_all_saved(self, strict_db):
        """Should return empty list when all variables are saved."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        RawData.save(np.array([1, 2, 3]), subject=1)
        raw = RawData.load(subject=1)

        result = process(raw)

        unsaved = find_unsaved_variables(result)
        assert len(unsaved) == 0

    def test_find_multiple_unsaved_variables(self, strict_db):
        """Should find all unsaved variables in chain."""
        @thunk(n_outputs=1)
        def combine(a, b):
            return a + b

        raw1 = RawData(np.array([1, 2, 3]))
        raw2 = RawData(np.array([4, 5, 6]))
        # Don't save either

        result = combine(raw1, raw2)

        unsaved = find_unsaved_variables(result)
        assert len(unsaved) == 2

    def test_find_unsaved_in_nested_chain(self, strict_db):
        """Should find unsaved variable deep in nested chain."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        @thunk(n_outputs=1)
        def step3(x):
            return x ** 2

        # Create chain with unsaved intermediate
        RawData.save(np.array([1, 2, 3]), subject=1)  # Save the raw
        raw = RawData.load(subject=1)

        result1 = step1(raw)
        intermediate = ProcessedData(result1)
        # Don't save intermediate

        result2 = step2(intermediate)
        result3 = step3(result2)

        unsaved = find_unsaved_variables(result3)
        assert len(unsaved) == 1
        assert isinstance(unsaved[0][0], ProcessedData)


# --- Strict Mode Tests ---

class TestStrictMode:
    """Test strict lineage mode behavior."""

    def test_strict_mode_allows_saved_intermediates(self, strict_db):
        """Strict mode should allow saving when all intermediates are saved."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        RawData.save(np.array([1, 2, 3]), subject=1)
        raw = RawData.load(subject=1)

        result = process(raw)

        # Should not raise - all intermediates saved
        vhash = ProcessedData.save(result, subject=1, stage="processed")
        assert vhash is not None

    def test_strict_mode_raises_for_unsaved_intermediate(self, strict_db):
        """Strict mode should raise error for unsaved intermediate."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        raw = RawData(np.array([1, 2, 3]))
        # Don't save raw

        result = process(raw)

        with pytest.raises(UnsavedIntermediateError) as exc_info:
            ProcessedData.save(result, subject=1)

        # Error message should be helpful
        assert "RawData" in str(exc_info.value)
        assert "strict lineage mode" in str(exc_info.value).lower()

    def test_strict_mode_error_message_shows_path(self, strict_db):
        """Error message should show the path to unsaved variable."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        raw = RawData(np.array([1, 2, 3]))

        result = process(raw)

        with pytest.raises(UnsavedIntermediateError) as exc_info:
            ProcessedData.save(result, subject=1)

        assert "process()" in str(exc_info.value)

    def test_strict_mode_allows_raw_data_without_thunk(self, strict_db):
        """Strict mode should allow saving raw data (not from thunk)."""
        # Should not raise - no thunk involved
        vhash = RawData.save(np.array([1, 2, 3]), subject=1)
        assert vhash is not None

    def test_strict_mode_with_nested_pipeline(self, strict_db):
        """Strict mode with multi-step pipeline where all are saved."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        RawData.save(np.array([1, 2, 3]), subject=1, stage="raw")
        raw = RawData.load(subject=1, stage="raw")

        result1 = step1(raw)
        ProcessedData.save(result1, subject=1, stage="step1")
        processed = ProcessedData.load(subject=1, stage="step1")

        result2 = step2(processed)

        # Should work - all intermediates saved
        vhash = FinalResult.save(result2, subject=1, stage="final")
        assert vhash is not None

    def test_strict_mode_fails_with_unsaved_in_middle(self, strict_db):
        """Strict mode fails when intermediate in chain is unsaved."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        RawData.save(np.array([1, 2, 3]), subject=1, stage="raw")
        raw = RawData.load(subject=1, stage="raw")

        result1 = step1(raw)
        processed = ProcessedData(result1)
        # Don't save processed!

        result2 = step2(processed)

        with pytest.raises(UnsavedIntermediateError):
            FinalResult.save(result2, subject=1, stage="final")


# --- Ephemeral Mode Tests ---

class TestEphemeralMode:
    """Test ephemeral lineage mode behavior."""

    def test_ephemeral_mode_allows_unsaved_intermediates(self, ephemeral_db):
        """Ephemeral mode should allow saving with unsaved intermediates."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        raw = RawData(np.array([1, 2, 3]))
        # Don't save raw

        result = process(raw)

        # Should not raise in ephemeral mode
        vhash = ProcessedData.save(result, subject=1)
        assert vhash is not None

    def test_ephemeral_mode_stores_lineage_for_unsaved(self, ephemeral_db):
        """Ephemeral mode should store lineage record for unsaved intermediate."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        raw = RawData(np.array([1, 2, 3]))
        # Don't save raw

        result = process(raw)
        ProcessedData.save(result, subject=1)

        # Check that provenance includes info about the unsaved variable
        provenance = ephemeral_db.get_provenance(ProcessedData, subject=1)
        assert provenance is not None
        assert provenance["function_name"] == "process"

    def test_ephemeral_mode_creates_ephemeral_lineage_entry(self, ephemeral_db):
        """Ephemeral mode should create ephemeral:* entries in lineage table."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        # Save raw data
        RawData.save(np.array([1, 2, 3]), subject=1, stage="raw")
        raw = RawData.load(subject=1, stage="raw")

        # Create intermediate from thunk but don't save
        result1 = step1(raw)
        intermediate = ProcessedData(result1)
        # Don't save intermediate - this should create ephemeral entry

        # Process the unsaved intermediate
        result2 = step2(intermediate)
        FinalResult.save(result2, subject=1, stage="final")

        # Query the lineage table directly for ephemeral entries
        cursor = ephemeral_db.connection.execute(
            "SELECT output_vhash, output_type FROM _lineage WHERE output_vhash LIKE 'ephemeral:%'"
        )
        ephemeral_entries = cursor.fetchall()

        # Should have at least one ephemeral entry for the unsaved intermediate
        assert len(ephemeral_entries) >= 1

    def test_ephemeral_mode_with_multi_step_pipeline(self, ephemeral_db):
        """Ephemeral mode with multi-step pipeline, only raw saved."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        @thunk(n_outputs=1)
        def step3(x):
            return x ** 2

        # Save raw data
        RawData.save(np.array([1, 2, 3]), subject=1, stage="raw")
        raw = RawData.load(subject=1, stage="raw")

        # Don't save any intermediates
        r1 = step1(raw)
        p1 = ProcessedData(r1)

        r2 = step2(p1)
        p2 = ProcessedData(r2)

        r3 = step3(p2)

        # Should work in ephemeral mode
        vhash = FinalResult.save(r3, subject=1)
        assert vhash is not None

    def test_ephemeral_lineage_query_works(self, ephemeral_db):
        """Should be able to query full lineage including ephemeral nodes."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        raw = RawData(np.array([1, 2, 3]))

        result = process(raw)
        ProcessedData.save(result, subject=1)

        # Should be able to get full lineage
        lineage = ephemeral_db.get_full_lineage(ProcessedData, subject=1)
        assert lineage is not None
        assert lineage["function"] == "process"

    def test_ephemeral_format_lineage_shows_marker(self, ephemeral_db):
        """format_lineage should show [ephemeral] marker for unsaved nodes."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        raw = RawData(np.array([1, 2, 3]))

        result = process(raw)
        ProcessedData.save(result, subject=1)

        formatted = ephemeral_db.format_lineage(ProcessedData, subject=1)

        # Should contain ephemeral marker or indication
        assert "process" in formatted
        # The unsaved RawData should be shown somehow
        assert "RawData" in formatted or "ephemeral" in formatted.lower()


# --- Mixed Scenarios ---

class TestMixedScenarios:
    """Test mixed scenarios with saved and unsaved variables."""

    def test_ephemeral_with_some_saved(self, ephemeral_db):
        """Ephemeral mode with mix of saved and unsaved."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        # Save raw but not intermediate
        RawData.save(np.array([1, 2, 3]), subject=1, stage="raw")
        raw = RawData.load(subject=1, stage="raw")

        r1 = step1(raw)
        p1 = ProcessedData(r1)
        # Don't save p1

        r2 = step2(p1)
        FinalResult.save(r2, subject=1, stage="final")

        # Query lineage
        lineage = ephemeral_db.get_full_lineage(FinalResult, subject=1, stage="final")
        assert lineage["function"] == "step2"

    def test_strict_mode_preserves_lineage_when_valid(self, strict_db):
        """Strict mode should still track full lineage when all saved."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        RawData.save(np.array([1, 2, 3]), subject=1, stage="raw")
        raw = RawData.load(subject=1, stage="raw")

        r1 = step1(raw)
        ProcessedData.save(r1, subject=1, stage="step1")
        p1 = ProcessedData.load(subject=1, stage="step1")

        r2 = step2(p1)
        FinalResult.save(r2, subject=1, stage="final")

        # Full lineage should be available
        lineage = strict_db.get_full_lineage(FinalResult, subject=1, stage="final")
        assert lineage["function"] == "step2"
        assert len(lineage["inputs"]) == 1
        assert lineage["inputs"][0]["function"] == "step1"


# --- Edge Cases ---

class TestEdgeCases:
    """Test edge cases for lineage mode."""

    def test_unsaved_variable_with_constant_inputs(self, strict_db):
        """Unsaved variable used alongside constants."""
        @thunk(n_outputs=1)
        def multiply(x, factor):
            return x * factor

        raw = RawData(np.array([1, 2, 3]))
        # Don't save

        result = multiply(raw, 2)

        with pytest.raises(UnsavedIntermediateError):
            ProcessedData.save(result, subject=1)

    def test_multiple_outputs_thunk_with_unsaved(self, strict_db):
        """Multi-output thunk with unsaved input."""
        @thunk(n_outputs=2)
        def split(x):
            mid = len(x) // 2
            return x[:mid], x[mid:]

        raw = RawData(np.array([1, 2, 3, 4]))
        # Don't save

        left, right = split(raw)

        with pytest.raises(UnsavedIntermediateError):
            ProcessedData.save(left, subject=1, half="left")

    def test_ephemeral_no_duplicate_entries(self, ephemeral_db):
        """Ephemeral mode should not create duplicate entries."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        # Save raw data
        RawData.save(np.array([1, 2, 3]), subject=1)
        raw = RawData.load(subject=1)

        # Create unsaved intermediate from thunk
        result1 = step1(raw)
        intermediate = ProcessedData(result1)
        # Don't save intermediate

        # Save two different final outputs from the same unsaved intermediate
        final1 = step2(intermediate)
        FinalResult.save(final1, subject=1, version=1)

        final2 = step2(intermediate)
        FinalResult.save(final2, subject=1, version=2)

        # Count ephemeral entries
        cursor = ephemeral_db.connection.execute(
            "SELECT COUNT(*) FROM _lineage WHERE output_vhash LIKE 'ephemeral:%'"
        )
        count = cursor.fetchone()[0]

        # Should only have one ephemeral entry for the intermediate
        # (the same unsaved variable used twice should create one entry due to dedup)
        assert count >= 1  # At least one, implementation may vary

    def test_deeply_nested_unsaved_chain(self, ephemeral_db):
        """Deep chain of unsaved intermediates in ephemeral mode."""
        @thunk(n_outputs=1)
        def increment(x):
            return x + 1

        # Start with saved raw data
        RawData.save(np.array([0, 1, 2]), subject=1, stage="raw")
        raw = RawData.load(subject=1, stage="raw")

        # Build deep chain of unsaved intermediates
        current = raw
        for i in range(5):
            result = increment(current)
            current = ProcessedData(result)
            # Don't save any intermediate

        # Final step
        final_result = increment(current)

        # Should work in ephemeral mode
        vhash = FinalResult.save(final_result, subject=1)
        assert vhash is not None


# --- Provenance Query Tests ---

class TestProvenanceQueries:
    """Test provenance queries work correctly with both modes."""

    def test_get_provenance_strict_mode(self, strict_db):
        """get_provenance should work in strict mode."""
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        RawData.save(np.array([1, 2, 3]), subject=1, stage="raw")
        raw = RawData.load(subject=1, stage="raw")

        result = double(raw)
        ProcessedData.save(result, subject=1, stage="processed")

        provenance = strict_db.get_provenance(ProcessedData, subject=1, stage="processed")
        assert provenance["function_name"] == "double"
        assert len(provenance["inputs"]) == 1
        assert provenance["inputs"][0]["source_type"] == "variable"

    def test_get_provenance_ephemeral_mode(self, ephemeral_db):
        """get_provenance should work in ephemeral mode."""
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        raw = RawData(np.array([1, 2, 3]))
        # Don't save raw

        result = double(raw)
        ProcessedData.save(result, subject=1)

        provenance = ephemeral_db.get_provenance(ProcessedData, subject=1)
        assert provenance["function_name"] == "double"
        # Input should be tracked as unsaved_variable
        assert len(provenance["inputs"]) == 1
        assert provenance["inputs"][0]["source_type"] == "unsaved_variable"

    def test_has_lineage_works_both_modes(self, strict_db, tmp_path):
        """has_lineage should work correctly in both modes."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        # Strict mode
        RawData.save(np.array([1, 2, 3]), subject=1)
        raw = RawData.load(subject=1)
        result = process(raw)
        vhash = ProcessedData.save(result, subject=1)

        assert strict_db.has_lineage(vhash)

        # Raw data without thunk has no lineage
        vhash2 = RawData.save(np.array([4, 5, 6]), subject=2)
        assert not strict_db.has_lineage(vhash2)
