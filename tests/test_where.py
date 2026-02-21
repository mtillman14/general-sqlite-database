"""Integration tests for the where= filter parameter in load() and load_all().

Tests verify that:
- Basic equality filters work end-to-end
- Compound (AND/OR) filters work
- ColumnFilter (MyVar["col"] == val) works
- raw_sql() escape hatch works
- Error cases raise informative ValueError messages
"""

import numpy as np
import pandas as pd
import pytest

from scidb import BaseVariable, configure_database, raw_sql
from scidb.database import _local
from scidb.exceptions import NotFoundError
from scidb.filters import VariableFilter, ColumnFilter, CompoundFilter

from conftest import DEFAULT_TEST_SCHEMA_KEYS


# ===========================================================================
# Variable classes for integration tests
# ===========================================================================

class StepLength(BaseVariable):
    """Trial-level step length (scalar)."""
    schema_version = 1


class Side(BaseVariable):
    """Trial-level gait side ('L' or 'R') — scalar string."""
    schema_version = 1


class Speed(BaseVariable):
    """Trial-level walking speed (float scalar)."""
    schema_version = 1


class GaitData(BaseVariable):
    """Trial-level tabular gait data (DataFrame with Side and Speed columns)."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return self.data.copy()

    @classmethod
    def from_db(cls, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def populated_db(db):
    """Database with StepLength and Side data for subjects 1/2, trials 1/2.

    Schema: subject + trial (DEFAULT_TEST_SCHEMA_KEYS = ["subject", "trial"])

    Data layout:
      subject=1, trial=1 → Side="L", StepLength=0.65, Speed=1.2
      subject=1, trial=2 → Side="R", StepLength=0.72, Speed=1.5
      subject=2, trial=1 → Side="L", StepLength=0.68, Speed=1.1
      subject=2, trial=2 → Side="R", StepLength=0.75, Speed=1.6
    """
    Side.save("L", subject=1, trial=1)
    Side.save("R", subject=1, trial=2)
    Side.save("L", subject=2, trial=1)
    Side.save("R", subject=2, trial=2)

    Speed.save(1.2, subject=1, trial=1)
    Speed.save(1.5, subject=1, trial=2)
    Speed.save(1.1, subject=2, trial=1)
    Speed.save(1.6, subject=2, trial=2)

    StepLength.save(0.65, subject=1, trial=1)
    StepLength.save(0.72, subject=1, trial=2)
    StepLength.save(0.68, subject=2, trial=1)
    StepLength.save(0.75, subject=2, trial=2)

    return db


@pytest.fixture
def columnar_db(db):
    """Database with GaitData (DataFrame) and StepLength for filtering tests."""
    # GaitData has columns "Side" and "Speed"
    GaitData.save(pd.DataFrame({"Side": ["L"], "Speed": [1.2]}), subject=1, trial=1)
    GaitData.save(pd.DataFrame({"Side": ["R"], "Speed": [1.5]}), subject=1, trial=2)
    GaitData.save(pd.DataFrame({"Side": ["L"], "Speed": [1.1]}), subject=2, trial=1)
    GaitData.save(pd.DataFrame({"Side": ["R"], "Speed": [1.6]}), subject=2, trial=2)

    StepLength.save(0.65, subject=1, trial=1)
    StepLength.save(0.72, subject=1, trial=2)
    StepLength.save(0.68, subject=2, trial=1)
    StepLength.save(0.75, subject=2, trial=2)

    return db


# ===========================================================================
# Basic equality filter tests
# ===========================================================================

class TestBasicEqualityFilter:
    """End-to-end tests for simple equality filters."""

    def test_load_all_with_eq_filter(self, populated_db):
        """load_all with where=Side=='L' returns only left-side records."""
        results = list(StepLength.load_all(where=Side == "L"))
        assert len(results) == 2
        for var in results:
            assert var.data in (0.65, 0.68)

    def test_load_all_returns_right_side(self, populated_db):
        """load_all with where=Side=='R' returns only right-side records."""
        results = list(StepLength.load_all(where=Side == "R"))
        assert len(results) == 2
        for var in results:
            assert var.data in (0.72, 0.75)

    def test_load_with_eq_filter(self, populated_db):
        """load() with where= filter and enough metadata for single match."""
        result = StepLength.load(subject=1, where=Side == "L")
        # subject=1, Side="L" → trial=1 → StepLength=0.65
        assert result.data == pytest.approx(0.65)

    def test_load_all_as_df_with_filter(self, populated_db):
        """load_all(as_df=True) with where= filter returns filtered DataFrame."""
        df = StepLength.load_all(as_df=True, where=Side == "L")
        assert len(df) == 2
        # All returned data should be left-side step lengths
        assert sorted(df["data"].tolist()) == pytest.approx(sorted([0.65, 0.68]))

    def test_filter_no_matches_raises_not_found(self, populated_db):
        """Filter that matches nothing raises NotFoundError on load."""
        with pytest.raises(NotFoundError):
            StepLength.load(subject=1, where=Side == "X")

    def test_load_all_no_matches_returns_empty(self, populated_db):
        """load_all (generator) that matches nothing yields nothing."""
        results = list(StepLength.load_all(where=Side == "X"))
        assert results == []

    def test_gt_filter(self, populated_db):
        """where=Speed > 1.3 filters out slow trials."""
        results = list(StepLength.load_all(where=Speed > 1.3))
        assert len(results) == 2
        for var in results:
            assert var.data in (0.72, 0.75)

    def test_lt_filter(self, populated_db):
        """where=Speed < 1.3 returns only slow trials."""
        results = list(StepLength.load_all(where=Speed < 1.3))
        assert len(results) == 2
        for var in results:
            assert var.data in (0.65, 0.68)

    def test_ne_filter(self, populated_db):
        """where=Side != 'L' is equivalent to where=Side == 'R'."""
        results = list(StepLength.load_all(where=Side != "L"))
        assert len(results) == 2
        for var in results:
            assert var.data in (0.72, 0.75)


# ===========================================================================
# Compound filter tests
# ===========================================================================

class TestCompoundFilter:
    """Tests for AND/OR compound filters."""

    def test_and_filter(self, populated_db):
        """(Side == 'L') & (Speed > 1.15) matches only subject=1, trial=1."""
        results = list(StepLength.load_all(where=(Side == "L") & (Speed > 1.15)))
        assert len(results) == 1
        assert results[0].data == pytest.approx(0.65)

    def test_or_filter(self, populated_db):
        """(Side == 'L') | (Speed > 1.5) = left side OR high speed."""
        # Left side: subject=1 trial=1, subject=2 trial=1
        # Speed > 1.5: subject=2 trial=2
        # Union: 3 records
        results = list(StepLength.load_all(where=(Side == "L") | (Speed > 1.5)))
        assert len(results) == 3

    def test_not_filter(self, populated_db):
        """~(Side == 'L') is equivalent to Side != 'L'."""
        results = list(StepLength.load_all(where=~(Side == "L")))
        assert len(results) == 2
        for var in results:
            assert var.data in (0.72, 0.75)

    def test_triple_and_filter(self, populated_db):
        """Three conditions AND'd together."""
        f = (Side == "L") & (Speed > 1.0) & (Speed < 1.3)
        results = list(StepLength.load_all(where=f))
        # subject=1 trial=1: L, speed=1.2 (1.0<1.2<1.3 ✓)
        # subject=2 trial=1: L, speed=1.1 (1.0<1.1<1.3 ✓)
        assert len(results) == 2


# ===========================================================================
# ColumnFilter tests (MyVar["col"] == val)
# ===========================================================================

class TestColumnFilter:
    """Tests for column-level filtering on tabular variables."""

    def test_column_eq_filter(self, columnar_db):
        """GaitData['Side'] == 'L' filters StepLength to left-side only."""
        results = list(StepLength.load_all(where=GaitData["Side"] == "L"))
        assert len(results) == 2
        for var in results:
            assert var.data in (0.65, 0.68)

    def test_column_gt_filter(self, columnar_db):
        """GaitData['Speed'] > 1.3 filters to high-speed trials."""
        results = list(StepLength.load_all(where=GaitData["Speed"] > 1.3))
        assert len(results) == 2
        for var in results:
            assert var.data in (0.72, 0.75)

    def test_column_compound_filter(self, columnar_db):
        """Combine column filter with compound AND."""
        f = (GaitData["Side"] == "L") & (GaitData["Speed"] > 1.15)
        results = list(StepLength.load_all(where=f))
        assert len(results) == 1
        assert results[0].data == pytest.approx(0.65)

    def test_column_isin_filter(self, columnar_db):
        """GaitData['Side'].isin(['L']) matches left-side only."""
        results = list(StepLength.load_all(where=GaitData["Side"].isin(["L"])))
        assert len(results) == 2

    def test_column_isin_both_sides(self, columnar_db):
        """isin with both values returns all."""
        results = list(StepLength.load_all(where=GaitData["Side"].isin(["L", "R"])))
        assert len(results) == 4


# ===========================================================================
# raw_sql escape hatch tests
# ===========================================================================

class TestRawSqlFilter:
    """Tests for the raw_sql() escape hatch."""

    def test_raw_sql_basic(self, populated_db):
        """raw_sql applied to the target variable (StepLength) directly."""
        # Filter StepLength by its own value column
        results = list(StepLength.load_all(where=raw_sql('"value" > 0.70')))
        assert len(results) == 2
        for var in results:
            assert var.data > 0.70

    def test_raw_sql_invalid_raises(self, populated_db):
        """Invalid SQL in raw_sql raises a ValueError."""
        with pytest.raises(ValueError, match="Invalid where= SQL"):
            list(StepLength.load_all(where=raw_sql("INVALID SQL SYNTAX !!!@@@")))


# ===========================================================================
# Error case tests
# ===========================================================================

class TestFilterErrorCases:
    """Tests for informative error messages on misuse."""

    def test_filter_var_not_registered_raises(self, db):
        """Filter variable with no saved data raises ValueError."""
        StepLength.save(0.65, subject=1, trial=1)

        class UnknownVar(BaseVariable):
            schema_version = 1

        with pytest.raises(ValueError, match="not registered"):
            list(StepLength.load_all(where=UnknownVar == "X"))

    def test_filter_finer_than_target_raises(self, tmp_path):
        """Filter at finer schema level than target raises ValueError."""
        from scidb import configure_database
        db = configure_database(
            tmp_path / "test.duckdb",
            ["subject", "trial"],
        )

        # SubjectHeight is at "subject" level (coarser)
        class SubjectHeight(BaseVariable):
            schema_version = 1

        # TrialStepLength is at "trial" level (finer)
        class TrialStepLength(BaseVariable):
            schema_version = 1

        # Subject-level data for SubjectHeight
        SubjectHeight.save(170.0, subject=1)
        SubjectHeight.save(175.0, subject=2)

        # Trial-level data for TrialStepLength (finer)
        TrialStepLength.save(0.65, subject=1, trial=1)
        TrialStepLength.save(0.72, subject=1, trial=2)

        # Filter at trial level (finer), target at subject level (coarser) → error
        # TrialStepLength is at trial level, SubjectHeight is at subject level
        # Using TrialStepLength as a FILTER for SubjectHeight → finer than target
        with pytest.raises(ValueError, match="finer than target"):
            list(SubjectHeight.load_all(where=TrialStepLength == 0.65))

        db.close()

    def test_filter_missing_coverage_raises(self, db):
        """Filter variable missing data at some target schema locations raises ValueError."""
        # Save StepLength for 4 trial locations
        StepLength.save(0.65, subject=1, trial=1)
        StepLength.save(0.72, subject=1, trial=2)
        StepLength.save(0.68, subject=2, trial=1)
        StepLength.save(0.75, subject=2, trial=2)

        # Save Side for only 2 of 4 locations (incomplete coverage)
        Side.save("L", subject=1, trial=1)
        Side.save("R", subject=1, trial=2)
        # Missing: subject=2, trial=1 and subject=2, trial=2

        with pytest.raises(ValueError, match="missing data at"):
            list(StepLength.load_all(where=Side == "L"))


# ===========================================================================
# Coarse-to-fine level expansion tests
# ===========================================================================

class TestCoarseFilterExpansion:
    """Tests for coarse-level filter expanding to fine-level target."""

    def test_subject_level_filter_expands_to_trials(self, tmp_path):
        """Subject-level filter variable correctly filters trial-level target."""
        from scidb import configure_database

        db = configure_database(
            tmp_path / "test.duckdb",
            ["subject", "trial"],
        )

        class SubjectHeight(BaseVariable):
            schema_version = 1

        class TrialStepLength(BaseVariable):
            schema_version = 1

        # Subject-level heights: tall subjects are 1, short are 2
        SubjectHeight.save(180.0, subject=1)
        SubjectHeight.save(160.0, subject=2)

        # Trial-level data for both subjects
        TrialStepLength.save(0.65, subject=1, trial=1)
        TrialStepLength.save(0.70, subject=1, trial=2)
        TrialStepLength.save(0.55, subject=2, trial=1)
        TrialStepLength.save(0.58, subject=2, trial=2)

        # Filter: only tall subjects (height > 170) → subject=1
        # This should expand to include both trials for subject=1
        results = list(TrialStepLength.load_all(where=SubjectHeight > 170.0))
        assert len(results) == 2
        for var in results:
            assert var.data in (0.65, 0.70)

        db.close()
