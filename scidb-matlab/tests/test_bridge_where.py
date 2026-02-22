"""Tests for the where= filter parameter in load_and_extract().

These tests verify that load_and_extract() correctly passes the where= filter
through to DatabaseManager.load_all().  Runs entirely in Python without MATLAB.
"""

import sys
from pathlib import Path

# Add source paths for the monorepo packages
_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_root / "src"))
sys.path.insert(0, str(_root / "thunk-lib" / "src"))
sys.path.insert(0, str(_root / "canonical-hash" / "src"))
sys.path.insert(0, str(_root / "sciduck" / "src"))
sys.path.insert(0, str(_root / "pipelinedb-lib" / "src"))
sys.path.insert(0, str(_root / "path-gen" / "src"))
sys.path.insert(0, str(_root / "scirun-lib" / "src"))
sys.path.insert(0, str(_root / "scidb-matlab" / "src"))

import numpy as np
import pytest

from scidb_matlab.bridge import load_and_extract, register_matlab_variable
from scidb.database import configure_database
from scidb.filters import VariableFilter, RawFilter, CompoundFilter, raw_sql


class TestLoadAndExtractWhere:
    """Verify load_and_extract() respects the where= filter."""

    def test_no_filter_returns_all(self, tmp_path):
        """Without where=, all records are returned."""
        db = configure_database(
            tmp_path / "test.duckdb",
            ["subject"],
        )
        try:
            Side = register_matlab_variable("Side_NoFilter")
            StepLength = register_matlab_variable("StepLength_NoFilter")

            # Save two subjects
            db.save_variable(Side, "L", subject=1)
            db.save_variable(Side, "R", subject=2)
            db.save_variable(StepLength, 0.65, subject=1)
            db.save_variable(StepLength, 0.55, subject=2)

            result = load_and_extract(StepLength, {}, version_id="latest", db=db)
            assert int(result["n"]) == 2
        finally:
            db.close()

    def test_variable_filter_equality(self, tmp_path):
        """where= VariableFilter restricts results to matching schema_ids."""
        db = configure_database(
            tmp_path / "test.duckdb",
            ["subject"],
        )
        try:
            Side = register_matlab_variable("Side_VarFilter")
            StepLength = register_matlab_variable("StepLength_VarFilter")

            # Save two subjects
            db.save_variable(Side, "L", subject=1)
            db.save_variable(Side, "R", subject=2)
            db.save_variable(StepLength, 0.65, subject=1)
            db.save_variable(StepLength, 0.55, subject=2)

            # Filter: Side == "L" → only subject 1
            filt = VariableFilter(Side, "==", "L")
            result = load_and_extract(
                StepLength, {}, version_id="latest", db=db, where=filt
            )
            assert int(result["n"]) == 1

            # Verify it's the correct record (subject=1, StepLength=0.65)
            import json
            meta_arr = json.loads(str(result["json_meta"]))
            assert len(meta_arr) == 1
            assert int(meta_arr[0]["subject"]) == 1
        finally:
            db.close()

    def test_variable_filter_inequality(self, tmp_path):
        """where= with != operator returns complement."""
        db = configure_database(
            tmp_path / "test.duckdb",
            ["subject"],
        )
        try:
            Side = register_matlab_variable("Side_Neq")
            StepLength = register_matlab_variable("StepLength_Neq")

            db.save_variable(Side, "L", subject=1)
            db.save_variable(Side, "R", subject=2)
            db.save_variable(StepLength, 0.65, subject=1)
            db.save_variable(StepLength, 0.55, subject=2)

            # Filter: Side != "L" → only subject 2
            filt = VariableFilter(Side, "!=", "L")
            result = load_and_extract(
                StepLength, {}, version_id="latest", db=db, where=filt
            )
            assert int(result["n"]) == 1

            import json
            meta_arr = json.loads(str(result["json_meta"]))
            assert int(meta_arr[0]["subject"]) == 2
        finally:
            db.close()

    def test_raw_filter(self, tmp_path):
        """where= RawFilter with raw SQL applies correctly."""
        db = configure_database(
            tmp_path / "test.duckdb",
            ["subject"],
        )
        try:
            StepLength = register_matlab_variable("StepLength_Raw")

            db.save_variable(StepLength, 0.65, subject=1)
            db.save_variable(StepLength, 0.55, subject=2)
            db.save_variable(StepLength, 0.45, subject=3)

            # Raw SQL filter: value > 0.60
            filt = raw_sql('"value" > 0.60')
            result = load_and_extract(
                StepLength, {}, version_id="latest", db=db, where=filt
            )
            assert int(result["n"]) == 1  # only subject 1 (0.65 > 0.60)
        finally:
            db.close()

    def test_compound_and_filter(self, tmp_path):
        """where= CompoundFilter (AND) combines two filters correctly."""
        db = configure_database(
            tmp_path / "test.duckdb",
            ["subject"],
        )
        try:
            Side = register_matlab_variable("Side_And")
            Speed = register_matlab_variable("Speed_And")
            StepLength = register_matlab_variable("StepLength_And")

            db.save_variable(Side, "L", subject=1)
            db.save_variable(Side, "L", subject=2)
            db.save_variable(Side, "R", subject=3)
            db.save_variable(Speed, 1.5, subject=1)
            db.save_variable(Speed, 0.8, subject=2)
            db.save_variable(Speed, 1.2, subject=3)
            db.save_variable(StepLength, 0.65, subject=1)
            db.save_variable(StepLength, 0.55, subject=2)
            db.save_variable(StepLength, 0.60, subject=3)

            # Filter: Side == "L" AND Speed > 1.0 → only subject 1
            side_filt = VariableFilter(Side, "==", "L")
            speed_filt = VariableFilter(Speed, ">", 1.0)
            compound = side_filt & speed_filt

            result = load_and_extract(
                StepLength, {}, version_id="latest", db=db, where=compound
            )
            assert int(result["n"]) == 1

            import json
            meta_arr = json.loads(str(result["json_meta"]))
            assert int(meta_arr[0]["subject"]) == 1
        finally:
            db.close()

    def test_no_where_is_none(self, tmp_path):
        """Passing where=None behaves identically to not passing where at all."""
        db = configure_database(
            tmp_path / "test.duckdb",
            ["subject"],
        )
        try:
            StepLength = register_matlab_variable("StepLength_NoneWhere")

            db.save_variable(StepLength, 0.65, subject=1)
            db.save_variable(StepLength, 0.55, subject=2)

            result = load_and_extract(
                StepLength, {}, version_id="latest", db=db, where=None
            )
            assert int(result["n"]) == 2
        finally:
            db.close()

    def test_filter_with_metadata_restriction(self, tmp_path):
        """where= filter AND metadata restriction combine correctly."""
        db = configure_database(
            tmp_path / "test.duckdb",
            ["subject", "session"],
        )
        try:
            Side = register_matlab_variable("Side_MetaFilter")
            StepLength = register_matlab_variable("StepLength_MetaFilter")

            # subject=1, session=A: Side=L
            db.save_variable(Side, "L", subject=1, session="A")
            # subject=1, session=B: Side=R
            db.save_variable(Side, "R", subject=1, session="B")
            # subject=2, session=A: Side=L
            db.save_variable(Side, "L", subject=2, session="A")

            db.save_variable(StepLength, 0.65, subject=1, session="A")
            db.save_variable(StepLength, 0.55, subject=1, session="B")
            db.save_variable(StepLength, 0.60, subject=2, session="A")

            # Metadata filter: subject=1 + where=Side=="L" → only session A
            filt = VariableFilter(Side, "==", "L")
            result = load_and_extract(
                StepLength, {"subject": 1}, version_id="latest", db=db, where=filt
            )
            assert int(result["n"]) == 1

            import json
            meta_arr = json.loads(str(result["json_meta"]))
            assert str(meta_arr[0]["session"]) == "A"
        finally:
            db.close()
