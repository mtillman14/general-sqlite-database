"""Tests for scidb.lineage module."""

import numpy as np
import pandas as pd
import pytest

from scidb.lineage import LineageRecord, extract_lineage, get_lineage_chain, get_raw_value
from scidb.thunk import OutputThunk, thunk


class TestLineageRecord:
    """Test LineageRecord dataclass."""

    def test_create_lineage_record(self):
        record = LineageRecord(
            function_name="process",
            function_hash="abc123",
            inputs=[{"name": "arg_0", "type": "Array", "vhash": "def456"}],
            constants=[{"name": "factor", "value_hash": "ghi789", "value_repr": "2"}],
        )

        assert record.function_name == "process"
        assert record.function_hash == "abc123"
        assert len(record.inputs) == 1
        assert len(record.constants) == 1

    def test_lineage_record_to_dict(self):
        record = LineageRecord(
            function_name="process",
            function_hash="abc123",
            inputs=[{"name": "arg_0", "vhash": "def456"}],
            constants=[],
        )

        d = record.to_dict()
        assert d["function_name"] == "process"
        assert d["function_hash"] == "abc123"
        assert d["inputs"] == [{"name": "arg_0", "vhash": "def456"}]
        assert d["constants"] == []

    def test_lineage_record_from_dict(self):
        d = {
            "function_name": "process",
            "function_hash": "abc123",
            "inputs": [{"name": "arg_0", "vhash": "def456"}],
            "constants": [{"name": "factor", "value_repr": "2"}],
        }

        record = LineageRecord.from_dict(d)
        assert record.function_name == "process"
        assert record.function_hash == "abc123"
        assert len(record.inputs) == 1
        assert len(record.constants) == 1

    def test_lineage_record_default_empty_lists(self):
        record = LineageRecord(function_name="test", function_hash="abc")
        assert record.inputs == []
        assert record.constants == []


class TestGetRawValue:
    """Test get_raw_value function."""

    def test_get_raw_value_from_output_thunk(self):
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        result = double(5)
        raw = get_raw_value(result)
        assert raw == 10

    def test_get_raw_value_passthrough(self):
        # Non-OutputThunk values should pass through unchanged
        assert get_raw_value(42) == 42
        assert get_raw_value("hello") == "hello"
        assert get_raw_value([1, 2, 3]) == [1, 2, 3]

    def test_get_raw_value_numpy_array(self):
        @thunk(n_outputs=1)
        def process(arr):
            return arr * 2

        arr = np.array([1, 2, 3])
        result = process(arr)
        raw = get_raw_value(result)

        np.testing.assert_array_equal(raw, np.array([2, 4, 6]))


class TestExtractLineage:
    """Test extract_lineage function."""

    def test_extract_lineage_basic(self):
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        result = double(5)
        lineage = extract_lineage(result)

        assert lineage.function_name == "double"
        assert isinstance(lineage.function_hash, str)
        assert len(lineage.function_hash) == 64

    def test_extract_lineage_with_constant_input(self):
        @thunk(n_outputs=1)
        def multiply(x, factor):
            return x * factor

        result = multiply(10, 2)
        lineage = extract_lineage(result)

        assert lineage.function_name == "multiply"
        # Both inputs are constants (not from other thunks or saved variables)
        assert len(lineage.constants) == 2
        assert len(lineage.inputs) == 0

    def test_extract_lineage_with_thunk_input(self):
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        intermediate = step1(5)
        result = step2(intermediate)
        lineage = extract_lineage(result)

        assert lineage.function_name == "step2"
        # The input is an OutputThunk
        assert len(lineage.inputs) == 1
        assert lineage.inputs[0]["source_type"] == "thunk"
        assert lineage.inputs[0]["source_function"] == "step1"

    def test_extract_lineage_constant_has_hash(self):
        @thunk(n_outputs=1)
        def process(data, factor):
            return data * factor

        result = process(10, 2)
        lineage = extract_lineage(result)

        # Find the constant entries
        for const in lineage.constants:
            assert "value_hash" in const
            assert "value_repr" in const
            assert "value_type" in const

    def test_extract_lineage_with_numpy(self):
        @thunk(n_outputs=1)
        def normalize(arr):
            return arr / arr.max()

        arr = np.array([1.0, 2.0, 4.0])
        result = normalize(arr)
        lineage = extract_lineage(result)

        assert lineage.function_name == "normalize"
        assert len(lineage.constants) == 1
        assert lineage.constants[0]["value_type"] == "ndarray"


class TestExtractLineageWithSavedVariables:
    """Test extract_lineage with saved BaseVariable instances."""

    def test_extract_lineage_with_variable_input(self, db, scalar_class):
        """Test that saved variables are tracked in lineage."""
        db.register(scalar_class)

        # Save a variable
        scalar_class.save(42, db=db, subject=1)
        var = scalar_class.load(db=db, subject=1)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        # Use the saved variable in a thunk
        result = process(var)
        lineage = extract_lineage(result)

        # The input should be tracked as a variable with vhash
        assert len(lineage.inputs) == 1
        assert lineage.inputs[0]["source_type"] == "variable"
        assert lineage.inputs[0]["type"] == "ScalarValue"
        assert lineage.inputs[0]["vhash"] == var.vhash


class TestGetLineageChain:
    """Test get_lineage_chain function."""

    def test_single_step_chain(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(5)
        chain = get_lineage_chain(result)

        assert len(chain) == 1
        assert chain[0].function_name == "process"

    def test_multi_step_chain(self):
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        @thunk(n_outputs=1)
        def step3(x):
            return x ** 2

        a = step1(5)
        b = step2(a)
        c = step3(b)

        chain = get_lineage_chain(c)

        # Should have 3 records in the chain
        assert len(chain) == 3

        # First record is the final step
        assert chain[0].function_name == "step3"

    def test_chain_max_depth(self):
        @thunk(n_outputs=1)
        def increment(x):
            return x + 1

        # Build a chain of 10 increments
        result = increment(0)
        for _ in range(9):
            result = increment(result)

        # With max_depth=3, should only get 3 records
        chain = get_lineage_chain(result, max_depth=3)
        assert len(chain) == 3


class TestLineageIntegration:
    """Integration tests for lineage with database."""

    def test_save_with_lineage_stores_provenance(self, db, scalar_class):
        """Saving a thunk result should store lineage."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        result = double(21)
        vhash = scalar_class.save(result, db=db, subject=1)

        # Check that lineage was stored
        assert db.has_lineage(vhash)

        provenance = db.get_provenance(scalar_class, subject=1)
        assert provenance is not None
        assert provenance["function_name"] == "double"

    def test_save_without_thunk_no_lineage(self, db, scalar_class):
        """Saving raw data should not have lineage."""
        db.register(scalar_class)

        vhash = scalar_class.save(42, db=db, subject=1)

        # Should not have lineage
        assert not db.has_lineage(vhash)

        provenance = db.get_provenance(scalar_class, subject=1)
        assert provenance is None

    def test_provenance_with_variable_inputs(self, db, scalar_class):
        """Test provenance tracking when thunk uses saved variables as inputs."""
        db.register(scalar_class)

        # Save input variable
        input_vhash = scalar_class.save(10, db=db, subject=1, role="input")
        input_var = scalar_class.load(db=db, subject=1, role="input")

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        # Process the input
        result = process(input_var)
        output_vhash = scalar_class.save(result, db=db, subject=1, role="output")

        # Check provenance
        provenance = db.get_provenance(scalar_class, subject=1, role="output")
        assert provenance is not None
        assert provenance["function_name"] == "process"

        # Input should reference the saved variable
        assert len(provenance["inputs"]) == 1
        assert provenance["inputs"][0]["vhash"] == input_vhash

    def test_get_derived_from(self, db, scalar_class):
        """Test querying what was derived from a variable."""
        db.register(scalar_class)

        # Save input
        scalar_class.save(10, db=db, subject=1, role="input")
        input_var = scalar_class.load(db=db, subject=1, role="input")

        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        @thunk(n_outputs=1)
        def triple(x):
            return x * 3

        # Create two outputs from the same input
        result1 = double(input_var)
        scalar_class.save(result1, db=db, subject=1, role="doubled")

        result2 = triple(input_var)
        scalar_class.save(result2, db=db, subject=1, role="tripled")

        # Query what was derived from the input
        derived = db.get_derived_from(scalar_class, subject=1, role="input")

        assert len(derived) == 2
        functions = {d["function"] for d in derived}
        assert "double" in functions
        assert "triple" in functions

    def test_lineage_chain_through_pipeline(self, db, scalar_class):
        """Test lineage through a multi-step pipeline."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 10

        @thunk(n_outputs=1)
        def step3(x):
            return x ** 2

        # Run pipeline
        a = step1(5)
        b = step2(a)
        c = step3(b)

        # Save final result
        scalar_class.save(c, db=db, subject=1)

        # Check provenance shows step3
        provenance = db.get_provenance(scalar_class, subject=1)
        assert provenance["function_name"] == "step3"

        # The input to step3 should be from step2
        assert len(provenance["inputs"]) == 1
        assert provenance["inputs"][0]["source_function"] == "step2"


class TestGetFullLineage:
    """Test get_full_lineage and format_lineage methods."""

    def test_get_full_lineage_single_step(self, db, scalar_class):
        """Test full lineage for a single processing step."""
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        # Save input
        scalar_class.save(10, db=db, subject=1, stage="raw")
        input_var = scalar_class.load(db=db, subject=1, stage="raw")

        # Process and save
        result = double(input_var)
        scalar_class.save(result, db=db, subject=1, stage="processed")

        # Get full lineage
        lineage = db.get_full_lineage(scalar_class, subject=1, stage="processed")

        assert lineage["type"] == "ScalarValue"
        assert lineage["function"] == "double"
        assert "function_hash" in lineage
        assert len(lineage["inputs"]) == 1
        assert lineage["inputs"][0]["type"] == "ScalarValue"
        assert lineage["inputs"][0]["function"] is None  # Source node
        assert lineage["inputs"][0]["source"] == "manual"

    def test_get_full_lineage_multi_step(self, db, scalar_class):
        """Test full lineage through a multi-step pipeline."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 10

        # Save input
        scalar_class.save(5, db=db, subject=1, stage="raw")
        input_var = scalar_class.load(db=db, subject=1, stage="raw")

        # Step 1
        result1 = step1(input_var)
        scalar_class.save(result1, db=db, subject=1, stage="step1")
        var1 = scalar_class.load(db=db, subject=1, stage="step1")

        # Step 2
        result2 = step2(var1)
        scalar_class.save(result2, db=db, subject=1, stage="step2")

        # Get full lineage from final output
        lineage = db.get_full_lineage(scalar_class, subject=1, stage="step2")

        # Check top level
        assert lineage["function"] == "step2"

        # Check first input (step1 output)
        step1_node = lineage["inputs"][0]
        assert step1_node["function"] == "step1"

        # Check step1's input (raw data)
        raw_node = step1_node["inputs"][0]
        assert raw_node["function"] is None
        assert raw_node["source"] == "manual"

    def test_get_full_lineage_with_constants(self, db, scalar_class):
        """Test that constants are captured in full lineage."""
        @thunk(n_outputs=1)
        def multiply(x, factor):
            return x * factor

        scalar_class.save(10, db=db, subject=1, stage="raw")
        input_var = scalar_class.load(db=db, subject=1, stage="raw")

        result = multiply(input_var, 5)
        scalar_class.save(result, db=db, subject=1, stage="scaled")

        lineage = db.get_full_lineage(scalar_class, subject=1, stage="scaled")

        assert lineage["function"] == "multiply"
        assert len(lineage["constants"]) == 1
        assert lineage["constants"][0]["name"] == "arg_1"

    def test_get_full_lineage_no_lineage(self, db, scalar_class):
        """Test full lineage for data without lineage (manually saved)."""
        scalar_class.save(42, db=db, subject=1)

        lineage = db.get_full_lineage(scalar_class, subject=1)

        assert lineage["type"] == "ScalarValue"
        assert lineage["function"] is None
        assert lineage["source"] == "manual"

    def test_get_full_lineage_max_depth(self, db, scalar_class):
        """Test max_depth parameter prevents infinite recursion."""
        @thunk(n_outputs=1)
        def increment(x):
            return x + 1

        # Build a chain of saves
        scalar_class.save(0, db=db, subject=1, step=0)
        var = scalar_class.load(db=db, subject=1, step=0)

        for i in range(5):
            result = increment(var)
            scalar_class.save(result, db=db, subject=1, step=i + 1)
            var = scalar_class.load(db=db, subject=1, step=i + 1)

        # With max_depth=2, should truncate
        lineage = db.get_full_lineage(scalar_class, subject=1, step=5, max_depth=2)

        # Should have truncated flag somewhere in the tree
        def has_truncated(node):
            if node.get("truncated"):
                return True
            for inp in node.get("inputs", []):
                if has_truncated(inp):
                    return True
            return False

        assert has_truncated(lineage)

    def test_format_lineage_basic(self, db, scalar_class):
        """Test format_lineage produces readable output."""
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        scalar_class.save(10, db=db, subject=1, stage="raw")
        input_var = scalar_class.load(db=db, subject=1, stage="raw")

        result = process(input_var)
        scalar_class.save(result, db=db, subject=1, stage="processed")

        formatted = db.format_lineage(scalar_class, subject=1, stage="processed")

        # Should be a non-empty string
        assert isinstance(formatted, str)
        assert len(formatted) > 0

        # Should contain key information
        assert "ScalarValue" in formatted
        assert "process" in formatted
        assert "vhash:" in formatted

    def test_format_lineage_shows_source(self, db, scalar_class):
        """Test format_lineage shows source for manual data."""
        scalar_class.save(42, db=db, subject=1)

        formatted = db.format_lineage(scalar_class, subject=1)

        assert "[source: manual]" in formatted

    def test_format_lineage_multi_step(self, db, scalar_class):
        """Test format_lineage with multi-step pipeline."""
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 10

        scalar_class.save(5, db=db, subject=1, stage="raw")
        input_var = scalar_class.load(db=db, subject=1, stage="raw")

        result1 = step1(input_var)
        scalar_class.save(result1, db=db, subject=1, stage="step1")
        var1 = scalar_class.load(db=db, subject=1, stage="step1")

        result2 = step2(var1)
        scalar_class.save(result2, db=db, subject=1, stage="step2")

        formatted = db.format_lineage(scalar_class, subject=1, stage="step2")

        # Should show the chain
        assert "step1" in formatted
        assert "step2" in formatted
        assert "[source: manual]" in formatted

    def test_format_lineage_with_constants(self, db, scalar_class):
        """Test format_lineage shows constants."""
        @thunk(n_outputs=1)
        def scale(x, factor):
            return x * factor

        scalar_class.save(10, db=db, subject=1, stage="raw")
        input_var = scalar_class.load(db=db, subject=1, stage="raw")

        result = scale(input_var, 3)
        scalar_class.save(result, db=db, subject=1, stage="scaled")

        formatted = db.format_lineage(scalar_class, subject=1, stage="scaled")

        assert "constants:" in formatted

    def test_get_full_lineage_by_vhash(self, db, scalar_class):
        """Test get_full_lineage with explicit vhash."""
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        scalar_class.save(10, db=db, subject=1, stage="raw")
        input_var = scalar_class.load(db=db, subject=1, stage="raw")

        result = double(input_var)
        vhash = scalar_class.save(result, db=db, subject=1, stage="processed")

        # Query by vhash
        lineage = db.get_full_lineage(scalar_class, version=vhash)

        assert lineage["vhash"] == vhash
        assert lineage["function"] == "double"
