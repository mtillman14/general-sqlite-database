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
        var = scalar_class(42)
        var.save(db=db, subject=1)

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
        var = scalar_class(result)
        vhash = var.save(db=db, subject=1)

        # Check that lineage was stored
        assert db.has_lineage(vhash)

        provenance = db.get_provenance(scalar_class, subject=1)
        assert provenance is not None
        assert provenance["function_name"] == "double"

    def test_save_without_thunk_no_lineage(self, db, scalar_class):
        """Saving raw data should not have lineage."""
        db.register(scalar_class)

        var = scalar_class(42)
        vhash = var.save(db=db, subject=1)

        # Should not have lineage
        assert not db.has_lineage(vhash)

        provenance = db.get_provenance(scalar_class, subject=1)
        assert provenance is None

    def test_provenance_with_variable_inputs(self, db, scalar_class):
        """Test provenance tracking when thunk uses saved variables as inputs."""
        db.register(scalar_class)

        # Save input variable
        input_var = scalar_class(10)
        input_vhash = input_var.save(db=db, subject=1, role="input")

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        # Process the input
        result = process(input_var)
        output_var = scalar_class(result)
        output_vhash = output_var.save(db=db, subject=1, role="output")

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
        input_var = scalar_class(10)
        input_var.save(db=db, subject=1, role="input")

        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        @thunk(n_outputs=1)
        def triple(x):
            return x * 3

        # Create two outputs from the same input
        result1 = double(input_var)
        output1 = scalar_class(result1)
        output1.save(db=db, subject=1, role="doubled")

        result2 = triple(input_var)
        output2 = scalar_class(result2)
        output2.save(db=db, subject=1, role="tripled")

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
        var = scalar_class(c)
        var.save(db=db, subject=1)

        # Check provenance shows step3
        provenance = db.get_provenance(scalar_class, subject=1)
        assert provenance["function_name"] == "step3"

        # The input to step3 should be from step2
        assert len(provenance["inputs"]) == 1
        assert provenance["inputs"][0]["source_function"] == "step2"
