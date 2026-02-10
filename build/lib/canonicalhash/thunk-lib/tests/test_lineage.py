"""Tests for thunk lineage tracking."""

import pytest

from thunk import (
    LineageRecord,
    extract_lineage,
    get_upstream_lineage,
    get_raw_value,
    thunk,
)


class TestExtractLineage:
    """Test extract_lineage function."""

    def test_basic_lineage(self):
        @thunk
        def process(x):
            return x * 2

        result = process(5)
        lineage = extract_lineage(result)

        assert lineage.function_name == "process"
        assert isinstance(lineage.function_hash, str)
        assert len(lineage.function_hash) == 64  # SHA-256

    def test_constant_inputs_captured(self):
        @thunk
        def process(x, factor):
            return x * factor

        result = process(5, 3)
        lineage = extract_lineage(result)

        # Both inputs should be constants (literals)
        assert len(lineage.constants) == 2
        names = {c["name"] for c in lineage.constants}
        assert "arg_0" in names
        assert "arg_1" in names

    def test_thunk_inputs_captured(self):
        @thunk
        def step1(x):
            return x + 1

        @thunk
        def step2(x):
            return x * 2

        result = step2(step1(5))
        lineage = extract_lineage(result)

        assert lineage.function_name == "step2"
        assert len(lineage.inputs) == 1
        assert lineage.inputs[0]["source_type"] == "thunk"
        assert lineage.inputs[0]["source_function"] == "step1"


class TestGetUpstreamLineage:
    """Test get_upstream_lineage function."""

    def test_single_step(self):
        @thunk
        def process(x):
            return x * 2

        result = process(5)
        chain = get_upstream_lineage(result)

        assert len(chain) == 1
        assert chain[0]["function_name"] == "process"

    def test_multi_step_chain(self):
        @thunk
        def step1(x):
            return x + 1

        @thunk
        def step2(x):
            return x * 2

        @thunk
        def step3(x):
            return x - 1

        result = step3(step2(step1(5)))
        chain = get_upstream_lineage(result)

        assert len(chain) == 3
        names = [r["function_name"] for r in chain]
        assert names == ["step3", "step2", "step1"]

    def test_max_depth_limit(self):
        @thunk
        def step(x):
            return x + 1

        # Build a long chain
        result = step(1)
        for _ in range(10):
            result = step(result)

        # With max_depth=3, should only get 3 records
        chain = get_upstream_lineage(result, max_depth=3)
        assert len(chain) <= 3


class TestGetRawValue:
    """Test get_raw_value function."""

    def test_unwraps_thunk_output(self):
        @thunk
        def process(x):
            return x * 2

        result = process(5)
        raw = get_raw_value(result)
        assert raw == 10

    def test_returns_raw_value_unchanged(self):
        raw = get_raw_value(42)
        assert raw == 42

        raw = get_raw_value([1, 2, 3])
        assert raw == [1, 2, 3]


class TestLineageRecord:
    """Test LineageRecord dataclass."""

    def test_to_dict(self):
        record = LineageRecord(
            function_name="process",
            function_hash="abc123",
            inputs=[{"name": "x", "source_type": "variable"}],
            constants=[{"name": "factor", "value_hash": "def456"}],
        )

        data = record.to_dict()
        assert data["function_name"] == "process"
        assert data["function_hash"] == "abc123"
        assert len(data["inputs"]) == 1
        assert len(data["constants"]) == 1

    def test_from_dict(self):
        data = {
            "function_name": "process",
            "function_hash": "abc123",
            "inputs": [{"name": "x"}],
            "constants": [{"name": "factor"}],
        }

        record = LineageRecord.from_dict(data)
        assert record.function_name == "process"
        assert record.function_hash == "abc123"
        assert len(record.inputs) == 1
        assert len(record.constants) == 1

    def test_roundtrip(self):
        original = LineageRecord(
            function_name="process",
            function_hash="abc123",
            inputs=[{"name": "x", "source_type": "variable", "record_id": "v1"}],
            constants=[{"name": "factor", "value_hash": "def456", "value_repr": "2"}],
        )

        restored = LineageRecord.from_dict(original.to_dict())

        assert restored.function_name == original.function_name
        assert restored.function_hash == original.function_hash
        assert restored.inputs == original.inputs
        assert restored.constants == original.constants
