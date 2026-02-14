"""Tests for for_each function."""

import pytest
from io import StringIO
import sys

from scirun import for_each, Fixed


class MockVariable:
    """Mock variable type for testing."""

    saved_data = []
    load_error = False

    def __init__(self, data):
        self.data = data

    @classmethod
    def load(cls, **metadata):
        if cls.load_error:
            raise ValueError("Mock load error")
        return cls(f"loaded_{metadata}")

    @classmethod
    def save(cls, data, **metadata):
        cls.saved_data.append({"data": data, "metadata": metadata})

    @classmethod
    def reset(cls):
        cls.saved_data = []
        cls.load_error = False


class MockVariableA(MockVariable):
    """First mock variable type."""

    saved_data = []
    load_error = False

    @classmethod
    def reset(cls):
        cls.saved_data = []
        cls.load_error = False


class MockVariableB(MockVariable):
    """Second mock variable type."""

    saved_data = []
    load_error = False

    @classmethod
    def reset(cls):
        cls.saved_data = []
        cls.load_error = False


class MockOutput(MockVariable):
    """Mock output variable type."""

    saved_data = []
    load_error = False

    @classmethod
    def reset(cls):
        cls.saved_data = []
        cls.load_error = False


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mock state before each test."""
    MockVariable.reset()
    MockVariableA.reset()
    MockVariableB.reset()
    MockOutput.reset()
    yield


class TestForEachBasic:
    """Basic tests for for_each function."""

    def test_single_iteration(self):
        """Should execute once for single value."""

        def process(x):
            return x + "_processed"

        for_each(
            process,
            inputs={"x": MockVariableA},
            outputs=[MockOutput],
            subject=[1],
        )

        assert len(MockOutput.saved_data) == 1
        assert MockOutput.saved_data[0]["metadata"] == {"subject": 1}

    def test_multiple_iterations(self):
        """Should execute for all combinations."""

        def process(x):
            return "result"

        for_each(
            process,
            inputs={"x": MockVariableA},
            outputs=[MockOutput],
            subject=[1, 2],
            session=["A", "B"],
        )

        # 2 subjects * 2 sessions = 4 iterations
        assert len(MockOutput.saved_data) == 4

    def test_multiple_inputs(self):
        """Should load multiple inputs."""

        def process(a, b):
            return f"{a}_{b}"

        for_each(
            process,
            inputs={"a": MockVariableA, "b": MockVariableB},
            outputs=[MockOutput],
            subject=[1],
        )

        assert len(MockOutput.saved_data) == 1

    def test_multiple_outputs(self):
        """Should save multiple outputs."""

        def process(x):
            return ("output_a", "output_b")

        for_each(
            process,
            inputs={"x": MockVariableA},
            outputs=[MockVariableA, MockVariableB],
            subject=[1],
        )

        assert len(MockVariableA.saved_data) == 1
        assert len(MockVariableB.saved_data) == 1
        assert MockVariableA.saved_data[0]["data"] == "output_a"
        assert MockVariableB.saved_data[0]["data"] == "output_b"


class TestForEachWithFixed:
    """Tests for for_each with Fixed inputs."""

    def test_fixed_overrides_metadata(self):
        """Fixed should override iteration metadata."""
        loaded_metadata = []

        class TrackingVariable:
            @classmethod
            def load(cls, **metadata):
                loaded_metadata.append(metadata)
                return MockVariableA(f"data_{metadata}")

        def process(baseline, current):
            return "result"

        for_each(
            process,
            inputs={
                "baseline": Fixed(TrackingVariable, session="BL"),
                "current": TrackingVariable,
            },
            outputs=[MockOutput],
            subject=[1],
            session=["A", "B"],
        )

        # Check baseline always loaded with session="BL"
        baseline_loads = [m for m in loaded_metadata if m["session"] == "BL"]
        assert len(baseline_loads) == 2  # Once per iteration

        # Check current loaded with iteration session
        current_loads = [m for m in loaded_metadata if m["session"] in ["A", "B"]]
        assert len(current_loads) == 2


class TestForEachDryRun:
    """Tests for dry_run mode."""

    def test_dry_run_no_execution(self, capsys):
        """Dry run should not execute function or save."""

        def process(x):
            raise RuntimeError("Should not be called")

        for_each(
            process,
            inputs={"x": MockVariableA},
            outputs=[MockOutput],
            dry_run=True,
            subject=[1, 2],
        )

        assert len(MockOutput.saved_data) == 0

        captured = capsys.readouterr()
        assert "[dry-run]" in captured.out

    def test_dry_run_shows_iterations(self, capsys):
        """Dry run should show what would happen."""

        def my_func(x):
            return x

        for_each(
            my_func,
            inputs={"x": MockVariableA},
            outputs=[MockOutput],
            dry_run=True,
            subject=[1],
            session=["A"],
        )

        captured = capsys.readouterr()
        assert "my_func" in captured.out
        assert "MockVariableA" in captured.out
        assert "MockOutput" in captured.out


class TestForEachErrorHandling:
    """Tests for error handling."""

    def test_skip_on_load_failure(self, capsys):
        """Should skip iteration if input fails to load."""
        MockVariableA.load_error = True

        def process(x):
            return "result"

        for_each(
            process,
            inputs={"x": MockVariableA},
            outputs=[MockOutput],
            subject=[1, 2],
        )

        # No outputs should be saved
        assert len(MockOutput.saved_data) == 0

        captured = capsys.readouterr()
        assert "[skip]" in captured.out

    def test_skip_on_function_error(self, capsys):
        """Should skip iteration if function raises."""

        def failing_process(x):
            raise ValueError("Processing failed")

        for_each(
            failing_process,
            inputs={"x": MockVariableA},
            outputs=[MockOutput],
            subject=[1],
        )

        assert len(MockOutput.saved_data) == 0

        captured = capsys.readouterr()
        assert "[skip]" in captured.out
        assert "Processing failed" in captured.out

    def test_continues_after_error(self, capsys):
        """Should continue processing after error."""
        call_count = [0]

        def sometimes_fails(x):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("First call fails")
            return "result"

        for_each(
            sometimes_fails,
            inputs={"x": MockVariableA},
            outputs=[MockOutput],
            subject=[1, 2, 3],
        )

        # First failed, second and third succeeded
        assert len(MockOutput.saved_data) == 2


class TestForEachOutput:
    """Tests for output normalization."""

    def test_single_output_not_tuple(self):
        """Single output should be normalized from non-tuple."""

        def process(x):
            return "single_result"  # Not a tuple

        for_each(
            process,
            inputs={"x": MockVariableA},
            outputs=[MockOutput],
            subject=[1],
        )

        assert len(MockOutput.saved_data) == 1
        assert MockOutput.saved_data[0]["data"] == "single_result"

    def test_output_metadata_matches_iteration(self):
        """Output should be saved with iteration metadata."""

        def process(x):
            return "result"

        for_each(
            process,
            inputs={"x": MockVariableA},
            outputs=[MockOutput],
            subject=[42],
            session=["XYZ"],
        )

        assert MockOutput.saved_data[0]["metadata"] == {
            "subject": 42,
            "session": "XYZ",
        }


class TestForEachWithConstants:
    """Tests for constant values in inputs dict."""

    def test_constant_passed_to_function(self):
        """Constants should be passed as kwargs to the function."""
        received_args = {}

        def process(x, smoothing):
            received_args["x"] = x
            received_args["smoothing"] = smoothing
            return "result"

        for_each(
            process,
            inputs={"x": MockVariableA, "smoothing": 0.2},
            outputs=[MockOutput],
            subject=[1],
        )

        assert received_args["smoothing"] == 0.2
        assert len(MockOutput.saved_data) == 1

    def test_constant_saved_as_metadata(self):
        """Constants should be included in save metadata as version keys."""

        def process(x, smoothing):
            return "result"

        for_each(
            process,
            inputs={"x": MockVariableA, "smoothing": 0.2},
            outputs=[MockOutput],
            subject=[1],
        )

        assert MockOutput.saved_data[0]["metadata"] == {
            "subject": 1,
            "smoothing": 0.2,
        }

    def test_constant_with_variable_inputs(self):
        """Constants and variable inputs should work together."""
        received_args = {}

        def process(a, b, threshold):
            received_args["a"] = a
            received_args["b"] = b
            received_args["threshold"] = threshold
            return "result"

        for_each(
            process,
            inputs={"a": MockVariableA, "b": MockVariableB, "threshold": 10},
            outputs=[MockOutput],
            subject=[1, 2],
        )

        # 2 iterations, both should have threshold in metadata
        assert len(MockOutput.saved_data) == 2
        assert MockOutput.saved_data[0]["metadata"] == {"subject": 1, "threshold": 10}
        assert MockOutput.saved_data[1]["metadata"] == {"subject": 2, "threshold": 10}
        assert received_args["threshold"] == 10

    def test_multiple_constants(self):
        """Multiple constants should all be passed and saved."""

        def process(x, low_hz, high_hz, method):
            return f"{low_hz}-{high_hz}-{method}"

        for_each(
            process,
            inputs={
                "x": MockVariableA,
                "low_hz": 20,
                "high_hz": 450,
                "method": "bandpass",
            },
            outputs=[MockOutput],
            subject=[1],
        )

        assert MockOutput.saved_data[0]["data"] == "20-450-bandpass"
        assert MockOutput.saved_data[0]["metadata"] == {
            "subject": 1,
            "low_hz": 20,
            "high_hz": 450,
            "method": "bandpass",
        }

    def test_constant_not_loaded(self):
        """Constants should not trigger .load() calls."""
        load_count = [0]

        class CountingVariable:
            @classmethod
            def load(cls, **metadata):
                load_count[0] += 1
                return MockVariableA(f"data_{metadata}")

        def process(x, factor):
            return "result"

        for_each(
            process,
            inputs={"x": CountingVariable, "factor": 2.5},
            outputs=[MockOutput],
            subject=[1],
        )

        # Only the variable should trigger a load, not the constant
        assert load_count[0] == 1

    def test_constant_in_dry_run(self, capsys):
        """Dry run should display constants correctly."""

        def my_func(x, smoothing):
            return x

        for_each(
            my_func,
            inputs={"x": MockVariableA, "smoothing": 0.2},
            outputs=[MockOutput],
            dry_run=True,
            subject=[1],
        )

        captured = capsys.readouterr()
        assert "constant smoothing = 0.2" in captured.out
        assert "smoothing=0.2" in captured.out  # in save line
        assert "MockVariableA" in captured.out

    def test_constant_with_fixed(self):
        """Constants should work alongside Fixed inputs."""

        def process(baseline, current, threshold):
            return "result"

        for_each(
            process,
            inputs={
                "baseline": Fixed(MockVariableA, session="BL"),
                "current": MockVariableB,
                "threshold": 5.0,
            },
            outputs=[MockOutput],
            subject=[1],
            session=["A"],
        )

        assert len(MockOutput.saved_data) == 1
        assert MockOutput.saved_data[0]["metadata"] == {
            "subject": 1,
            "session": "A",
            "threshold": 5.0,
        }


class TestForEachAsTable:
    """Tests for as_table parameter in for_each."""

    def test_as_table_converts_list_to_dataframe(self):
        """When as_table includes an input name, multi-result loads become DataFrames."""
        import pandas as pd

        class MultiResultVar:
            """Mock variable that returns a list from load()."""

            @classmethod
            def load(cls, **metadata):
                # Simulate multi-result: return a list of mock vars
                results = []
                for i in range(3):
                    v = MockVariableA(i * 10)
                    v.metadata = {"subject": str(metadata.get("subject", 1)), "trial": str(i + 1)}
                    v.version_id = 1
                    results.append(v)
                return results

            @classmethod
            def view_name(cls):
                return "MultiResultVar"

        received = {}

        def process(values):
            received["values"] = values
            return "result"

        for_each(
            process,
            inputs={"values": MultiResultVar},
            outputs=[MockOutput],
            as_table=["values"],
            subject=[1],
        )

        assert isinstance(received["values"], pd.DataFrame)
        assert len(received["values"]) == 3
        assert "MultiResultVar" in received["values"].columns
        assert "version_id" in received["values"].columns

    def test_without_as_table_returns_list(self):
        """Without as_table, multi-result loads stay as lists."""

        class MultiResultVar:
            @classmethod
            def load(cls, **metadata):
                results = []
                for i in range(3):
                    v = MockVariableA(i * 10)
                    v.metadata = {"subject": "1", "trial": str(i + 1)}
                    v.version_id = 1
                    results.append(v)
                return results

            @classmethod
            def view_name(cls):
                return "MultiResultVar"

        received = {}

        def process(values):
            received["values"] = values
            return "result"

        for_each(
            process,
            inputs={"values": MultiResultVar},
            outputs=[MockOutput],
            subject=[1],
        )

        # Without as_table, the list should be unwrapped to raw data
        # (each element's .data), not converted to DataFrame
        assert not hasattr(received["values"], "columns")

    def test_as_table_only_affects_specified_inputs(self):
        """as_table should only convert specified inputs, not all."""
        import pandas as pd

        class MultiA:
            @classmethod
            def load(cls, **metadata):
                results = []
                for i in range(2):
                    v = MockVariableA(i)
                    v.metadata = {"subject": "1", "trial": str(i + 1)}
                    v.version_id = 1
                    results.append(v)
                return results

            @classmethod
            def view_name(cls):
                return "MultiA"

        class SingleB:
            @classmethod
            def load(cls, **metadata):
                v = MockVariableB("single")
                return v

        received = {}

        def process(a, b):
            received["a"] = a
            received["b"] = b
            return "result"

        for_each(
            process,
            inputs={"a": MultiA, "b": SingleB},
            outputs=[MockOutput],
            as_table=["a"],
            subject=[1],
        )

        assert isinstance(received["a"], pd.DataFrame)
        # b is single, so it gets unwrapped to raw data
        assert received["b"] == "single"

    def test_as_table_single_result_not_converted(self):
        """as_table should not convert single-result loads."""
        import pandas as pd

        class SingleVar:
            @classmethod
            def load(cls, **metadata):
                v = MockVariableA(42)
                v.metadata = {"subject": "1"}
                v.version_id = 1
                return v  # Single result, not a list

            @classmethod
            def view_name(cls):
                return "SingleVar"

        received = {}

        def process(values):
            received["values"] = values
            return "result"

        for_each(
            process,
            inputs={"values": SingleVar},
            outputs=[MockOutput],
            as_table=["values"],
            subject=[1],
        )

        # Single result gets unwrapped to .data, not converted to DataFrame
        assert not isinstance(received["values"], pd.DataFrame)
        assert received["values"] == 42

    def test_as_table_true_converts_all_loadable_inputs(self):
        """as_table=True should convert all loadable multi-result inputs."""
        import pandas as pd

        class MultiA:
            @classmethod
            def load(cls, **metadata):
                results = []
                for i in range(2):
                    v = MockVariableA(i)
                    v.metadata = {"subject": "1", "trial": str(i + 1)}
                    v.version_id = 1
                    results.append(v)
                return results

            @classmethod
            def view_name(cls):
                return "MultiA"

        class MultiB:
            @classmethod
            def load(cls, **metadata):
                results = []
                for i in range(2):
                    v = MockVariableB(i * 100)
                    v.metadata = {"subject": "1", "trial": str(i + 1)}
                    v.version_id = 1
                    results.append(v)
                return results

            @classmethod
            def view_name(cls):
                return "MultiB"

        received = {}

        def process(a, b):
            received["a"] = a
            received["b"] = b
            return "result"

        for_each(
            process,
            inputs={"a": MultiA, "b": MultiB},
            outputs=[MockOutput],
            as_table=True,
            subject=[1],
        )

        assert isinstance(received["a"], pd.DataFrame)
        assert isinstance(received["b"], pd.DataFrame)
        assert "MultiA" in received["a"].columns
        assert "MultiB" in received["b"].columns

    def test_as_table_false_no_conversion(self):
        """as_table=False should not convert any inputs."""

        class MultiResultVar:
            @classmethod
            def load(cls, **metadata):
                results = []
                for i in range(2):
                    v = MockVariableA(i)
                    v.metadata = {"subject": "1", "trial": str(i + 1)}
                    v.version_id = 1
                    results.append(v)
                return results

            @classmethod
            def view_name(cls):
                return "MultiResultVar"

        received = {}

        def process(values):
            received["values"] = values
            return "result"

        for_each(
            process,
            inputs={"values": MultiResultVar},
            outputs=[MockOutput],
            as_table=False,
            subject=[1],
        )

        # False should mean no conversion — list gets unwrapped
        assert not hasattr(received["values"], "columns")

    def test_as_table_true_skips_constants(self):
        """as_table=True should only affect loadable inputs, not constants."""
        import pandas as pd

        class MultiA:
            @classmethod
            def load(cls, **metadata):
                results = []
                for i in range(2):
                    v = MockVariableA(i)
                    v.metadata = {"subject": "1", "trial": str(i + 1)}
                    v.version_id = 1
                    results.append(v)
                return results

            @classmethod
            def view_name(cls):
                return "MultiA"

        received = {}

        def process(a, factor):
            received["a"] = a
            received["factor"] = factor
            return "result"

        for_each(
            process,
            inputs={"a": MultiA, "factor": 2.5},
            outputs=[MockOutput],
            as_table=True,
            subject=[1],
        )

        assert isinstance(received["a"], pd.DataFrame)
        assert received["factor"] == 2.5
