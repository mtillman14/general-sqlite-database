"""Tests for scidb.foreach module."""

import pytest
import pandas as pd

from scidb import for_each, Fixed, BaseVariable, thunk


class InputVar(BaseVariable):
    """Input variable for testing."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": [self.data]})

    @classmethod
    def from_db(cls, df: pd.DataFrame):
        return df["value"].iloc[0]


class OutputVar(BaseVariable):
    """Output variable for testing."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": [self.data]})

    @classmethod
    def from_db(cls, df: pd.DataFrame):
        return df["value"].iloc[0]


class BaselineVar(BaseVariable):
    """Baseline variable for Fixed testing."""
    schema_version = 1

    def to_db(self) -> pd.DataFrame:
        return pd.DataFrame({"value": [self.data]})

    @classmethod
    def from_db(cls, df: pd.DataFrame):
        return df["value"].iloc[0]


class TestFixed:
    """Test the Fixed wrapper class."""

    def test_fixed_stores_type_and_metadata(self):
        fixed = Fixed(InputVar, session="BL", condition="control")

        assert fixed.var_type is InputVar
        assert fixed.fixed_metadata == {"session": "BL", "condition": "control"}

    def test_fixed_empty_metadata(self):
        fixed = Fixed(InputVar)

        assert fixed.var_type is InputVar
        assert fixed.fixed_metadata == {}


class TestForEachDryRun:
    """Test for_each in dry-run mode."""

    def test_dry_run_prints_summary(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        def process(x):
            return x * 2

        for_each(
            process,
            inputs={"x": InputVar},
            outputs=[OutputVar],
            dry_run=True,
            subject=["S1", "S2"],
            session=["A", "B"],
        )

        captured = capsys.readouterr()
        assert "[dry-run] for_each(process)" in captured.out
        assert "4 iterations" in captured.out
        assert "['subject', 'session']" in captured.out

    def test_dry_run_shows_each_iteration(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        def process(x):
            return x * 2

        for_each(
            process,
            inputs={"x": InputVar},
            outputs=[OutputVar],
            dry_run=True,
            subject=["S1"],
            session=["A"],
        )

        captured = capsys.readouterr()
        assert "load x = InputVar.load(subject=S1, session=A)" in captured.out
        assert "save OutputVar.save(..., subject=S1, session=A)" in captured.out

    def test_dry_run_shows_fixed_metadata(self, configured_db, capsys):
        configured_db.register(BaselineVar)
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        def compare(baseline, current):
            return current - baseline

        for_each(
            compare,
            inputs={
                "baseline": Fixed(BaselineVar, session="BL"),
                "current": InputVar,
            },
            outputs=[OutputVar],
            dry_run=True,
            subject=["S1"],
            session=["POST"],
        )

        captured = capsys.readouterr()
        # Baseline should use fixed session="BL"
        assert "load baseline = BaselineVar.load(subject=S1, session=BL)" in captured.out
        # Current should use iteration session="POST"
        assert "load current = InputVar.load(subject=S1, session=POST)" in captured.out


class TestForEachExecution:
    """Test for_each actual execution."""

    def test_basic_execution(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        # Save some input data
        InputVar.save(10, subject="S1", session="A")
        InputVar.save(20, subject="S1", session="B")

        def double(x):
            return x * 2

        for_each(
            double,
            inputs={"x": InputVar},
            outputs=[OutputVar],
            subject=["S1"],
            session=["A", "B"],
        )

        captured = capsys.readouterr()
        assert "completed=2" in captured.out

        # Verify outputs were saved
        out_a = OutputVar.load(subject="S1", session="A")
        out_b = OutputVar.load(subject="S1", session="B")
        assert out_a.data == 20
        assert out_b.data == 40

    def test_skips_missing_data(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        # Only save data for one combination
        InputVar.save(10, subject="S1", session="A")
        # S1/B is missing

        def double(x):
            return x * 2

        for_each(
            double,
            inputs={"x": InputVar},
            outputs=[OutputVar],
            subject=["S1"],
            session=["A", "B"],
        )

        captured = capsys.readouterr()
        assert "completed=1" in captured.out
        assert "skipped=1" in captured.out
        assert "[skip]" in captured.out

    def test_multiple_inputs(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        class InputVar2(BaseVariable):
            schema_version = 1
            def to_db(self):
                return pd.DataFrame({"value": [self.data]})
            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        configured_db.register(InputVar2)

        InputVar.save(10, subject="S1", session="A")
        InputVar2.save(5, subject="S1", session="A")

        def add(x, y):
            return x + y

        for_each(
            add,
            inputs={"x": InputVar, "y": InputVar2},
            outputs=[OutputVar],
            subject=["S1"],
            session=["A"],
        )

        out = OutputVar.load(subject="S1", session="A")
        assert out.data == 15

    def test_multiple_outputs(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        class OutputVar2(BaseVariable):
            schema_version = 1
            def to_db(self):
                return pd.DataFrame({"value": [self.data]})
            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        configured_db.register(OutputVar2)

        InputVar.save(10, subject="S1", session="A")

        def split(x):
            return x * 2, x * 3

        for_each(
            split,
            inputs={"x": InputVar},
            outputs=[OutputVar, OutputVar2],
            subject=["S1"],
            session=["A"],
        )

        out1 = OutputVar.load(subject="S1", session="A")
        out2 = OutputVar2.load(subject="S1", session="A")
        assert out1.data == 20
        assert out2.data == 30

    def test_fixed_metadata_override(self, configured_db, capsys):
        configured_db.register(BaselineVar)
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        # Baseline is always at session="BL"
        BaselineVar.save(100, subject="S1", session="BL")
        # Current varies by session
        InputVar.save(150, subject="S1", session="POST")
        InputVar.save(120, subject="S1", session="MID")

        def delta(baseline, current):
            return current - baseline

        for_each(
            delta,
            inputs={
                "baseline": Fixed(BaselineVar, session="BL"),
                "current": InputVar,
            },
            outputs=[OutputVar],
            subject=["S1"],
            session=["POST", "MID"],
        )

        out_post = OutputVar.load(subject="S1", session="POST")
        out_mid = OutputVar.load(subject="S1", session="MID")
        assert out_post.data == 50  # 150 - 100
        assert out_mid.data == 20   # 120 - 100

    def test_works_with_thunked_function(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        InputVar.save(10, subject="S1", session="A")

        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        for_each(
            double,
            inputs={"x": InputVar},
            outputs=[OutputVar],
            subject=["S1"],
            session=["A"],
        )

        out = OutputVar.load(subject="S1", session="A")
        assert out.data == 20

    def test_handles_function_exception(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        InputVar.save(10, subject="S1", session="A")
        InputVar.save(0, subject="S1", session="B")

        def divide_100_by(x):
            return 100 / x  # Will raise ZeroDivisionError for x=0

        for_each(
            divide_100_by,
            inputs={"x": InputVar},
            outputs=[OutputVar],
            subject=["S1"],
            session=["A", "B"],
        )

        captured = capsys.readouterr()
        assert "completed=1" in captured.out
        assert "skipped=1" in captured.out
        assert "divide_100_by raised" in captured.out

        # First one should have succeeded
        out = OutputVar.load(subject="S1", session="A")
        assert out.data == 10.0


class TestForEachEdgeCases:
    """Test edge cases for for_each."""

    def test_empty_iteration(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        def process(x):
            return x

        for_each(
            process,
            inputs={"x": InputVar},
            outputs=[OutputVar],
            subject=[],  # Empty!
            session=["A"],
        )

        captured = capsys.readouterr()
        assert "completed=0" in captured.out
        assert "total=0" in captured.out

    def test_single_iteration(self, configured_db, capsys):
        configured_db.register(InputVar)
        configured_db.register(OutputVar)

        InputVar.save(42, subject="S1", session="A")

        def process(x):
            return x

        for_each(
            process,
            inputs={"x": InputVar},
            outputs=[OutputVar],
            subject=["S1"],
            session=["A"],
        )

        captured = capsys.readouterr()
        assert "completed=1" in captured.out
        assert "total=1" in captured.out
