"""Tests for thunk core functionality."""

import pytest

from thunk import (
    ThunkOutput,
    PipelineThunk,
    Thunk,
    thunk,
)


class TestThunkDecorator:
    """Test the @thunk decorator."""

    def test_basic_thunk(self):
        @thunk
        def double(x):
            return x * 2

        result = double(5)
        assert isinstance(result, ThunkOutput)
        assert result.data == 10

    def test_thunk_preserves_name(self):
        @thunk
        def my_function(x):
            return x

        assert my_function.__name__ == "my_function"

    def test_multi_output(self):
        @thunk(unpack_output=True)
        def split(x):
            return x, x * 2

        a, b = split(5)
        assert a.data == 5
        assert b.data == 10
        assert a.output_num == 0
        assert b.output_num == 1

    def test_non_tuple_with_unpack_raises(self):
        @thunk(unpack_output=True)
        def wrong():
            return 42

        with pytest.raises(ValueError):
            wrong()


class TestThunkOutput:
    """Test ThunkOutput behavior."""

    def test_hash_deterministic(self):
        @thunk
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(5)
        assert r1.hash == r2.hash

    def test_hash_different_for_different_inputs(self):
        @thunk
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(6)
        assert r1.hash != r2.hash

    def test_str_shows_data(self):
        @thunk
        def process(x):
            return x * 2

        result = process(5)
        assert str(result) == "10"

    def test_equality_with_same_hash(self):
        @thunk
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(5)
        assert r1 == r2

    def test_equality_with_raw_data(self):
        @thunk
        def process(x):
            return x * 2

        result = process(5)
        assert result == 10


class TestPipelineThunk:
    """Test PipelineThunk behavior."""

    def test_captures_inputs(self):
        @thunk
        def process(x, y):
            return x + y

        result = process(5, 10)
        pt = result.pipeline_thunk
        assert pt.inputs == {"arg_0": 5, "arg_1": 10}

    def test_captures_kwargs(self):
        @thunk
        def process(x, factor=2):
            return x * factor

        result = process(5, factor=3)
        pt = result.pipeline_thunk
        assert "factor" in pt.inputs
        assert pt.inputs["factor"] == 3

    def test_lineage_hash_deterministic(self):
        @thunk
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(5)
        key1 = r1.pipeline_thunk.compute_lineage_hash()
        key2 = r2.pipeline_thunk.compute_lineage_hash()
        assert key1 == key2

    def test_lineage_hash_different_for_different_inputs(self):
        @thunk
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(6)
        key1 = r1.pipeline_thunk.compute_lineage_hash()
        key2 = r2.pipeline_thunk.compute_lineage_hash()
        assert key1 != key2


class TestSavedVariableClassification:
    """Test that saved variables with lineage are classified like ThunkOutputs."""

    def _make_saved_variable(self, lineage_hash=None):
        """Create a mock saved variable (duck-typed to match BaseVariable)."""
        class FakeVariable:
            def __init__(self, data, record_id, lineage_hash):
                self.data = data
                self.record_id = record_id
                self.lineage_hash = lineage_hash
                self.content_hash = "content123"
                self.metadata = {"subject": 1}

            def to_db(self):
                return self.data

            @classmethod
            def from_db(cls, df):
                return df

        return FakeVariable(42, "rec_abc", lineage_hash)

    def test_saved_variable_with_lineage_matches_thunk_output(self):
        """A saved variable with lineage_hash should produce the same
        cache tuple as the ThunkOutput it was saved from."""
        from thunk.inputs import classify_input

        @thunk
        def process(x):
            return x * 2

        result = process(5)

        # Classify the live ThunkOutput
        thunk_classified = classify_input("arg_0", result)

        # Create a saved variable with the ThunkOutput's hash
        saved_var = self._make_saved_variable(lineage_hash=result.hash)
        saved_classified = classify_input("arg_0", saved_var)

        assert thunk_classified.to_cache_tuple() == saved_classified.to_cache_tuple()

    def test_saved_variable_with_lineage_classified_as_thunk_output(self):
        """A saved variable with lineage_hash should be classified as THUNK_OUTPUT."""
        from thunk.inputs import classify_input, InputKind

        saved_var = self._make_saved_variable(lineage_hash="somehash")
        classified = classify_input("x", saved_var)

        assert classified.kind == InputKind.THUNK_OUTPUT

    def test_saved_variable_without_lineage_classified_as_saved(self):
        """A saved variable without lineage_hash should still be SAVED_VARIABLE."""
        from thunk.inputs import classify_input, InputKind

        saved_var = self._make_saved_variable(lineage_hash=None)
        classified = classify_input("x", saved_var)

        assert classified.kind == InputKind.SAVED_VARIABLE

    def test_downstream_lineage_hash_matches(self):
        """A downstream thunk should compute the same lineage hash whether
        its input is a live ThunkOutput or a saved-and-reloaded variable."""
        from thunk.inputs import classify_inputs

        @thunk
        def step1(x):
            return x + 1

        @thunk
        def step2(x):
            return x * 2

        # Path A: chain ThunkOutputs directly
        out1 = step1(5)
        out2_live = step2(out1)
        hash_live = out2_live.pipeline_thunk.compute_lineage_hash()

        # Path B: simulate save/reload of out1 then feed to step2
        saved_var = self._make_saved_variable(lineage_hash=out1.hash)
        out2_reloaded = step2(saved_var)
        hash_reloaded = out2_reloaded.pipeline_thunk.compute_lineage_hash()

        assert hash_live == hash_reloaded


class TestChaining:
    """Test chained thunk computations."""

    def test_basic_chain(self):
        @thunk
        def add_one(x):
            return x + 1

        @thunk
        def double(x):
            return x * 2

        result = double(add_one(5))
        assert result.data == 12  # (5 + 1) * 2

    def test_chain_captures_lineage(self):
        @thunk
        def step1(x):
            return x + 1

        @thunk
        def step2(x):
            return x * 2

        result = step2(step1(5))
        pt = result.pipeline_thunk

        # Input should be an ThunkOutput
        input_val = pt.inputs["arg_0"]
        assert isinstance(input_val, ThunkOutput)
        assert input_val.data == 6

    def test_unwrap_true_by_default(self):
        @thunk
        def check_type(x):
            # With unwrap=True, x should be raw data
            assert not isinstance(x, ThunkOutput)
            return x * 2

        @thunk
        def produce(x):
            return x + 1

        result = check_type(produce(5))
        assert result.data == 12

    def test_unwrap_false(self):
        @thunk
        def produce(x):
            return x + 1

        @thunk(unwrap=False)
        def check_type(x):
            # With unwrap=False, x should be ThunkOutput
            assert isinstance(x, ThunkOutput)
            return x.data * 2

        result = check_type(produce(5))
        assert result.data == 12
