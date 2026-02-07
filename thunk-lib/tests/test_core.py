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
