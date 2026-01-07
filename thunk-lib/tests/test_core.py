"""Tests for thunk core functionality."""

import pytest

from thunk import (
    OutputThunk,
    PipelineThunk,
    Thunk,
    configure_cache,
    get_cache_backend,
    thunk,
)


class TestThunkDecorator:
    """Test the @thunk decorator."""

    def test_basic_thunk(self):
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        result = double(5)
        assert isinstance(result, OutputThunk)
        assert result.data == 10

    def test_thunk_preserves_name(self):
        @thunk(n_outputs=1)
        def my_function(x):
            return x

        assert my_function.__name__ == "my_function"

    def test_multi_output(self):
        @thunk(n_outputs=2)
        def split(x):
            return x, x * 2

        a, b = split(5)
        assert a.data == 5
        assert b.data == 10
        assert a.output_num == 0
        assert b.output_num == 1

    def test_wrong_output_count_raises(self):
        @thunk(n_outputs=2)
        def wrong():
            return 42

        with pytest.raises(ValueError):
            wrong()


class TestOutputThunk:
    """Test OutputThunk behavior."""

    def test_hash_deterministic(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(5)
        assert r1.hash == r2.hash

    def test_hash_different_for_different_inputs(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(6)
        assert r1.hash != r2.hash

    def test_str_shows_data(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(5)
        assert str(result) == "10"

    def test_equality_with_same_hash(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(5)
        assert r1 == r2

    def test_equality_with_raw_data(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(5)
        assert result == 10


class TestPipelineThunk:
    """Test PipelineThunk behavior."""

    def test_captures_inputs(self):
        @thunk(n_outputs=1)
        def process(x, y):
            return x + y

        result = process(5, 10)
        pt = result.pipeline_thunk
        assert pt.inputs == {"arg_0": 5, "arg_1": 10}

    def test_captures_kwargs(self):
        @thunk(n_outputs=1)
        def process(x, factor=2):
            return x * factor

        result = process(5, factor=3)
        pt = result.pipeline_thunk
        assert "factor" in pt.inputs
        assert pt.inputs["factor"] == 3

    def test_cache_key_deterministic(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(5)
        key1 = r1.pipeline_thunk.compute_cache_key()
        key2 = r2.pipeline_thunk.compute_cache_key()
        assert key1 == key2

    def test_cache_key_different_for_different_inputs(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        r1 = process(5)
        r2 = process(6)
        key1 = r1.pipeline_thunk.compute_cache_key()
        key2 = r2.pipeline_thunk.compute_cache_key()
        assert key1 != key2


class TestChaining:
    """Test chained thunk computations."""

    def test_basic_chain(self):
        @thunk(n_outputs=1)
        def add_one(x):
            return x + 1

        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        result = double(add_one(5))
        assert result.data == 12  # (5 + 1) * 2

    def test_chain_captures_lineage(self):
        @thunk(n_outputs=1)
        def step1(x):
            return x + 1

        @thunk(n_outputs=1)
        def step2(x):
            return x * 2

        result = step2(step1(5))
        pt = result.pipeline_thunk

        # Input should be an OutputThunk
        input_val = pt.inputs["arg_0"]
        assert isinstance(input_val, OutputThunk)
        assert input_val.data == 6

    def test_unwrap_true_by_default(self):
        @thunk(n_outputs=1)
        def check_type(x):
            # With unwrap=True, x should be raw data
            assert not isinstance(x, OutputThunk)
            return x * 2

        @thunk(n_outputs=1)
        def produce(x):
            return x + 1

        result = check_type(produce(5))
        assert result.data == 12

    def test_unwrap_false(self):
        @thunk(n_outputs=1)
        def produce(x):
            return x + 1

        @thunk(n_outputs=1, unwrap=False)
        def check_type(x):
            # With unwrap=False, x should be OutputThunk
            assert isinstance(x, OutputThunk)
            return x.data * 2

        result = check_type(produce(5))
        assert result.data == 12


class TestCacheBackend:
    """Test cache backend configuration."""

    def setup_method(self):
        # Clear cache before each test
        configure_cache(None)

    def teardown_method(self):
        # Clear cache after each test
        configure_cache(None)

    def test_no_cache_by_default(self):
        assert get_cache_backend() is None

    def test_configure_cache(self):
        class MyCache:
            def get_cached(self, cache_key, n_outputs):
                return None

        cache = MyCache()
        configure_cache(cache)
        assert get_cache_backend() is cache

    def test_cache_hit(self):
        execution_count = [0]

        @thunk(n_outputs=1)
        def expensive(x):
            execution_count[0] += 1
            return x * 2

        class SimpleCache:
            def __init__(self):
                self.store = {}

            def get_cached(self, cache_key, n_outputs):
                if cache_key in self.store:
                    return self.store[cache_key]
                return None

        cache = SimpleCache()
        configure_cache(cache)

        # First call - executes
        r1 = expensive(5)
        assert execution_count[0] == 1
        assert r1.was_cached is False

        # Store in cache
        cache.store[r1.pipeline_thunk.compute_cache_key()] = [(r1.data, "id1")]

        # Second call - cache hit
        r2 = expensive(5)
        assert execution_count[0] == 1  # Not executed again
        assert r2.was_cached is True
        assert r2.cached_id == "id1"
