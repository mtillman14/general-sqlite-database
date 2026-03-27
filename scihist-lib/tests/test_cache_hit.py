"""Tests that reloaded variables produce lineage cache hits."""

import numpy as np
import pytest

from scidb import BaseVariable
from scilineage import lineage_fcn
from scihist import save

from conftest import DEFAULT_TEST_SCHEMA_KEYS


class ArrayValue(BaseVariable):
    schema_version = 1


class ScalarValue(BaseVariable):
    schema_version = 1


class TestCacheHitAfterReload:
    """Test that reloaded variables produce cache hits."""

    def test_reload_and_rerun_hits_cache(self, db):
        """Saving a lineage_fcn result, reloading it, and re-running the same
        function with the reloaded input should hit the cache."""
        call_count = 0

        @lineage_fcn
        def double(x):
            nonlocal call_count
            call_count += 1
            return x * 2

        # Save raw data and load it back
        ArrayValue.save(np.array([1, 2, 3]), subject=1)
        loaded = ArrayValue.load(subject=1)

        # Run the function and save the result
        result1 = double(loaded)
        save(ScalarValue, result1, subject=1, trial=1)
        assert call_count == 1

        # Reload the input and re-run the same function
        reloaded = ArrayValue.load(subject=1)
        result2 = double(reloaded)

        # Function should NOT have been called again — cache hit
        assert call_count == 1
        np.testing.assert_array_equal(result2.data, result1.data)

    def test_chained_reload_hits_cache(self, db):
        """Multi-step pipeline: save intermediates, reload, re-run → cache hits."""
        sum_call_count = 0

        @lineage_fcn
        def add_one(x):
            return x + 1

        @lineage_fcn
        def sum_all(x):
            nonlocal sum_call_count
            sum_call_count += 1
            return float(np.sum(x))

        # Run full pipeline
        ArrayValue.save(np.array([10, 20, 30]), subject=1)
        loaded = ArrayValue.load(subject=1)

        step1 = add_one(loaded)
        save(ArrayValue, step1, subject=1, stage="incremented")

        step2 = sum_all(step1)
        save(ScalarValue, step2, subject=1, stage="total")
        assert sum_call_count == 1

        # Reload intermediate and re-run second step
        reloaded_step1 = ArrayValue.load(subject=1, stage="incremented")
        step2_again = sum_all(reloaded_step1)

        # Function should NOT have been called again — cache hit
        assert sum_call_count == 1
        assert step2_again.data == step2.data
