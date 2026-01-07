"""Tests for scidb computation caching (Phase 3)."""

import numpy as np
import pytest

from scidb import (
    BaseVariable,
    DatabaseManager,
    check_cache,
    thunk,
)
from scidb.thunk import OutputThunk, PipelineThunk


class TestCacheTableCreation:
    """Test that the cache table is created properly."""

    def test_cache_table_exists(self, db):
        """The _computation_cache table should be created on init."""
        cursor = db.connection.execute(
            """SELECT name FROM sqlite_master
               WHERE type='table' AND name='_computation_cache'"""
        )
        assert cursor.fetchone() is not None

    def test_cache_table_has_correct_columns(self, db):
        """Check cache table schema."""
        cursor = db.connection.execute("PRAGMA table_info(_computation_cache)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "id",
            "cache_key",
            "output_num",
            "function_name",
            "function_hash",
            "output_type",
            "output_vhash",
            "created_at",
        }
        assert columns == expected

    def test_cache_index_exists(self, db):
        """Check that cache key index exists."""
        cursor = db.connection.execute(
            """SELECT name FROM sqlite_master
               WHERE type='index' AND name='idx_cache_key'"""
        )
        assert cursor.fetchone() is not None


class TestCacheKeyComputation:
    """Test PipelineThunk.compute_cache_key()."""

    def test_cache_key_is_string(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(10)
        cache_key = result.pipeline_thunk.compute_cache_key()

        assert isinstance(cache_key, str)
        assert len(cache_key) == 64  # SHA-256

    def test_same_inputs_same_cache_key(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result1 = process(10)
        result2 = process(10)

        key1 = result1.pipeline_thunk.compute_cache_key()
        key2 = result2.pipeline_thunk.compute_cache_key()

        assert key1 == key2

    def test_different_inputs_different_cache_key(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result1 = process(10)
        result2 = process(20)

        key1 = result1.pipeline_thunk.compute_cache_key()
        key2 = result2.pipeline_thunk.compute_cache_key()

        assert key1 != key2

    def test_cache_key_with_saved_variable(self, db, scalar_class):
        """Cache key should use vhash for saved variables."""
        db.register(scalar_class)

        var = scalar_class(42)
        var.save(db=db, subject=1)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(var)
        cache_key = result.pipeline_thunk.compute_cache_key()

        # Same saved variable should produce same key
        result2 = process(var)
        cache_key2 = result2.pipeline_thunk.compute_cache_key()

        assert cache_key == cache_key2

    def test_different_functions_different_cache_key(self):
        @thunk(n_outputs=1)
        def process1(x):
            return x * 2

        @thunk(n_outputs=1)
        def process2(x):
            return x * 3

        result1 = process1(10)
        result2 = process2(10)

        key1 = result1.pipeline_thunk.compute_cache_key()
        key2 = result2.pipeline_thunk.compute_cache_key()

        assert key1 != key2


class TestCachePopulation:
    """Test that saving populates the cache."""

    def test_save_populates_cache(self, db, scalar_class):
        """Saving a thunk result should populate the cache."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        result = double(21)
        var = scalar_class(result)
        vhash = var.save(db=db, subject=1)

        # Check cache table has an entry
        cursor = db.connection.execute(
            "SELECT * FROM _computation_cache WHERE output_vhash = ?",
            (vhash,),
        )
        row = cursor.fetchone()

        assert row is not None
        assert row["function_name"] == "double"
        assert row["output_type"] == "ScalarValue"

    def test_save_without_thunk_no_cache(self, db, scalar_class):
        """Saving raw data should not populate cache."""
        db.register(scalar_class)

        var = scalar_class(42)
        vhash = var.save(db=db, subject=1)

        # No cache entry should exist
        cursor = db.connection.execute(
            "SELECT * FROM _computation_cache WHERE output_vhash = ?",
            (vhash,),
        )
        assert cursor.fetchone() is None

    def test_cache_key_stored_correctly(self, db, scalar_class):
        """Verify the cache key matches what we compute."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def triple(x):
            return x * 3

        result = triple(10)
        expected_key = result.pipeline_thunk.compute_cache_key()

        var = scalar_class(result)
        var.save(db=db, subject=1)

        cursor = db.connection.execute(
            "SELECT cache_key FROM _computation_cache WHERE function_name = 'triple'"
        )
        row = cursor.fetchone()

        assert row["cache_key"] == expected_key


class TestCacheLookup:
    """Test cache lookup functionality."""

    def test_get_cached_computation_hit(self, db, scalar_class):
        """Should return cached result when available."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        # Save to populate cache
        result = process(10)
        var = scalar_class(result)
        original_vhash = var.save(db=db, subject=1)

        # Compute cache key
        cache_key = result.pipeline_thunk.compute_cache_key()

        # Look up from cache
        cached = db.get_cached_computation(cache_key, scalar_class)

        assert cached is not None
        assert cached.data == 20
        assert cached.vhash == original_vhash

    def test_get_cached_computation_miss(self, db, scalar_class):
        """Should return None when not cached."""
        db.register(scalar_class)

        cached = db.get_cached_computation("nonexistent_key", scalar_class)
        assert cached is None

    def test_get_cached_computation_wrong_type(self, db, scalar_class, array_class):
        """Should return None when type doesn't match."""
        db.register(scalar_class)
        db.register(array_class)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        # Save as scalar
        result = process(10)
        var = scalar_class(result)
        var.save(db=db, subject=1)

        cache_key = result.pipeline_thunk.compute_cache_key()

        # Try to load as array - should fail
        cached = db.get_cached_computation(cache_key, array_class)
        assert cached is None


class TestCheckCacheHelper:
    """Test the check_cache helper function."""

    def test_check_cache_returns_output_thunk_on_hit(self, db, scalar_class):
        """check_cache should return OutputThunk when cached."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        # Save to populate cache
        result = process(10)
        var = scalar_class(result)
        var.save(db=db, subject=1)

        # Check cache with same inputs
        result2 = process(10)
        cached = check_cache(result2.pipeline_thunk, scalar_class, db=db)

        assert cached is not None
        assert isinstance(cached, OutputThunk)
        assert cached.data == 20
        assert cached.was_cached is True

    def test_check_cache_returns_none_on_miss(self, db, scalar_class):
        """check_cache should return None when not cached."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(10)
        cached = check_cache(result.pipeline_thunk, scalar_class, db=db)

        assert cached is None

    def test_check_cache_cached_vhash(self, db, scalar_class):
        """Cached OutputThunk should have cached_vhash set."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(10)
        var = scalar_class(result)
        original_vhash = var.save(db=db, subject=1)

        result2 = process(10)
        cached = check_cache(result2.pipeline_thunk, scalar_class, db=db)

        assert cached.cached_vhash == original_vhash


class TestOutputThunkCacheProperties:
    """Test OutputThunk cache-related properties."""

    def test_was_cached_false_by_default(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(10)
        assert result.was_cached is False

    def test_cached_vhash_none_by_default(self):
        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(10)
        assert result.cached_vhash is None

    def test_was_cached_true_when_from_cache(self, db, scalar_class):
        """OutputThunk from cache should have was_cached=True."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(10)
        scalar_class(result).save(db=db, subject=1)

        result2 = process(10)
        cached = check_cache(result2.pipeline_thunk, scalar_class, db=db)

        assert cached.was_cached is True


class TestCacheInvalidation:
    """Test cache invalidation functionality."""

    def test_invalidate_all(self, db, scalar_class):
        """invalidate_cache() should clear all cache entries."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        # Create multiple cache entries
        scalar_class(process(10)).save(db=db, subject=1)
        scalar_class(process(20)).save(db=db, subject=2)

        # Verify entries exist
        stats = db.get_cache_stats()
        assert stats["total_entries"] == 2

        # Invalidate all
        count = db.invalidate_cache()
        assert count == 2

        # Verify cleared
        stats = db.get_cache_stats()
        assert stats["total_entries"] == 0

    def test_invalidate_by_function_name(self, db, scalar_class):
        """invalidate_cache(function_name=...) should only clear that function."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def process1(x):
            return x * 2

        @thunk(n_outputs=1)
        def process2(x):
            return x * 3

        scalar_class(process1(10)).save(db=db, subject=1)
        scalar_class(process2(10)).save(db=db, subject=2)

        # Invalidate only process1
        count = db.invalidate_cache(function_name="process1")
        assert count == 1

        # process2 should still be cached
        stats = db.get_cache_stats()
        assert stats["total_entries"] == 1
        assert "process2" in stats["entries_by_function"]

    def test_invalidate_by_function_hash(self, db, scalar_class):
        """invalidate_cache(function_hash=...) should only clear that version."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(10)
        function_hash = result.pipeline_thunk.thunk.hash
        scalar_class(result).save(db=db, subject=1)

        count = db.invalidate_cache(function_hash=function_hash)
        assert count == 1

        stats = db.get_cache_stats()
        assert stats["total_entries"] == 0


class TestCacheStats:
    """Test cache statistics."""

    def test_empty_cache_stats(self, db):
        """Empty cache should return zero stats."""
        stats = db.get_cache_stats()
        assert stats["total_entries"] == 0
        assert stats["functions"] == 0
        assert stats["entries_by_function"] == {}

    def test_cache_stats_after_saves(self, db, scalar_class):
        """Stats should reflect cache contents."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def process1(x):
            return x * 2

        @thunk(n_outputs=1)
        def process2(x):
            return x * 3

        scalar_class(process1(10)).save(db=db, subject=1)
        scalar_class(process1(20)).save(db=db, subject=2)
        scalar_class(process2(10)).save(db=db, subject=3)

        stats = db.get_cache_stats()
        assert stats["total_entries"] == 3
        assert stats["functions"] == 2
        assert stats["entries_by_function"]["process1"] == 2
        assert stats["entries_by_function"]["process2"] == 1


class TestCacheWithArrays:
    """Test caching with numpy arrays."""

    def test_cache_numpy_computation(self, db, array_class):
        """Numpy computations should be cacheable."""
        db.register(array_class)

        @thunk(n_outputs=1)
        def normalize(arr):
            return arr / arr.max()

        arr = np.array([1.0, 2.0, 4.0])
        result = normalize(arr)
        array_class(result).save(db=db, subject=1)

        # Check cache
        result2 = normalize(arr)
        cached = check_cache(result2.pipeline_thunk, array_class, db=db)

        assert cached is not None
        np.testing.assert_array_almost_equal(
            cached.data, np.array([0.25, 0.5, 1.0])
        )


class TestCacheWorkflow:
    """Test complete caching workflows."""

    def test_cache_avoids_recomputation(self, db, scalar_class):
        """Demonstrate caching pattern to avoid recomputation."""
        db.register(scalar_class)

        computation_count = [0]

        @thunk(n_outputs=1)
        def expensive_compute(x):
            computation_count[0] += 1
            return x * 2

        # First run - compute and save
        result = expensive_compute(10)
        scalar_class(result).save(db=db, subject=1)
        assert computation_count[0] == 1

        # Second run - check cache first
        result2 = expensive_compute(10)
        cached = check_cache(result2.pipeline_thunk, scalar_class, db=db)

        if cached:
            # Use cached result
            assert cached.data == 20
            assert cached.was_cached is True
        else:
            # Would recompute (but we expect cache hit)
            assert False, "Expected cache hit"

        # Computation was called twice (for the thunk call), but we'd skip
        # the expensive part in a real scenario by checking cache first

    def test_cache_with_chained_computations(self, db, scalar_class):
        """Test caching in a multi-step pipeline."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 10

        # Run pipeline and save final result
        intermediate = step1(5)
        final = step2(intermediate)
        scalar_class(final).save(db=db, subject=1)

        # Check if final result is cached
        intermediate2 = step1(5)
        final2 = step2(intermediate2)
        cached = check_cache(final2.pipeline_thunk, scalar_class, db=db)

        assert cached is not None
        assert cached.data == 20  # (5*2) + 10


class TestAutomaticCacheChecking:
    """Test automatic cache checking in Thunk.__call__()."""

    def test_auto_cache_skips_execution(self, configured_db, scalar_class):
        """When cached, function should not execute."""
        configured_db.register(scalar_class)
        execution_count = [0]

        @thunk(n_outputs=1)
        def expensive_compute(x):
            execution_count[0] += 1
            return x * 2

        # First run - executes
        result1 = expensive_compute(10)
        assert execution_count[0] == 1
        assert result1.data == 20
        assert result1.was_cached is False

        # Save to populate cache
        scalar_class(result1).save(db=configured_db, subject=1)

        # Second run - should use cache, not execute
        result2 = expensive_compute(10)
        assert execution_count[0] == 1  # Still 1! Not executed again
        assert result2.data == 20
        assert result2.was_cached is True

    def test_auto_cache_returns_correct_data(self, configured_db, scalar_class):
        """Cached result should have correct data and properties."""
        configured_db.register(scalar_class)

        @thunk(n_outputs=1)
        def compute(x):
            return x * 3

        result1 = compute(7)
        vhash = scalar_class(result1).save(db=configured_db, subject=1)

        result2 = compute(7)
        assert result2.data == 21
        assert result2.was_cached is True
        assert result2.cached_vhash == vhash
        assert result2.is_complete is True

    def test_auto_cache_miss_executes_function(self, configured_db, scalar_class):
        """When not cached, function should execute normally."""
        configured_db.register(scalar_class)
        execution_count = [0]

        @thunk(n_outputs=1)
        def compute(x):
            execution_count[0] += 1
            return x + 5

        # First run - no cache, should execute
        result = compute(10)
        assert execution_count[0] == 1
        assert result.data == 15
        assert result.was_cached is False

    def test_auto_cache_different_inputs_miss(self, configured_db, scalar_class):
        """Different inputs should not hit cache."""
        configured_db.register(scalar_class)
        execution_count = [0]

        @thunk(n_outputs=1)
        def compute(x):
            execution_count[0] += 1
            return x * 2

        # First run with x=10
        result1 = compute(10)
        scalar_class(result1).save(db=configured_db, subject=1)
        assert execution_count[0] == 1

        # Second run with x=20 - different input, should execute
        result2 = compute(20)
        assert execution_count[0] == 2
        assert result2.data == 40
        assert result2.was_cached is False

    def test_auto_cache_no_database_configured(self):
        """Without database, function should execute normally."""
        # No configure_database() called
        execution_count = [0]

        @thunk(n_outputs=1)
        def compute(x):
            execution_count[0] += 1
            return x * 2

        result = compute(10)
        assert execution_count[0] == 1
        assert result.data == 20
        assert result.was_cached is False

    def test_auto_cache_with_registry_cleared(self, configured_db, scalar_class):
        """Auto-registration via global registry should enable cache hit even when db registry is cleared."""
        # Save registers the class automatically
        execution_count = [0]

        @thunk(n_outputs=1)
        def compute(x):
            execution_count[0] += 1
            return x * 2

        # First run and save
        result1 = compute(10)
        scalar_class(result1).save(db=configured_db, subject=1)
        assert execution_count[0] == 1

        # Clear the db's registered types to simulate new session
        # But class is still in global registry via __init_subclass__
        configured_db._registered_types.clear()

        # Second run - auto-registration via global registry should find the class
        result2 = compute(10)
        assert execution_count[0] == 1  # NOT executed again - cache hit!
        assert result2.was_cached is True

    def test_auto_cache_multi_output_all_saved(self, configured_db, scalar_class):
        """Multi-output functions should use cache when ALL outputs are saved."""
        configured_db.register(scalar_class)
        execution_count = [0]

        @thunk(n_outputs=2)
        def split_compute(x):
            execution_count[0] += 1
            return x, x * 2

        # First run
        a1, b1 = split_compute(10)
        assert execution_count[0] == 1

        # Save both outputs
        scalar_class(a1).save(db=configured_db, subject=1, output="a")
        scalar_class(b1).save(db=configured_db, subject=1, output="b")

        # Second run - both outputs cached, should NOT execute
        a2, b2 = split_compute(10)
        assert execution_count[0] == 1  # Still 1! Not executed again
        assert a2.was_cached is True
        assert b2.was_cached is True
        assert a2.data == 10
        assert b2.data == 20

    def test_auto_cache_multi_output_partial_save(self, configured_db, scalar_class):
        """Multi-output should execute if only some outputs are saved."""
        configured_db.register(scalar_class)
        execution_count = [0]

        @thunk(n_outputs=2)
        def split_compute(x):
            execution_count[0] += 1
            return x, x * 2

        # First run
        a1, b1 = split_compute(10)
        assert execution_count[0] == 1

        # Save only first output
        scalar_class(a1).save(db=configured_db, subject=1, output="a")
        # Don't save b1

        # Second run - not all outputs cached, should execute
        a2, b2 = split_compute(10)
        assert execution_count[0] == 2  # Executed again
        assert a2.was_cached is False
        assert b2.was_cached is False

    def test_auto_cache_with_array_input(self, configured_db, array_class):
        """Auto-cache should work with array inputs."""
        configured_db.register(array_class)
        execution_count = [0]

        @thunk(n_outputs=1)
        def normalize(arr):
            execution_count[0] += 1
            return arr / arr.max()

        arr = np.array([1.0, 2.0, 4.0])

        # First run
        result1 = normalize(arr)
        array_class(result1).save(db=configured_db, subject=1)
        assert execution_count[0] == 1

        # Second run with same array
        result2 = normalize(arr)
        assert execution_count[0] == 1  # Not executed again
        assert result2.was_cached is True
        np.testing.assert_array_equal(result2.data, np.array([0.25, 0.5, 1.0]))

    def test_auto_cache_with_chained_thunks(self, configured_db, scalar_class):
        """Auto-cache should work in chained computations."""
        configured_db.register(scalar_class)
        step1_count = [0]
        step2_count = [0]

        @thunk(n_outputs=1)
        def step1(x):
            step1_count[0] += 1
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            step2_count[0] += 1
            return x + 10

        # First run of full pipeline
        intermediate = step1(5)
        final = step2(intermediate)
        scalar_class(final).save(db=configured_db, subject=1)
        assert step1_count[0] == 1
        assert step2_count[0] == 1

        # Second run - step2 should be cached
        intermediate2 = step1(5)
        final2 = step2(intermediate2)
        assert step2_count[0] == 1  # step2 not executed again
        assert final2.was_cached is True
        assert final2.data == 20

    def test_auto_cache_preserves_pipeline_thunk(self, configured_db, scalar_class):
        """Cached OutputThunk should have valid pipeline_thunk."""
        configured_db.register(scalar_class)

        @thunk(n_outputs=1)
        def compute(x):
            return x * 2

        result1 = compute(10)
        scalar_class(result1).save(db=configured_db, subject=1)

        result2 = compute(10)
        assert result2.was_cached is True
        assert result2.pipeline_thunk is not None
        assert result2.pipeline_thunk.thunk.fcn.__name__ == "compute"
        assert result2.output_num == 0


class TestGetCachedByKey:
    """Test DatabaseManager.get_cached_by_key() method."""

    def test_returns_none_when_not_cached(self, db):
        """Should return None for unknown cache key."""
        result = db.get_cached_by_key("nonexistent_key")
        assert result is None

    def test_returns_list_with_data_and_vhash_when_cached(self, db, scalar_class):
        """Should return list of (data, vhash) tuples when cached."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def compute(x):
            return x * 2

        result = compute(10)
        vhash = scalar_class(result).save(db=db, subject=1)

        cache_key = result.pipeline_thunk.compute_cache_key()
        cached = db.get_cached_by_key(cache_key)

        assert cached is not None
        assert len(cached) == 1
        data, cached_vhash = cached[0]
        assert data == 20
        assert cached_vhash == vhash

    def test_returns_cached_via_auto_registration(self, db, scalar_class):
        """Should return cached results via auto-registration from global registry."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def compute(x):
            return x * 2

        result = compute(10)
        vhash = scalar_class(result).save(db=db, subject=1)

        # Clear db registry - but class is still in global registry via __init_subclass__
        db._registered_types.clear()

        cache_key = result.pipeline_thunk.compute_cache_key()
        cached = db.get_cached_by_key(cache_key)

        # Auto-registration via global registry should find the class and return cached result
        assert cached is not None
        assert len(cached) == 1
        assert cached[0] == (20, vhash)

    def test_returns_none_when_data_deleted(self, db, scalar_class):
        """Should return None if underlying data was deleted."""
        db.register(scalar_class)

        @thunk(n_outputs=1)
        def compute(x):
            return x * 2

        result = compute(10)
        scalar_class(result).save(db=db, subject=1)
        cache_key = result.pipeline_thunk.compute_cache_key()

        # Delete the data from the variable table
        db.connection.execute(f"DELETE FROM {scalar_class.table_name()}")
        db.connection.commit()

        cached = db.get_cached_by_key(cache_key)
        assert cached is None

    def test_multi_output_returns_all_outputs(self, db, scalar_class):
        """Should return list with all outputs for multi-output functions."""
        db.register(scalar_class)

        @thunk(n_outputs=2)
        def compute(x):
            return x, x * 2

        a, b = compute(10)
        vhash_a = scalar_class(a).save(db=db, subject=1, output="a")
        vhash_b = scalar_class(b).save(db=db, subject=1, output="b")

        cache_key = a.pipeline_thunk.compute_cache_key()
        cached = db.get_cached_by_key(cache_key, n_outputs=2)

        assert cached is not None
        assert len(cached) == 2
        assert cached[0] == (10, vhash_a)
        assert cached[1] == (20, vhash_b)

    def test_multi_output_returns_none_when_partial(self, db, scalar_class):
        """Should return None if not all outputs are cached."""
        db.register(scalar_class)

        @thunk(n_outputs=2)
        def compute(x):
            return x, x * 2

        a, b = compute(10)
        scalar_class(a).save(db=db, subject=1, output="a")
        # Don't save b

        cache_key = a.pipeline_thunk.compute_cache_key()
        cached = db.get_cached_by_key(cache_key, n_outputs=2)

        assert cached is None


class TestAutoRegistration:
    """Test that classes are auto-registered from global registry."""

    def test_global_registry_populated_on_class_definition(self):
        """Classes should be added to global registry when defined."""

        class MyTestVar(BaseVariable):
            schema_version = 1

            def to_db(self):
                import pandas as pd

                return pd.DataFrame({"value": [self.data]})

            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        assert "MyTestVar" in BaseVariable._all_subclasses
        assert BaseVariable._all_subclasses["MyTestVar"] is MyTestVar

    def test_for_type_adds_to_global_registry(self):
        """for_type() created classes should be in global registry."""

        class TypedVar(BaseVariable):
            schema_version = 1

            def to_db(self):
                import pandas as pd

                return pd.DataFrame({"value": [self.data]})

            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        SpecializedVar = TypedVar.for_type("specialized")

        assert "TypedVarSpecialized" in BaseVariable._all_subclasses
        assert BaseVariable._all_subclasses["TypedVarSpecialized"] is SpecializedVar

    def test_get_subclass_by_name(self):
        """get_subclass_by_name should find classes in global registry."""

        class LookupTestVar(BaseVariable):
            schema_version = 1

            def to_db(self):
                import pandas as pd

                return pd.DataFrame({"value": [self.data]})

            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        found = BaseVariable.get_subclass_by_name("LookupTestVar")
        assert found is LookupTestVar

        not_found = BaseVariable.get_subclass_by_name("NonExistentClass")
        assert not_found is None

    def test_auto_registration_on_cache_lookup(self, db):
        """get_cached_by_key should auto-register from global registry."""
        import pandas as pd

        # Define a class that's in global registry but NOT registered with db
        class AutoRegVar(BaseVariable):
            schema_version = 1

            def to_db(self):
                return pd.DataFrame({"value": [self.data]})

            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        # Verify it's in global registry but not db registry
        assert "AutoRegVar" in BaseVariable._all_subclasses
        assert "AutoRegVar" not in db._registered_types

        # Manually register to save (simulating a previous session)
        db.register(AutoRegVar)
        vhash = AutoRegVar(42).save(db=db, test=1)

        @thunk(n_outputs=1)
        def process(x):
            return x * 2

        result = process(42)
        AutoRegVar(result).save(db=db, test=2)
        cache_key = result.pipeline_thunk.compute_cache_key()

        # Now simulate a fresh session: remove from db registry
        del db._registered_types["AutoRegVar"]
        assert "AutoRegVar" not in db._registered_types

        # Cache lookup should auto-register and succeed
        cached = db.get_cached_by_key(cache_key, n_outputs=1)

        assert cached is not None
        assert cached[0][0] == 84  # 42 * 2
        assert "AutoRegVar" in db._registered_types  # Now registered

    def test_auto_cache_works_without_explicit_registration(self, configured_db):
        """Full auto-caching should work even without explicit registration."""
        import pandas as pd

        # Define class (goes into global registry via __init_subclass__)
        class PipelineVar(BaseVariable):
            schema_version = 1

            def to_db(self):
                return pd.DataFrame({"value": [self.data]})

            @classmethod
            def from_db(cls, df):
                return df["value"].iloc[0]

        execution_count = 0

        @thunk(n_outputs=1)
        def expensive_fn(x):
            nonlocal execution_count
            execution_count += 1
            return x * 10

        # First run: executes (no cache yet)
        result1 = expensive_fn(5)
        assert execution_count == 1
        PipelineVar(result1).save(db=configured_db, run=1)

        # Remove from db registry to simulate fresh session
        # Class is still in global registry via __init_subclass__
        del configured_db._registered_types["PipelineVar"]

        # Second run: should hit cache via auto-registration from global registry
        result2 = expensive_fn(5)

        assert result2.was_cached
        assert result2.data == 50
        assert execution_count == 1  # No additional execution
