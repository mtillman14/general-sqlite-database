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
