"""Tests for scidb public API exports."""

import pytest


class TestPublicAPIExports:
    """Test that all expected symbols are exported from scidb."""

    def test_import_scidb(self):
        import scidb
        assert scidb is not None

    def test_version_exists(self):
        import scidb
        assert hasattr(scidb, "__version__")
        assert scidb.__version__ == "0.1.0"

    def test_database_manager_exported(self):
        from scidb import DatabaseManager
        assert DatabaseManager is not None

    def test_base_variable_exported(self):
        from scidb import BaseVariable
        assert BaseVariable is not None

    def test_configure_database_exported(self):
        from scidb import configure_database
        assert callable(configure_database)

    def test_get_database_exported(self):
        from scidb import get_database
        assert callable(get_database)

    def test_scidb_error_exported(self):
        from scidb import SciDBError
        assert issubclass(SciDBError, Exception)

    def test_not_registered_error_exported(self):
        from scidb import NotRegisteredError
        assert issubclass(NotRegisteredError, Exception)

    def test_not_found_error_exported(self):
        from scidb import NotFoundError
        assert issubclass(NotFoundError, Exception)

    def test_database_not_configured_error_exported(self):
        from scidb import DatabaseNotConfiguredError
        assert issubclass(DatabaseNotConfiguredError, Exception)

    def test_reserved_metadata_key_error_exported(self):
        from scidb import ReservedMetadataKeyError
        assert issubclass(ReservedMetadataKeyError, Exception)

    def test_thunk_decorator_exported(self):
        from scidb import thunk
        assert callable(thunk)

    def test_thunk_class_exported(self):
        from scidb import Thunk
        assert Thunk is not None

    def test_pipeline_thunk_exported(self):
        from scidb import PipelineThunk
        assert PipelineThunk is not None

    def test_thunk_output_exported(self):
        from scidb import ThunkOutput
        assert ThunkOutput is not None

    def test_lineage_record_exported(self):
        from scidb import LineageRecord
        assert LineageRecord is not None

    def test_extract_lineage_exported(self):
        from scidb import extract_lineage
        assert callable(extract_lineage)

    def test_get_raw_value_exported(self):
        from scidb import get_raw_value
        assert callable(get_raw_value)

    def test_check_cache_exported(self):
        from scidb import check_cache
        assert callable(check_cache)


class TestAllExports:
    """Test __all__ contains expected exports."""

    def test_all_contains_database_manager(self):
        import scidb
        assert "DatabaseManager" in scidb.__all__

    def test_all_contains_base_variable(self):
        import scidb
        assert "BaseVariable" in scidb.__all__

    def test_all_contains_configure_database(self):
        import scidb
        assert "configure_database" in scidb.__all__

    def test_all_contains_get_database(self):
        import scidb
        assert "get_database" in scidb.__all__

    def test_all_contains_exceptions(self):
        import scidb
        assert "SciDBError" in scidb.__all__
        assert "NotRegisteredError" in scidb.__all__
        assert "NotFoundError" in scidb.__all__
        assert "DatabaseNotConfiguredError" in scidb.__all__
        assert "ReservedMetadataKeyError" in scidb.__all__

    def test_all_contains_thunk_exports(self):
        import scidb
        assert "thunk" in scidb.__all__
        assert "Thunk" in scidb.__all__
        assert "PipelineThunk" in scidb.__all__
        assert "ThunkOutput" in scidb.__all__

    def test_all_contains_lineage_exports(self):
        import scidb
        assert "LineageRecord" in scidb.__all__
        assert "extract_lineage" in scidb.__all__
        assert "get_raw_value" in scidb.__all__
        assert "check_cache" in scidb.__all__

    def test_all_exports_are_accessible(self):
        """Verify all items in __all__ are actually accessible."""
        import scidb
        for name in scidb.__all__:
            assert hasattr(scidb, name), f"{name} in __all__ but not accessible"


class TestModuleDocstring:
    """Test module has proper documentation."""

    def test_module_has_docstring(self):
        import scidb
        assert scidb.__doc__ is not None
        assert len(scidb.__doc__) > 0

    def test_docstring_mentions_scidb(self):
        import scidb
        assert "SciDB" in scidb.__doc__
