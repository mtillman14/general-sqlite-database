"""Tests for PathGenerator."""

from pathlib import Path

import pytest

from scipathgen import PathGenerator


class TestPathGenerator:
    """Tests for the PathGenerator class."""

    def test_basic_generation(self):
        """Test basic path generation with simple metadata."""
        paths = PathGenerator(
            "{subject}/trial_{trial}.mat",
            subject=[1, 2],
            trial=[1, 2, 3],
        )

        assert len(paths) == 6

        # Check first and last items
        path, meta = paths[0]
        assert meta == {"subject": 1, "trial": 1}
        assert "1/trial_1.mat" in str(path)

        path, meta = paths[-1]
        assert meta == {"subject": 2, "trial": 3}
        assert "2/trial_3.mat" in str(path)

    def test_with_root_folder(self, tmp_path):
        """Test path generation with root folder."""
        paths = PathGenerator(
            "{name}.txt",
            root_folder=tmp_path,
            name=["a", "b"],
        )

        assert len(paths) == 2

        path, meta = paths[0]
        assert path == tmp_path / "a.txt"
        assert meta == {"name": "a"}

    def test_iteration(self):
        """Test that PathGenerator is iterable."""
        paths = PathGenerator("{x}.txt", x=[1, 2, 3])

        results = list(paths)
        assert len(results) == 3

        for path, meta in paths:
            assert isinstance(path, Path)
            assert isinstance(meta, dict)

    def test_to_list(self):
        """Test to_list method."""
        paths = PathGenerator("{x}.txt", x=[1, 2])

        result = paths.to_list()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_repr(self):
        """Test string representation."""
        paths = PathGenerator("{a}/{b}.txt", a=[1], b=[2])

        repr_str = repr(paths)
        assert "PathGenerator" in repr_str
        assert "{a}/{b}.txt" in repr_str

    def test_empty_metadata(self):
        """Test with no metadata produces single path."""
        paths = PathGenerator("fixed.txt")

        assert len(paths) == 1
        path, meta = paths[0]
        assert "fixed.txt" in str(path)
        assert meta == {}

    def test_single_value_metadata(self):
        """Test with single-value iterables."""
        paths = PathGenerator("{x}.txt", x=[42])

        assert len(paths) == 1
        path, meta = paths[0]
        assert meta == {"x": 42}
