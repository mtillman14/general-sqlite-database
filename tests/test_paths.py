"""Tests for PathGenerator class."""

from pathlib import Path

import pytest

from scidb import PathGenerator


class TestPathGenerator:
    """Tests for the PathGenerator class."""

    def test_basic_generation(self):
        """Test basic path generation with two metadata fields."""
        paths = PathGenerator(
            "data/{subject}/trial_{trial}.mat",
            subject=range(2),
            trial=range(3),
        )

        result = paths.to_list()

        assert len(result) == 6  # 2 subjects x 3 trials

        # Check first item - path should be absolute
        path, meta = result[0]
        assert path.is_absolute()
        assert path.name == "trial_0.mat"
        assert meta == {"subject": 0, "trial": 0}

        # Check last item
        path, meta = result[5]
        assert path.is_absolute()
        assert path.name == "trial_2.mat"
        assert meta == {"subject": 1, "trial": 2}

    def test_with_root_folder(self):
        """Test path generation with root_folder."""
        paths = PathGenerator(
            "{subject}/trial_{trial}.mat",
            root_folder="/data/experiment",
            subject=range(2),
            trial=range(2),
        )

        result = paths.to_list()

        path, meta = result[0]
        assert path == Path("/data/experiment/0/trial_0.mat")
        assert meta == {"subject": 0, "trial": 0}

        path, meta = result[3]
        assert path == Path("/data/experiment/1/trial_1.mat")
        assert meta == {"subject": 1, "trial": 1}

    def test_root_folder_as_path(self):
        """Test that root_folder accepts Path objects."""
        paths = PathGenerator(
            "{x}.txt",
            root_folder=Path("/my/root"),
            x=[1, 2],
        )

        path, meta = paths[0]
        assert path == Path("/my/root/1.txt")

    def test_paths_are_always_absolute(self):
        """Test that paths are always absolute, even without root_folder."""
        paths = PathGenerator(
            "data/{x}.txt",
            x=[1],
        )

        path, meta = paths[0]
        assert path.is_absolute()
        assert isinstance(path, Path)

    def test_single_metadata_field(self):
        """Test with a single metadata field."""
        paths = PathGenerator(
            "{subject}.csv",
            root_folder="/data",
            subject=[1, 2, 3],
        )

        result = paths.to_list()

        assert len(result) == 3
        assert result[0] == (Path("/data/1.csv"), {"subject": 1})
        assert result[1] == (Path("/data/2.csv"), {"subject": 2})
        assert result[2] == (Path("/data/3.csv"), {"subject": 3})

    def test_string_metadata_values(self):
        """Test with string metadata values."""
        paths = PathGenerator(
            "{condition}/{session}.mat",
            root_folder="/data",
            condition=["control", "treatment"],
            session=["pre", "post"],
        )

        result = paths.to_list()

        assert len(result) == 4

        path, meta = result[0]
        assert path == Path("/data/control/pre.mat")
        assert meta == {"condition": "control", "session": "pre"}

        path, meta = result[3]
        assert path == Path("/data/treatment/post.mat")
        assert meta == {"condition": "treatment", "session": "post"}

    def test_mixed_metadata_types(self):
        """Test with mixed metadata types (int, str)."""
        paths = PathGenerator(
            "{group}/subject_{subject}.csv",
            root_folder="/data",
            group=["A", "B"],
            subject=range(2),
        )

        result = paths.to_list()

        assert len(result) == 4
        path, meta = result[0]
        assert path == Path("/data/A/subject_0.csv")
        assert meta["group"] == "A"
        assert meta["subject"] == 0

    def test_iteration_with_unpacking(self):
        """Test iterating with tuple unpacking."""
        paths = PathGenerator(
            "{x}.txt",
            root_folder="/data",
            x=[1, 2, 3],
        )

        collected = []
        for path, metadata in paths:
            collected.append((path, metadata["x"]))

        assert collected == [
            (Path("/data/1.txt"), 1),
            (Path("/data/2.txt"), 2),
            (Path("/data/3.txt"), 3),
        ]

    def test_len(self):
        """Test len() on PathGenerator."""
        paths = PathGenerator(
            "data/{a}/{b}/{c}.txt",
            a=range(2),
            b=range(3),
            c=range(4),
        )

        assert len(paths) == 2 * 3 * 4  # 24

    def test_indexing(self):
        """Test indexing PathGenerator."""
        paths = PathGenerator(
            "{x}.txt",
            root_folder="/data",
            x=[10, 20, 30],
        )

        path, meta = paths[0]
        assert meta["x"] == 10

        path, meta = paths[1]
        assert meta["x"] == 20

        path, meta = paths[2]
        assert meta["x"] == 30

        path, meta = paths[-1]
        assert meta["x"] == 30

    def test_empty_metadata(self):
        """Test with empty metadata (no combinations)."""
        paths = PathGenerator(
            "fixed_path.txt",
            root_folder="/data",
        )

        result = paths.to_list()

        # With no metadata, should produce exactly one path
        assert len(result) == 1
        path, meta = result[0]
        assert path == Path("/data/fixed_path.txt")
        assert meta == {}

    def test_repr(self):
        """Test string representation."""
        paths = PathGenerator(
            "data/{subject}/{trial}.mat",
            subject=range(3),
            trial=range(5),
        )

        repr_str = repr(paths)
        assert "PathGenerator" in repr_str
        assert "data/{subject}/{trial}.mat" in repr_str
        assert "subject" in repr_str
        assert "trial" in repr_str

    def test_repr_with_root_folder(self):
        """Test string representation includes root_folder."""
        paths = PathGenerator(
            "{x}.txt",
            root_folder="/data",
            x=[1],
        )

        repr_str = repr(paths)
        assert "root_folder" in repr_str
        assert "/data" in repr_str

    def test_order_preservation(self):
        """Test that iteration order matches Cartesian product order."""
        paths = PathGenerator(
            "{a}/{b}.txt",
            root_folder="/data",
            a=[1, 2],
            b=["x", "y", "z"],
        )

        result = paths.to_list()

        # First metadata field varies slowest (outer loop)
        expected_order = [
            (1, "x"), (1, "y"), (1, "z"),
            (2, "x"), (2, "y"), (2, "z"),
        ]

        for i, (expected_a, expected_b) in enumerate(expected_order):
            path, meta = result[i]
            assert meta["a"] == expected_a
            assert meta["b"] == expected_b

    def test_complex_path_template(self):
        """Test with a more complex path template."""
        paths = PathGenerator(
            "experiment_{exp}/subject_{subject:03d}/session_{session}/recording.bin",
            root_folder="/home/user/data",
            exp=["exp1", "exp2"],
            subject=[1, 2],
            session=["morning", "evening"],
        )

        result = paths.to_list()

        assert len(result) == 2 * 2 * 2  # 8

        # Check that formatting works correctly
        path, meta = result[0]
        assert path == Path("/home/user/data/experiment_exp1/subject_001/session_morning/recording.bin")
        assert meta["exp"] == "exp1"
        assert meta["subject"] == 1
        assert meta["session"] == "morning"

    def test_use_in_for_loop(self):
        """Test typical usage in a for loop with tuple unpacking."""
        paths = PathGenerator(
            "{subject}/trial_{trial}.mat",
            root_folder="/data",
            subject=range(2),
            trial=range(2),
        )

        collected = []
        for path, metadata in paths:
            collected.append((path, metadata["subject"], metadata["trial"]))

        assert collected == [
            (Path("/data/0/trial_0.mat"), 0, 0),
            (Path("/data/0/trial_1.mat"), 0, 1),
            (Path("/data/1/trial_0.mat"), 1, 0),
            (Path("/data/1/trial_1.mat"), 1, 1),
        ]
