"""Tests for scidb.thunk module."""

import numpy as np
import pytest

from scidb.thunk import OutputThunk, PipelineThunk, Thunk, thunk


class TestThunkBasics:
    """Test basic Thunk functionality."""

    def test_thunk_decorator_creates_thunk(self):
        @thunk(n_outputs=1)
        def my_func(x):
            return x * 2

        assert isinstance(my_func, Thunk)

    def test_thunk_has_correct_n_outputs(self):
        @thunk(n_outputs=3)
        def multi_output(x):
            return x, x * 2, x * 3

        assert multi_output.n_outputs == 3

    def test_thunk_has_hash(self):
        @thunk(n_outputs=1)
        def my_func(x):
            return x * 2

        assert isinstance(my_func.hash, str)
        assert len(my_func.hash) == 64  # SHA-256 hex

    def test_same_function_same_hash(self):
        @thunk(n_outputs=1)
        def func1(x):
            return x * 2

        # Note: This creates a new Thunk, so hash will be same if bytecode same
        # But we can't easily test this without redefining the exact same function

    def test_different_functions_different_hash(self):
        @thunk(n_outputs=1)
        def func1(x):
            return x * 2

        @thunk(n_outputs=1)
        def func2(x):
            return x * 3

        assert func1.hash != func2.hash

    def test_thunk_preserves_function_name(self):
        @thunk(n_outputs=1)
        def my_named_function(x):
            return x

        assert my_named_function.fcn.__name__ == "my_named_function"

    def test_thunk_repr(self):
        @thunk(n_outputs=1)
        def my_func(x):
            return x

        repr_str = repr(my_func)
        assert "Thunk" in repr_str
        assert "my_func" in repr_str


class TestThunkExecution:
    """Test Thunk execution and OutputThunk creation."""

    def test_thunk_call_returns_output_thunk(self):
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        result = double(5)
        assert isinstance(result, OutputThunk)

    def test_output_thunk_has_value(self):
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        result = double(5)
        assert result.value == 10

    def test_output_thunk_is_complete(self):
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        result = double(5)
        assert result.is_complete is True

    def test_multi_output_returns_tuple(self):
        @thunk(n_outputs=3)
        def multi(x):
            return x, x * 2, x * 3

        result = multi(5)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(r, OutputThunk) for r in result)

    def test_multi_output_values(self):
        @thunk(n_outputs=3)
        def multi(x):
            return x, x * 2, x * 3

        a, b, c = multi(5)
        assert a.value == 5
        assert b.value == 10
        assert c.value == 15

    def test_thunk_with_kwargs(self):
        @thunk(n_outputs=1)
        def add(a, b=10):
            return a + b

        result = add(5, b=20)
        assert result.value == 25


class TestPipelineThunk:
    """Test PipelineThunk functionality."""

    def test_pipeline_thunk_captures_inputs(self):
        @thunk(n_outputs=1)
        def process(data, factor):
            return data * factor

        result = process(10, 2)
        pt = result.pipeline_thunk

        assert "arg_0" in pt.inputs
        assert "arg_1" in pt.inputs
        assert pt.inputs["arg_0"] == 10
        assert pt.inputs["arg_1"] == 2

    def test_pipeline_thunk_captures_kwargs(self):
        @thunk(n_outputs=1)
        def process(data, factor=1):
            return data * factor

        result = process(10, factor=3)
        pt = result.pipeline_thunk

        assert pt.inputs["factor"] == 3

    def test_pipeline_thunk_has_hash(self):
        @thunk(n_outputs=1)
        def process(data):
            return data * 2

        result = process(10)
        assert isinstance(result.pipeline_thunk.hash, str)
        assert len(result.pipeline_thunk.hash) == 64

    def test_same_inputs_same_pipeline_hash(self):
        @thunk(n_outputs=1)
        def process(data):
            return data * 2

        result1 = process(10)
        result2 = process(10)

        # Same inputs should result in same pipeline thunk being reused
        assert result1.pipeline_thunk is result2.pipeline_thunk

    def test_different_inputs_different_pipeline_hash(self):
        @thunk(n_outputs=1)
        def process(data):
            return data * 2

        result1 = process(10)
        result2 = process(20)

        # Different inputs should have different pipeline thunks
        assert result1.pipeline_thunk.hash != result2.pipeline_thunk.hash


class TestOutputThunk:
    """Test OutputThunk functionality."""

    def test_output_thunk_has_pipeline_thunk_reference(self):
        @thunk(n_outputs=1)
        def process(data):
            return data * 2

        result = process(10)
        assert isinstance(result.pipeline_thunk, PipelineThunk)

    def test_output_thunk_output_num(self):
        @thunk(n_outputs=3)
        def multi(x):
            return x, x * 2, x * 3

        a, b, c = multi(5)
        assert a.output_num == 0
        assert b.output_num == 1
        assert c.output_num == 2

    def test_output_thunk_hash(self):
        @thunk(n_outputs=1)
        def process(data):
            return data * 2

        result = process(10)
        assert isinstance(result.hash, str)
        assert len(result.hash) == 64

    def test_output_thunk_equality_by_hash(self):
        @thunk(n_outputs=1)
        def process(data):
            return data * 2

        result1 = process(10)
        result2 = process(10)

        # Same computation should have same hash
        assert result1 == result2
        assert result1.hash == result2.hash

    def test_output_thunk_equality_with_value(self):
        @thunk(n_outputs=1)
        def process(data):
            return data * 2

        result = process(5)
        assert result == 10  # Compares with value

    def test_output_thunk_str(self):
        @thunk(n_outputs=1)
        def process(data):
            return data * 2

        result = process(5)
        assert str(result) == "10"

    def test_output_thunk_repr(self):
        @thunk(n_outputs=1)
        def my_func(data):
            return data * 2

        result = my_func(5)
        repr_str = repr(result)
        assert "OutputThunk" in repr_str
        assert "my_func" in repr_str


class TestThunkChaining:
    """Test chaining multiple thunks together."""

    def test_chain_two_thunks(self):
        @thunk(n_outputs=1)
        def double(x):
            return x * 2

        @thunk(n_outputs=1)
        def add_one(x):
            return x + 1

        intermediate = double(5)  # OutputThunk with value 10
        result = add_one(intermediate)  # Should unwrap and use 10

        assert result.value == 11

    def test_chain_preserves_lineage(self):
        @thunk(n_outputs=1)
        def step1(x):
            return x * 2

        @thunk(n_outputs=1)
        def step2(x):
            return x + 1

        intermediate = step1(5)
        result = step2(intermediate)

        # Result's pipeline thunk should have the intermediate as input
        pt = result.pipeline_thunk
        assert "arg_0" in pt.inputs
        assert isinstance(pt.inputs["arg_0"], OutputThunk)
        assert pt.inputs["arg_0"].value == 10

    def test_long_chain(self):
        @thunk(n_outputs=1)
        def add(x, y):
            return x + y

        @thunk(n_outputs=1)
        def multiply(x, y):
            return x * y

        @thunk(n_outputs=1)
        def square(x):
            return x * x

        a = add(2, 3)  # 5
        b = multiply(a, 2)  # 10
        c = square(b)  # 100

        assert c.value == 100


class TestThunkWithNumpy:
    """Test thunks with numpy arrays."""

    def test_numpy_array_input(self):
        @thunk(n_outputs=1)
        def normalize(arr):
            return arr / arr.max()

        arr = np.array([1.0, 2.0, 4.0])
        result = normalize(arr)

        expected = np.array([0.25, 0.5, 1.0])
        np.testing.assert_array_almost_equal(result.value, expected)

    def test_numpy_multiple_array_inputs(self):
        @thunk(n_outputs=1)
        def add_arrays(a, b):
            return a + b

        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5, 6])
        result = add_arrays(arr1, arr2)

        np.testing.assert_array_equal(result.value, np.array([5, 7, 9]))

    def test_numpy_chain(self):
        @thunk(n_outputs=1)
        def double(arr):
            return arr * 2

        @thunk(n_outputs=1)
        def sum_array(arr):
            return arr.sum()

        arr = np.array([1, 2, 3])
        doubled = double(arr)
        total = sum_array(doubled)

        assert total.value == 12


class TestThunkErrorHandling:
    """Test error handling in thunks."""

    def test_wrong_number_of_outputs_raises(self):
        @thunk(n_outputs=2)
        def single_output(x):
            return x * 2  # Returns 1 output, but declared 2

        with pytest.raises(ValueError, match="returned 1 outputs, but 2 were expected"):
            single_output(5)

    def test_exception_in_function_propagates(self):
        @thunk(n_outputs=1)
        def raises_error(x):
            raise RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Test error"):
            raises_error(5)
