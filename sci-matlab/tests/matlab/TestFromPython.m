classdef TestFromPython < matlab.unittest.TestCase
%TESTFROMPYTHON  Unit tests for scidb.internal.from_python.
%
%   Verifies that Python proxy objects (py.int, py.str, py.float, py.list)
%   are properly converted to native MATLAB types, and that native MATLAB
%   types that were auto-converted by the Python bridge are returned
%   directly without hitting the expensive fallback path.

    methods (TestClassSetup)
        function addPaths(~)
            this_dir = fileparts(mfilename('fullpath'));
            run(fullfile(this_dir, 'setup_paths.m'));
        end
    end

    methods (Test)

        % --- Python proxy objects must be converted, not returned as-is ---

        function test_py_int_converts_to_double(testCase)
            result = scidb.internal.from_python(py.int(42));
            testCase.verifyClass(result, 'double');
            testCase.verifyEqual(result, 42);
        end

        function test_py_float_converts_to_double(testCase)
            result = scidb.internal.from_python(py.float(3.14));
            testCase.verifyClass(result, 'double');
            testCase.verifyEqual(result, 3.14, 'AbsTol', 1e-12);
        end

        function test_py_str_converts_to_string(testCase)
            result = scidb.internal.from_python(py.str('hello'));
            testCase.verifyClass(result, 'string');
            testCase.verifyEqual(result, "hello");
        end

        function test_py_bool_true_converts_to_logical(testCase)
            result = scidb.internal.from_python(py.bool(true));
            testCase.verifyClass(result, 'logical');
            testCase.verifyTrue(result);
        end

        function test_py_bool_false_converts_to_logical(testCase)
            result = scidb.internal.from_python(py.bool(false));
            testCase.verifyClass(result, 'logical');
            testCase.verifyFalse(result);
        end

        function test_py_none_converts_to_empty(testCase)
            result = scidb.internal.from_python(py.None);
            testCase.verifyEmpty(result);
        end

        % --- Native MATLAB types (auto-converted by bridge) pass through ---

        function test_native_double_passthrough(testCase)
            result = scidb.internal.from_python(42.5);
            testCase.verifyClass(result, 'double');
            testCase.verifyEqual(result, 42.5);
        end

        function test_native_logical_passthrough(testCase)
            result = scidb.internal.from_python(true);
            testCase.verifyClass(result, 'logical');
            testCase.verifyTrue(result);
        end

        function test_native_string_passthrough(testCase)
            result = scidb.internal.from_python("hello");
            testCase.verifyClass(result, 'string');
            testCase.verifyEqual(result, "hello");
        end

        % --- Lists of numeric values ---

        function test_py_list_of_floats(testCase)
            py_list = py.list({1.0, 2.0, 3.0});
            result = scidb.internal.from_python(py_list);
            testCase.verifyClass(result, 'double');
            testCase.verifyEqual(result(:)', [1 2 3], 'AbsTol', 1e-12);
        end

        function test_py_list_of_ints(testCase)
            py_list = py.list({py.int(10), py.int(20), py.int(30)});
            result = scidb.internal.from_python(py_list);
            testCase.verifyTrue(isnumeric(result));
            testCase.verifyEqual(double(result(:)'), [10 20 30]);
        end

        function test_py_list_of_strings(testCase)
            py_list = py.list({"alpha", "beta", "gamma"});
            result = scidb.internal.from_python(py_list);
            testCase.verifyTrue(isstring(result));
            testCase.verifyEqual(result, ["alpha", "beta", "gamma"]);
        end

        % --- Numpy arrays ---

        function test_numpy_1d_float(testCase)
            py_arr = py.numpy.array({1.0, 2.0, 3.0});
            result = scidb.internal.from_python(py_arr);
            testCase.verifyClass(result, 'double');
            testCase.verifyEqual(result, [1; 2; 3]);
        end

        function test_numpy_bool(testCase)
            py_arr = py.numpy.array({true, false, true});
            result = scidb.internal.from_python(py_arr);
            testCase.verifyClass(result, 'logical');
            testCase.verifyEqual(result, [true; false; true]);
        end

        % --- Dict ---

        function test_py_dict_converts_to_struct(testCase)
            py_dict = py.dict(pyargs('a', py.int(1), 'b', 'hello'));
            result = scidb.internal.from_python(py_dict);
            testCase.verifyClass(result, 'struct');
            testCase.verifyEqual(result.a, 1);
            testCase.verifyEqual(result.b, "hello");
        end

    end
end
