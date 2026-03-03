classdef TestThunk < matlab.unittest.TestCase
%TESTTHUNK  Integration tests for scidb.Thunk execution and caching.

    properties
        test_dir
    end

    methods (TestClassSetup)
        function addPaths(~)
            this_dir = fileparts(mfilename('fullpath'));
            run(fullfile(this_dir, 'setup_paths.m'));
        end
    end

    methods (TestMethodSetup)
        function setupDatabase(testCase)
            testCase.test_dir = tempname;
            mkdir(testCase.test_dir);
            scidb.configure_database( ...
                fullfile(testCase.test_dir, 'test.duckdb'), ...
                ["subject", "session"]);
        end
    end

    methods (TestMethodTeardown)
        function cleanup(testCase)
            try
                scidb.get_database().close();
            catch
            end
            if isfolder(testCase.test_dir)
                rmdir(testCase.test_dir, 's');
            end
        end
    end

    methods (Test)
        % --- Basic Thunk creation ---

        function test_create_thunk(testCase)
            thunk = scidb.Thunk(@double_values);
            testCase.verifyClass(thunk, 'scidb.Thunk');
        end

        function test_create_thunk_with_unpack_output(testCase)
            thunk = scidb.Thunk(@split_data, 'unpack_output', true);
            testCase.verifyClass(thunk, 'scidb.Thunk');
        end

        function test_anonymous_function_errors(testCase)
            testCase.verifyError( ...
                @() scidb.Thunk(@(x) x + 1), ...
                'scidb:AnonymousFunction');
        end

        % --- Thunk execution ---

        function test_thunk_returns_thunk_output(testCase)
            thunk = scidb.Thunk(@double_values);
            result = thunk([1 2 3]);
            testCase.verifyClass(result, 'scidb.ThunkOutput');
        end

        function test_thunk_computes_correct_result(testCase)
            thunk = scidb.Thunk(@double_values);
            result = thunk([1 2 3]);
            testCase.verifyEqual(result.data, [2 4 6], 'AbsTol', 1e-10);
        end

        function test_thunk_with_constant_argument(testCase)
            thunk = scidb.Thunk(@add_offset);
            result = thunk([10 20 30], 5);
            testCase.verifyEqual(result.data, [15 25 35], 'AbsTol', 1e-10);
        end

        function test_thunk_with_matrix_input(testCase)
            thunk = scidb.Thunk(@double_values);
            data = [1 2; 3 4; 5 6];
            result = thunk(data);
            testCase.verifyEqual(result.data, data * 2, 'AbsTol', 1e-10);
            testCase.verifyEqual(size(result.data), [3, 2]);
        end

        function test_thunk_with_scalar_input(testCase)
            thunk = scidb.Thunk(@double_values);
            result = thunk(21);
            testCase.verifyEqual(result.data, 42, 'AbsTol', 1e-10);
        end

        function test_thunk_output_has_py_obj(testCase)
            thunk = scidb.Thunk(@double_values);
            result = thunk([1 2 3]);
            testCase.verifyNotEmpty(result.py_obj);
        end

        % --- Thunk with loaded inputs ---

        function test_thunk_with_loaded_variable(testCase)
            % Save raw data, load it, pass to thunk
            RawSignal().save([10 20 30], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@double_values);
            result = thunk(raw);

            testCase.verifyEqual(result.data, [20 40 60]', 'AbsTol', 1e-10);
        end

        function test_thunk_with_two_loaded_variables(testCase)
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            ProcessedSignal().save([10 20 30]', 'subject', 1, 'session', 'A');

            raw = RawSignal().load('subject', 1, 'session', 'A');
            proc = ProcessedSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@sum_inputs);
            result = thunk(raw, proc);

            testCase.verifyEqual(result.data, [11 22 33]', 'AbsTol', 1e-10);
        end

        function test_thunk_with_mixed_inputs(testCase)
            % One loaded variable + one constant
            RawSignal().save([10 20 30], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@add_offset);
            result = thunk(raw, 5);

            testCase.verifyEqual(result.data, [15 25 35]', 'AbsTol', 1e-10);
        end

        % --- Chained thunks ---

        function test_chained_thunks(testCase)
            thunk1 = scidb.Thunk(@double_values);
            thunk2 = scidb.Thunk(@triple_values);

            result1 = thunk1([1 2 3]);
            result2 = thunk2(result1);

            testCase.verifyEqual(result1.data, [2 4 6], 'AbsTol', 1e-10);
            testCase.verifyEqual(result2.data, [6 12 18], 'AbsTol', 1e-10);
        end

        function test_chained_thunks_preserve_lineage(testCase)
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk1 = scidb.Thunk(@double_values);
            thunk2 = scidb.Thunk(@triple_values);

            step1 = thunk1(raw);
            step2 = thunk2(step1);

            % Save final result and verify lineage is tracked
            FilteredSignal().save(step2, 'subject', 1, 'session', 'A');
            loaded = FilteredSignal().load('subject', 1, 'session', 'A');
            testCase.verifyTrue(strlength(loaded.lineage_hash) > 0);

            % Provenance should reference the last function in the chain
            prov = FilteredSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyEqual(char(prov.function_name), 'triple_values');
        end

        % --- Unpack output (multi-output) ---

        function test_unpack_output_returns_multiple(testCase)
            thunk = scidb.Thunk(@split_data, 'unpack_output', true);
            [first, second] = thunk([1 2 3 4 5 6]);
            testCase.verifyClass(first, 'scidb.ThunkOutput');
            testCase.verifyClass(second, 'scidb.ThunkOutput');
            testCase.verifyEqual(first.data, [1 2 3], 'AbsTol', 1e-10);
            testCase.verifyEqual(second.data, [4 5 6], 'AbsTol', 1e-10);
        end

        function test_unpack_output_save_separately(testCase)
            thunk = scidb.Thunk(@split_data, 'unpack_output', true);
            [first, second] = thunk([10 20 30 40]);

            SplitFirst().save(first, 'subject', 1, 'session', 'A');
            SplitSecond().save(second, 'subject', 1, 'session', 'A');

            r1 = SplitFirst().load('subject', 1, 'session', 'A');
            r2 = SplitSecond().load('subject', 1, 'session', 'A');

            testCase.verifyEqual(r1.data, [10 20]', 'AbsTol', 1e-10);
            testCase.verifyEqual(r2.data, [30 40]', 'AbsTol', 1e-10);
        end

        function test_unpack_output_different_lineage_hashes(testCase)
            thunk = scidb.Thunk(@split_data, 'unpack_output', true);
            [first, second] = thunk([1 2 3 4]');

            SplitFirst().save(first, 'subject', 1, 'session', 'A');
            SplitSecond().save(second, 'subject', 1, 'session', 'A');

            r1 = SplitFirst().load('subject', 1, 'session', 'A');
            r2 = SplitSecond().load('subject', 1, 'session', 'A');

            % Different output_nums should produce different lineage hashes
            testCase.verifyNotEqual(r1.lineage_hash, r2.lineage_hash);
        end

        % --- Caching ---

        function test_cache_hit_returns_same_data(testCase)
            thunk = scidb.Thunk(@double_values);

            % First call: cache miss, function executes
            result1 = thunk([1 2 3]);
            ProcessedSignal().save(result1, 'subject', 1, 'session', 'A');

            % Second call with same args: should hit cache
            result2 = thunk([1 2 3]);
            testCase.verifyEqual(result2.data, [2 4 6]', 'AbsTol', 1e-10);
        end

        function test_cache_miss_with_different_inputs(testCase)
            thunk = scidb.Thunk(@double_values);

            result1 = thunk([1 2 3]);
            ProcessedSignal().save(result1, 'subject', 1, 'session', 'A');

            % Different input should not hit cache
            result2 = thunk([10 20 30]);
            testCase.verifyEqual(result2.data, [20 40 60], 'AbsTol', 1e-10);
        end

        function test_cache_miss_with_different_function(testCase)
            thunk1 = scidb.Thunk(@double_values);
            thunk2 = scidb.Thunk(@triple_values);

            result1 = thunk1([1 2 3]);
            ProcessedSignal().save(result1, 'subject', 1, 'session', 'A');

            % Different function with same input should not hit cache
            result2 = thunk2([1 2 3]);
            testCase.verifyEqual(result2.data, [3 6 9], 'AbsTol', 1e-10);
        end

        % --- Save thunk output and verify lineage ---

        function test_save_thunk_output_stores_lineage_hash(testCase)
            thunk = scidb.Thunk(@double_values);
            result = thunk([1 2 3]);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            loaded = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyTrue(strlength(loaded.lineage_hash) > 0);
        end

        function test_thunk_output_as_input_tracks_lineage(testCase)
            % Save raw, load, thunk, save processed
            RawSignal().save([5 10 15], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@double_values);
            result = thunk(raw);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            % Verify provenance references the input
            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyFalse(isempty(prov));
            testCase.verifyEqual(char(prov.function_name), 'double_values');
            testCase.verifyTrue(numel(prov.inputs) >= 1);
        end
    end
end
