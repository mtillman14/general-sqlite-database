classdef TestForEach < matlab.unittest.TestCase
%TESTFOREACH  Integration tests for scidb.for_each batch processing.

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
                ["subject", "session"], ...
                fullfile(testCase.test_dir, 'pipeline.db'));
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
        % --- Basic iteration ---

        function test_single_key_iteration(testCase)
            % Save input data for 3 subjects
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            RawSignal().save([4 5 6], 'subject', 2, 'session', 'A');
            RawSignal().save([7 8 9], 'subject', 3, 'session', 'A');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', [1 2 3], ...
                'session', "A");

            % Verify all 3 outputs were saved
            for s = [1 2 3]
                result = ProcessedSignal().load('subject', s, 'session', 'A');
                raw = RawSignal().load('subject', s, 'session', 'A');
                testCase.verifyEqual(result.data, raw.data * 2, 'AbsTol', 1e-10);
            end
        end

        function test_cartesian_product(testCase)
            % Save input data for all combinations
            for s = [1 2]
                for sess = ["A", "B"]
                    RawSignal().save(s * [1 2 3], ...
                        'subject', s, 'session', sess);
                end
            end

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', [1 2], ...
                'session', ["A", "B"]);

            % Should produce 2 * 2 = 4 outputs
            all_results = ProcessedSignal().load_all();
            testCase.verifyEqual(numel(all_results), 4);
        end

        function test_output_data_correct(testCase)
            RawSignal().save([10 20 30], 'subject', 1, 'session', 'A');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', 1, ...
                'session', "A");

            result = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [20 40 60], 'AbsTol', 1e-10);
        end

        % --- Constants ---

        function test_constant_input_passed_to_function(testCase)
            RawSignal().save([10 20 30], 'subject', 1, 'session', 'A');

            scidb.for_each(@add_offset, ...
                struct('x', RawSignal(), 'offset', 5), ...
                {ProcessedSignal()}, ...
                'subject', 1, ...
                'session', "A");

            result = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [15 25 35], 'AbsTol', 1e-10);
        end

        function test_constant_included_in_save_metadata(testCase)
            RawSignal().save([10 20 30], 'subject', 1, 'session', 'A');

            scidb.for_each(@add_offset, ...
                struct('x', RawSignal(), 'offset', 5), ...
                {ProcessedSignal()}, ...
                'subject', 1, ...
                'session', "A");

            % The constant 'offset' should appear in saved metadata
            versions = ProcessedSignal().list_versions( ...
                'subject', 1, 'session', 'A');
            testCase.verifyGreaterThanOrEqual(numel(versions), 1);
        end

        % --- Two loadable inputs ---

        function test_two_variable_inputs(testCase)
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            ProcessedSignal().save([10 20 30], 'subject', 1, 'session', 'A');

            scidb.for_each(@sum_inputs, ...
                struct('a', RawSignal(), 'b', ProcessedSignal()), ...
                {FilteredSignal()}, ...
                'subject', 1, ...
                'session', "A");

            result = FilteredSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [11 22 33], 'AbsTol', 1e-10);
        end

        % --- Fixed inputs ---

        function test_fixed_input_overrides_metadata(testCase)
            % Save baseline data with session="BL"
            BaselineSignal().save([100 200 300], ...
                'subject', 1, 'session', 'BL');
            BaselineSignal().save([100 200 300], ...
                'subject', 2, 'session', 'BL');

            % Save current data with session="A" and "B"
            RawSignal().save([110 210 310], 'subject', 1, 'session', 'A');
            RawSignal().save([120 220 320], 'subject', 2, 'session', 'A');

            % Use Fixed to always load baseline from session="BL"
            scidb.for_each(@subtract_baseline, ...
                struct('current', RawSignal(), ...
                       'baseline', scidb.Fixed(BaselineSignal(), ...
                           'session', 'BL')), ...
                {DeltaSignal()}, ...
                'subject', [1 2], ...
                'session', "A");

            % Subject 1: [110-100, 210-200, 310-300] = [10, 10, 10]
            d1 = DeltaSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(d1.data, [10 10 10], 'AbsTol', 1e-10);

            % Subject 2: [120-100, 220-200, 320-300] = [20, 20, 20]
            d2 = DeltaSignal().load('subject', 2, 'session', 'A');
            testCase.verifyEqual(d2.data, [20 20 20], 'AbsTol', 1e-10);
        end

        % --- dry_run ---

        function test_dry_run_does_not_save(testCase)
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'dry_run', true, ...
                'subject', 1, ...
                'session', "A");

            % Nothing should be saved
            results = ProcessedSignal().load_all('subject', 1, 'session', 'A');
            testCase.verifyEmpty(results);
        end

        % --- save=false ---

        function test_save_false_executes_but_does_not_save(testCase)
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'save', false, ...
                'subject', 1, ...
                'session', "A");

            results = ProcessedSignal().load_all('subject', 1, 'session', 'A');
            testCase.verifyEmpty(results);
        end

        % --- Thunk as function ---

        function test_with_thunk_function(testCase)
            RawSignal().save([5 10 15], 'subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@double_values);
            scidb.for_each(thunk, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', 1, ...
                'session', "A");

            result = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [10 20 30], 'AbsTol', 1e-10);

            % Thunk output should have lineage
            testCase.verifyTrue(strlength(result.lineage_hash) > 0);
        end

        function test_with_thunk_and_constant(testCase)
            RawSignal().save([5 10 15], 'subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@add_offset);
            scidb.for_each(thunk, ...
                struct('x', RawSignal(), 'offset', 100), ...
                {ProcessedSignal()}, ...
                'subject', 1, ...
                'session', "A");

            result = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [105 110 115], 'AbsTol', 1e-10);
        end

        % --- Multiple outputs ---

        function test_multiple_outputs_with_plain_function(testCase)
            RawSignal().save([1 2 3 4], 'subject', 1, 'session', 'A');

            scidb.for_each(@split_data, ...
                struct('x', RawSignal()), ...
                {SplitFirst(), SplitSecond()}, ...
                'subject', 1, ...
                'session', "A");

            r1 = SplitFirst().load('subject', 1, 'session', 'A');
            r2 = SplitSecond().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(r1.data, [1 2], 'AbsTol', 1e-10);
            testCase.verifyEqual(r2.data, [3 4], 'AbsTol', 1e-10);
        end

        function test_multiple_outputs_with_thunk(testCase)
            RawSignal().save([10 20 30 40], 'subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@split_data, 'unpack_output', true);
            scidb.for_each(thunk, ...
                struct('x', RawSignal()), ...
                {SplitFirst(), SplitSecond()}, ...
                'subject', 1, ...
                'session', "A");

            r1 = SplitFirst().load('subject', 1, 'session', 'A');
            r2 = SplitSecond().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(r1.data, [10 20], 'AbsTol', 1e-10);
            testCase.verifyEqual(r2.data, [30 40], 'AbsTol', 1e-10);
        end

        % --- PathInput ---

        function test_path_input_resolves_template(testCase)
            % PathInput resolves template to file path; function receives path string
            scidb.for_each(@path_length, ...
                struct('filepath', scidb.PathInput("{subject}/data.mat", ...
                    'root_folder', '/data')), ...
                {ScalarVar()}, ...
                'subject', 1, ...
                'session', "A");

            % The function should have received a resolved path
            result = ScalarVar().load('subject', 1, 'session', 'A');
            expected_path = fullfile('/data', '1', 'data.mat');
            testCase.verifyEqual(result.data, double(strlength(expected_path)), ...
                'AbsTol', 1e-10);
        end

        % --- Skipped iterations ---

        function test_missing_input_skips_iteration(testCase)
            % Only save data for subject 1, not subject 2
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', [1 2], ...
                'session', "A");

            % Subject 1 should be saved
            r1 = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(r1.data, [2 4 6], 'AbsTol', 1e-10);

            % Subject 2 should be skipped (no input data)
            results = ProcessedSignal().load_all('subject', 2, 'session', 'A');
            testCase.verifyEmpty(results);
        end

        % --- Parallel mode ---

        function test_parallel_basic(testCase)
            % Same as test_single_key_iteration but with parallel=true
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            RawSignal().save([4 5 6], 'subject', 2, 'session', 'A');
            RawSignal().save([7 8 9], 'subject', 3, 'session', 'A');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'parallel', true, ...
                'subject', [1 2 3], ...
                'session', "A");

            for s = [1 2 3]
                result = ProcessedSignal().load('subject', s, 'session', 'A');
                raw = RawSignal().load('subject', s, 'session', 'A');
                testCase.verifyEqual(result.data, raw.data * 2, 'AbsTol', 1e-10);
            end
        end

        function test_parallel_cartesian(testCase)
            % Same as test_cartesian_product but with parallel=true
            for s = [1 2]
                for sess = ["A", "B"]
                    RawSignal().save(s * [1 2 3], ...
                        'subject', s, 'session', sess);
                end
            end

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'parallel', true, ...
                'subject', [1 2], ...
                'session', ["A", "B"]);

            all_results = ProcessedSignal().load_all();
            testCase.verifyEqual(numel(all_results), 4);
        end

        function test_parallel_with_constant(testCase)
            % Same as test_constant_input_passed_to_function with parallel=true
            RawSignal().save([10 20 30], 'subject', 1, 'session', 'A');

            scidb.for_each(@add_offset, ...
                struct('x', RawSignal(), 'offset', 5), ...
                {ProcessedSignal()}, ...
                'parallel', true, ...
                'subject', 1, ...
                'session', "A");

            result = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [15 25 35], 'AbsTol', 1e-10);
        end

        function test_parallel_skips_missing(testCase)
            % Same as test_missing_input_skips_iteration with parallel=true
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'parallel', true, ...
                'subject', [1 2], ...
                'session', "A");

            r1 = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(r1.data, [2 4 6], 'AbsTol', 1e-10);

            results = ProcessedSignal().load_all('subject', 2, 'session', 'A');
            testCase.verifyEmpty(results);
        end

        function test_parallel_false_is_default(testCase)
            % parallel=false should behave identically to the default
            RawSignal().save([10 20 30], 'subject', 1, 'session', 'A');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'parallel', false, ...
                'subject', 1, ...
                'session', "A");

            result = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [20 40 60], 'AbsTol', 1e-10);
        end

        function test_parallel_multiple_outputs(testCase)
            % Parallel mode with multiple outputs
            RawSignal().save([1 2 3 4], 'subject', 1, 'session', 'A');

            scidb.for_each(@split_data, ...
                struct('x', RawSignal()), ...
                {SplitFirst(), SplitSecond()}, ...
                'parallel', true, ...
                'subject', 1, ...
                'session', "A");

            r1 = SplitFirst().load('subject', 1, 'session', 'A');
            r2 = SplitSecond().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(r1.data, [1 2], 'AbsTol', 1e-10);
            testCase.verifyEqual(r2.data, [3 4], 'AbsTol', 1e-10);
        end

        function test_parallel_thunk_errors(testCase)
            % Thunks are not supported in parallel mode
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@double_values);
            testCase.verifyError(@() ...
                scidb.for_each(thunk, ...
                    struct('x', RawSignal()), ...
                    {ProcessedSignal()}, ...
                    'parallel', true, ...
                    'subject', 1, ...
                    'session', "A"), ...
                'scidb:for_each');
        end

        % --- Multiple subjects and sessions ---

        function test_full_pipeline(testCase)
            % Populate raw data for a 2x2 grid
            subjects = [1 2];
            sessions = ["A", "B"];
            for s = subjects
                for sess = sessions
                    RawSignal().save(s * [1 2 3], ...
                        'subject', s, 'session', sess);
                end
            end

            % Process all combinations
            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', subjects, ...
                'session', sessions);

            % Verify each output
            for s = subjects
                for sess = sessions
                    result = ProcessedSignal().load('subject', s, 'session', sess);
                    testCase.verifyEqual(result.data, s * [2 4 6], 'AbsTol', 1e-10);
                end
            end
        end
    end
end
