classdef TestForEachSchemaFiltering < matlab.unittest.TestCase
%TESTFOREACHSCHEMAFILTERING  Tests for filtering cartesian product to
%   existing schema combinations when [] is used in for_each.

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

        function test_filtering_removes_nonexistent_combos(testCase)
            % Save data only for (1,A) and (2,B) — not (1,B) or (2,A)
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            RawSignal().save([4 5 6], 'subject', 2, 'session', 'B');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', [], ...
                'session', []);

            % Only 2 of 4 possible combos should have been processed
            all_results = ProcessedSignal().load_all();
            testCase.verifyEqual(numel(all_results), 2);
        end

        function test_no_filtering_when_all_explicit(testCase)
            % Save data only for (1,A) — but provide explicit lists
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', [1 2], ...
                'session', ["A", "B"]);

            % Explicit values => no filtering; 3 of 4 combos are skipped
            % (only (1,A) has data), but all 4 are attempted
            all_results = ProcessedSignal().load_all();
            testCase.verifyEqual(numel(all_results), 1);
        end

        function test_no_filtering_with_pathinput(testCase)
            % With PathInput, filtering should be bypassed even with []
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');

            % PathInput resolves templates — no DB filtering should happen
            scidb.for_each(@path_length, ...
                struct('filepath', scidb.PathInput("{subject}/data.mat", ...
                    'root_folder', '/data')), ...
                {ScalarVar()}, ...
                'subject', [], ...
                'session', []);

            % Subject 1, session A exist in DB. With PathInput present,
            % filtering is skipped so all combos are attempted.
            % (subject=1, session=A) succeeds; no others exist but all
            % are tried because PathInput bypasses filtering.
            all_results = ScalarVar().load_all();
            testCase.verifyGreaterThanOrEqual(numel(all_results), 1);
        end

        function test_no_filtering_with_fixed_pathinput(testCase)
            % Fixed(PathInput) should also bypass filtering
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');

            scidb.for_each(@path_length, ...
                struct('filepath', scidb.Fixed( ...
                    scidb.PathInput("{subject}/data.mat", ...
                        'root_folder', '/data'), ...
                    'session', 'A')), ...
                {ScalarVar()}, ...
                'subject', [], ...
                'session', []);

            all_results = ScalarVar().load_all();
            testCase.verifyGreaterThanOrEqual(numel(all_results), 1);
        end

        function test_mixed_resolved_and_explicit(testCase)
            % One key resolved via [], one explicit — filtering applies
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            RawSignal().save([4 5 6], 'subject', 2, 'session', 'A');
            RawSignal().save([7 8 9], 'subject', 3, 'session', 'B');

            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', [], ...
                'session', ["A", "B"]);

            % Existing combos: (1,A), (2,A), (3,B). Full product would be
            % 3 subjects * 2 sessions = 6. After filtering: 3.
            all_results = ProcessedSignal().load_all();
            testCase.verifyEqual(numel(all_results), 3);
        end

        function test_info_message_printed(testCase)
            % Verify [info] message is printed when combos are removed
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            RawSignal().save([4 5 6], 'subject', 2, 'session', 'B');

            output = evalc( ...
                'scidb.for_each(@double_values, struct(''x'', RawSignal()), {ProcessedSignal()}, ''subject'', [], ''session'', [])');
            testCase.verifySubstring(output, '[info] filtered');
            testCase.verifySubstring(output, 'from 4 to 2');
        end

        function test_no_info_message_when_nothing_filtered(testCase)
            % All combos exist — no [info] message
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            RawSignal().save([4 5 6], 'subject', 1, 'session', 'B');
            RawSignal().save([7 8 9], 'subject', 2, 'session', 'A');
            RawSignal().save([10 11 12], 'subject', 2, 'session', 'B');

            output = evalc( ...
                'scidb.for_each(@double_values, struct(''x'', RawSignal()), {ProcessedSignal()}, ''subject'', [], ''session'', [])');
            testCase.verifyTrue(~contains(output, '[info] filtered'));
            all_results = ProcessedSignal().load_all();
            testCase.verifyEqual(numel(all_results), 4);
        end

        function test_integer_to_string_coercion(testCase)
            % Integer subject values should match string DB values
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            RawSignal().save([4 5 6], 'subject', 2, 'session', 'B');

            % subject=[] resolves to numeric [1, 2] from DB, but
            % _schema stores VARCHAR "1", "2". Coercion must handle this.
            scidb.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', [], ...
                'session', []);

            all_results = ProcessedSignal().load_all();
            testCase.verifyEqual(numel(all_results), 2);
        end

        function test_dry_run_reflects_filtered_count(testCase)
            % dry_run should show the filtered iteration count
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            RawSignal().save([4 5 6], 'subject', 2, 'session', 'B');

            output = evalc( ...
                'scidb.for_each(@double_values, struct(''x'', RawSignal()), {ProcessedSignal()}, ''dry_run'', true, ''subject'', [], ''session'', [])');
            % Should say "2 iterations" not "4 iterations"
            testCase.verifySubstring(output, '2 iterations');
        end

    end
end
