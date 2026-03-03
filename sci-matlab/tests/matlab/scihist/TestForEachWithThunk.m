classdef TestForEachWithThunk < matlab.unittest.TestCase
%TESTFOREACHWITHTHUNK  Tests for scihist.for_each with Thunk lineage tracking.

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
            scihist.configure_database( ...
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
            try
                py.thunk.Thunk.query = py.None;
            catch
            end
            if isfolder(testCase.test_dir)
                rmdir(testCase.test_dir, 's');
            end
        end
    end

    methods (Test)

        function test_with_thunk_function(testCase)
            % scihist.for_each auto-wraps or accepts a Thunk; output has lineage
            RawSignal().save([5 10 15], 'subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@double_values);
            scihist.for_each(thunk, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', 1, ...
                'session', "A");

            result = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [10 20 30]', 'AbsTol', 1e-10);

            % Thunk output should have lineage hash
            testCase.verifyTrue(strlength(result.lineage_hash) > 0);
        end

        function test_with_plain_function_records_lineage(testCase)
            % scihist.for_each auto-wraps a plain function in Thunk
            RawSignal().save([5 10 15], 'subject', 1, 'session', 'A');

            scihist.for_each(@double_values, ...
                struct('x', RawSignal()), ...
                {ProcessedSignal()}, ...
                'subject', 1, ...
                'session', "A");

            result = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [10 20 30]', 'AbsTol', 1e-10);

            % Auto-wrapped plain function should also produce lineage
            testCase.verifyTrue(strlength(result.lineage_hash) > 0);
        end

        function test_with_thunk_and_constant(testCase)
            % Thunk with constant input tracked in lineage
            RawSignal().save([5 10 15], 'subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@add_offset);
            scihist.for_each(thunk, ...
                struct('x', RawSignal(), 'offset', 100), ...
                {ProcessedSignal()}, ...
                'subject', 1, ...
                'session', "A");

            result = ProcessedSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(result.data, [105 110 115]', 'AbsTol', 1e-10);
        end

        function test_multiple_outputs_with_thunk(testCase)
            % unpack_output=true with scihist.for_each
            RawSignal().save([10 20 30 40], 'subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@split_data, 'unpack_output', true);
            scihist.for_each(thunk, ...
                struct('x', RawSignal()), ...
                {SplitFirst(), SplitSecond()}, ...
                'subject', 1, ...
                'session', "A");

            r1 = SplitFirst().load('subject', 1, 'session', 'A');
            r2 = SplitSecond().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(r1.data, [10 20]', 'AbsTol', 1e-10);
            testCase.verifyEqual(r2.data, [30 40]', 'AbsTol', 1e-10);
        end

        function test_cache_hit_in_for_each(testCase)
            % Pre-processing with same inputs produces cache hit in for_each
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');

            % Process manually (creates cache entry)
            raw = RawSignal().load('subject', 1, 'session', 'A');
            thunk = scidb.Thunk(@double_values);
            result = thunk(raw);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            % Process again via for_each with same thunk (should hit cache)
            scihist.for_each(thunk, ...
                struct('x', RawSignal()), ...
                {FilteredSignal()}, ...
                'subject', 1, ...
                'session', "A");

            % Both outputs should have the same data
            proc = ProcessedSignal().load('subject', 1, 'session', 'A');
            filt = FilteredSignal().load('subject', 1, 'session', 'A');
            testCase.verifyEqual(proc.data, filt.data, 'AbsTol', 1e-10);
        end

    end
end
