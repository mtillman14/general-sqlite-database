classdef TestSaveLoadWithThunk < matlab.unittest.TestCase
%TESTSAVELOADWITHTHUNK  Tests for save/load behavior with Thunk lineage.

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

        function test_save_thunk_output_preserves_lineage(testCase)
            % Save raw data, load it, pass through a thunk, save result
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@double_values);
            result = thunk(raw);

            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');
            loaded = ProcessedSignal().load('subject', 1, 'session', 'A');

            testCase.verifyEqual(loaded.data, [2 4 6]', 'AbsTol', 1e-10);
            testCase.verifyTrue(strlength(loaded.lineage_hash) > 0);
        end

        function test_thunk_output_provenance_accessible(testCase)
            % After saving a ThunkOutput, provenance should be queryable
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@double_values);
            result = thunk(raw);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyFalse(isempty(prov));
            testCase.verifyEqual(char(prov.function_name), 'double_values');
        end

    end
end
