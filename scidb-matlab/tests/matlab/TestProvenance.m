classdef TestProvenance < matlab.unittest.TestCase
%TESTPROVENANCE  Integration tests for provenance/lineage tracking.

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
        function test_raw_data_no_provenance(testCase)
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            prov = RawSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyEmpty(prov);
        end

        function test_thunk_output_has_provenance(testCase)
            thunk = scidb.Thunk(@double_values);
            result = thunk([1 2 3]);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyFalse(isempty(prov));
            testCase.verifyTrue(isstruct(prov));
        end

        function test_provenance_function_name(testCase)
            thunk = scidb.Thunk(@double_values);
            result = thunk([1 2 3]);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyEqual(char(prov.function_name), 'double_values');
        end

        function test_provenance_function_hash(testCase)
            thunk = scidb.Thunk(@double_values);
            result = thunk([1 2 3]);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyTrue(strlength(prov.function_hash) == 64);
        end

        function test_provenance_has_required_fields(testCase)
            thunk = scidb.Thunk(@double_values);
            result = thunk([1 2 3]);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyTrue(isfield(prov, 'function_name'));
            testCase.verifyTrue(isfield(prov, 'function_hash'));
            testCase.verifyTrue(isfield(prov, 'inputs'));
            testCase.verifyTrue(isfield(prov, 'constants'));
        end

        function test_provenance_constants_from_constant_args(testCase)
            thunk = scidb.Thunk(@add_offset);
            result = thunk([1 2 3], 10);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            % Two args: [1 2 3] (constant array) and 10 (constant scalar)
            testCase.verifyEqual(numel(prov.constants), 2);
        end

        function test_provenance_inputs_from_loaded_variable(testCase)
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@add_offset);
            result = thunk(raw, 5);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyEqual(numel(prov.inputs), 1);
            testCase.verifyEqual(numel(prov.constants), 1);
        end

        function test_provenance_chained_thunks(testCase)
            % Chain: raw -> double_values -> triple_values
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk1 = scidb.Thunk(@double_values);
            step1 = thunk1(raw);
            ProcessedSignal().save(step1, 'subject', 1, 'session', 'A');

            proc = ProcessedSignal().load('subject', 1, 'session', 'A');
            thunk2 = scidb.Thunk(@triple_values);
            step2 = thunk2(proc);
            FilteredSignal().save(step2, 'subject', 1, 'session', 'A');

            % Provenance of final result references triple_values
            prov = FilteredSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyEqual(char(prov.function_name), 'triple_values');
            testCase.verifyEqual(numel(prov.inputs), 1);

            % The input should reference the upstream thunk
            input_info = prov.inputs{1};
            testCase.verifyTrue(isstruct(input_info));
        end

        function test_provenance_two_variable_inputs(testCase)
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            ProcessedSignal().save([10 20 30], 'subject', 1, 'session', 'A');

            raw = RawSignal().load('subject', 1, 'session', 'A');
            proc = ProcessedSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@sum_inputs);
            result = thunk(raw, proc);
            FilteredSignal().save(result, 'subject', 1, 'session', 'A');

            prov = FilteredSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyEqual(char(prov.function_name), 'sum_inputs');
            testCase.verifyEqual(numel(prov.inputs), 2);
            testCase.verifyEmpty(prov.constants);
        end

        function test_different_functions_different_hashes(testCase)
            thunk1 = scidb.Thunk(@double_values);
            result1 = thunk1([1 2 3]);
            ProcessedSignal().save(result1, 'subject', 1, 'session', 'A');

            thunk2 = scidb.Thunk(@triple_values);
            result2 = thunk2([1 2 3]);
            FilteredSignal().save(result2, 'subject', 1, 'session', 'A');

            prov1 = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            prov2 = FilteredSignal().provenance('subject', 1, 'session', 'A');

            testCase.verifyNotEqual(prov1.function_hash, prov2.function_hash);
        end

        function test_lineage_hash_deterministic(testCase)
            % Same computation twice should produce the same lineage hash
            thunk = scidb.Thunk(@double_values);

            result1 = thunk([1 2 3]);
            ProcessedSignal().save(result1, 'subject', 1, 'session', 'A');

            result2 = thunk([1 2 3]);
            ProcessedSignal().save(result2, 'subject', 1, 'session', 'B');

            r1 = ProcessedSignal().load('subject', 1, 'session', 'A');
            r2 = ProcessedSignal().load('subject', 1, 'session', 'B');

            testCase.verifyEqual(r1.lineage_hash, r2.lineage_hash);
        end

        function test_lineage_hash_changes_with_inputs(testCase)
            thunk = scidb.Thunk(@double_values);

            result1 = thunk([1 2 3]);
            ProcessedSignal().save(result1, 'subject', 1, 'session', 'A');

            result2 = thunk([4 5 6]);
            ProcessedSignal().save(result2, 'subject', 1, 'session', 'B');

            r1 = ProcessedSignal().load('subject', 1, 'session', 'A');
            r2 = ProcessedSignal().load('subject', 1, 'session', 'B');

            testCase.verifyNotEqual(r1.lineage_hash, r2.lineage_hash);
        end

        function test_provenance_constants_have_name_field(testCase)
            % Constants in provenance carry a 'name' field (arg_0, arg_1, ...)
            % and a 'value_repr' field — the new JSON column design preserves both.
            RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@add_offset);
            result = thunk(raw, 10);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            % raw -> input, 10 -> constant
            testCase.verifyEqual(numel(prov.constants), 1);

            const_info = prov.constants{1};
            testCase.verifyTrue(isstruct(const_info));
            testCase.verifyTrue(isfield(const_info, 'name'));
            testCase.verifyFalse(isempty(const_info.name));
            testCase.verifyTrue(isfield(const_info, 'value_repr'));
        end

        function test_provenance_inputs_have_record_id(testCase)
            % Variable inputs in provenance carry the source record_id,
            % allowing exact tracing back to the specific saved variable used.
            record_id = RawSignal().save([1 2 3], 'subject', 1, 'session', 'A');
            raw = RawSignal().load('subject', 1, 'session', 'A');

            thunk = scidb.Thunk(@add_offset);
            result = thunk(raw, 10);
            ProcessedSignal().save(result, 'subject', 1, 'session', 'A');

            prov = ProcessedSignal().provenance('subject', 1, 'session', 'A');
            testCase.verifyEqual(numel(prov.inputs), 1);

            input_info = prov.inputs{1};
            testCase.verifyTrue(isstruct(input_info));
            testCase.verifyTrue(isfield(input_info, 'record_id'));
            testCase.verifyEqual(string(input_info.record_id), string(record_id));
        end
    end
end
