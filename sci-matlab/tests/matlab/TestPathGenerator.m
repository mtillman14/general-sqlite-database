classdef TestPathGenerator < matlab.unittest.TestCase
%TESTPATHGENERATOR  Integration tests for scidb.PathGenerator.

    methods (TestClassSetup)
        function addPaths(~)
            this_dir = fileparts(mfilename('fullpath'));
            run(fullfile(this_dir, 'setup_paths.m'));
        end
    end

    methods (Test)
        function test_single_key(testCase)
            pg = scidb.PathGenerator("{subject}/data.mat", ...
                'root_folder', '/data', ...
                'subject', [1 2 3]);
            testCase.verifyEqual(length(pg), 3);
        end

        function test_cartesian_product(testCase)
            pg = scidb.PathGenerator("{subject}/session_{session}.mat", ...
                'root_folder', '/data', ...
                'subject', [1 2], ...
                'session', ["A", "B", "C"]);
            testCase.verifyEqual(length(pg), 6);  % 2 * 3
        end

        function test_path_template_resolution(testCase)
            pg = scidb.PathGenerator("{subject}/data.mat", ...
                'root_folder', '/data', ...
                'subject', [1 2 3]);
            [path, ~] = pg.get(1);
            expected = string(fullfile('/data', '1', 'data.mat'));
            testCase.verifyEqual(path, expected);
        end

        function test_numeric_metadata_in_path(testCase)
            pg = scidb.PathGenerator("sub{subject}/trial{trial}.mat", ...
                'root_folder', '/experiment', ...
                'subject', [1 2], ...
                'trial', [0 1]);

            [path1, meta1] = pg.get(1);
            testCase.verifyTrue(contains(path1, "sub1"));
            testCase.verifyTrue(contains(path1, "trial0"));
            testCase.verifyEqual(meta1.subject, 1);
            testCase.verifyEqual(meta1.trial, 0);
        end

        function test_string_metadata_in_path(testCase)
            pg = scidb.PathGenerator("{group}/data.mat", ...
                'root_folder', '/data', ...
                'group', ["control", "treatment"]);

            [path1, meta1] = pg.get(1);
            testCase.verifyTrue(contains(path1, "control"));
            testCase.verifyEqual(string(meta1.group), "control");
        end

        function test_get_returns_correct_metadata(testCase)
            pg = scidb.PathGenerator("{subject}/data.mat", ...
                'root_folder', '/data', ...
                'subject', [10 20 30]);

            [~, meta1] = pg.get(1);
            [~, meta2] = pg.get(2);
            [~, meta3] = pg.get(3);

            testCase.verifyEqual(meta1.subject, 10);
            testCase.verifyEqual(meta2.subject, 20);
            testCase.verifyEqual(meta3.subject, 30);
        end

        function test_get_out_of_range_errors(testCase)
            pg = scidb.PathGenerator("{x}/data.mat", ...
                'root_folder', '/data', ...
                'x', [1 2]);

            testCase.verifyError(@() pg.get(0), 'scidb:PathGenerator');
            testCase.verifyError(@() pg.get(3), 'scidb:PathGenerator');
        end

        function test_to_table(testCase)
            pg = scidb.PathGenerator("{subject}/data.mat", ...
                'root_folder', '/data', ...
                'subject', [1 2 3]);

            t = pg.to_table();
            testCase.verifyClass(t, 'table');
            testCase.verifyEqual(height(t), 3);
            testCase.verifyTrue(ismember('path', t.Properties.VariableNames));
            testCase.verifyTrue(ismember('subject', t.Properties.VariableNames));
        end

        function test_to_table_values(testCase)
            pg = scidb.PathGenerator("{subject}/data.mat", ...
                'root_folder', '/data', ...
                'subject', [1 2]);

            t = pg.to_table();
            testCase.verifyEqual(t.subject(1), 1);
            testCase.verifyEqual(t.subject(2), 2);
        end

        function test_paths_property(testCase)
            pg = scidb.PathGenerator("{x}.mat", ...
                'root_folder', '/data', ...
                'x', [1 2 3]);

            testCase.verifyEqual(numel(pg.paths), 3);
            testCase.verifyTrue(isstring(pg.paths));
        end

        function test_metadata_property(testCase)
            pg = scidb.PathGenerator("{x}.mat", ...
                'root_folder', '/data', ...
                'x', [1 2]);

            testCase.verifyEqual(numel(pg.metadata), 2);
            testCase.verifyTrue(isstruct(pg.metadata));
            testCase.verifyEqual(pg.metadata(1).x, 1);
            testCase.verifyEqual(pg.metadata(2).x, 2);
        end

        function test_no_root_folder_uses_pwd(testCase)
            pg = scidb.PathGenerator("{x}/data.mat", 'x', [1]);
            [path, ~] = pg.get(1);
            expected = string(fullfile(pwd, '1', 'data.mat'));
            testCase.verifyEqual(path, expected);
        end

        function test_two_keys_all_combinations(testCase)
            pg = scidb.PathGenerator("{a}_{b}.mat", ...
                'root_folder', '/out', ...
                'a', [1 2], ...
                'b', [10 20 30]);

            testCase.verifyEqual(length(pg), 6);

            % Collect all metadata combinations
            seen = zeros(length(pg), 2);
            for i = 1:length(pg)
                [~, m] = pg.get(i);
                seen(i, :) = [m.a, m.b];
            end

            % All 6 combinations should be present
            testCase.verifyEqual(size(unique(seen, 'rows'), 1), 6);
        end
    end
end
