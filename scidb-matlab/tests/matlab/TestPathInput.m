classdef TestPathInput < matlab.unittest.TestCase
%TESTPATHINPUT  Integration tests for scidb.PathInput.

    methods (TestClassSetup)
        function addPaths(~)
            this_dir = fileparts(mfilename('fullpath'));
            run(fullfile(this_dir, 'setup_paths.m'));
        end
    end

    methods (Test)
        function test_basic_resolution(testCase)
            pi = scidb.PathInput("{subject}/data.mat", ...
                'root_folder', '/data');
            path = pi.load('subject', 1);
            expected = string(fullfile('/data', '1', 'data.mat'));
            testCase.verifyEqual(path, expected);
        end

        function test_multiple_placeholders(testCase)
            pi = scidb.PathInput("{subject}/session_{session}/trial.mat", ...
                'root_folder', '/experiment');
            path = pi.load('subject', 1, 'session', 'A');
            expected = string(fullfile('/experiment', '1', 'session_A', 'trial.mat'));
            testCase.verifyEqual(path, expected);
        end

        function test_numeric_value_in_template(testCase)
            pi = scidb.PathInput("sub{subject}_trial{trial}.mat", ...
                'root_folder', '/data');
            path = pi.load('subject', 3, 'trial', 7);
            testCase.verifyTrue(contains(path, "sub3"));
            testCase.verifyTrue(contains(path, "trial7"));
        end

        function test_string_value_in_template(testCase)
            pi = scidb.PathInput("{group}/results.csv", ...
                'root_folder', '/output');
            path = pi.load('group', 'control');
            expected = string(fullfile('/output', 'control', 'results.csv'));
            testCase.verifyEqual(path, expected);
        end

        function test_no_root_folder_uses_pwd(testCase)
            pi = scidb.PathInput("{x}/data.mat");
            path = pi.load('x', 1);
            expected = string(fullfile(pwd, '1', 'data.mat'));
            testCase.verifyEqual(path, expected);
        end

        function test_returns_string(testCase)
            pi = scidb.PathInput("{x}.mat", 'root_folder', '/data');
            path = pi.load('x', 1);
            testCase.verifyClass(path, 'string');
        end

        function test_unused_metadata_ignored(testCase)
            % Extra metadata keys not in template should not cause errors
            pi = scidb.PathInput("{subject}/data.mat", ...
                'root_folder', '/data');
            path = pi.load('subject', 1, 'session', 'A');
            expected = string(fullfile('/data', '1', 'data.mat'));
            testCase.verifyEqual(path, expected);
        end

        function test_absolute_path_in_template(testCase)
            pi = scidb.PathInput("{subject}/data.mat", ...
                'root_folder', '/absolute/root');
            path = pi.load('subject', 5);
            % Verify the path contains the root folder and resolved template
            testCase.verifyTrue(contains(path, "absolute"));
            testCase.verifyTrue(contains(path, "root"));
            testCase.verifyTrue(contains(path, "5"));
            testCase.verifyTrue(contains(path, "data.mat"));
        end
    end
end
