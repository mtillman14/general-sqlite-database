function setup_paths()
%SETUP_PATHS  Add MATLAB and Python paths for scihist integration tests.
%
%   Delegates to the parent setup_paths, then also adds scihist-lib/src
%   to the Python path.

    % Run the parent setup_paths to add common paths
    parent_dir = fullfile(fileparts(mfilename('fullpath')), '..');
    run(fullfile(parent_dir, 'setup_paths.m'));
end
