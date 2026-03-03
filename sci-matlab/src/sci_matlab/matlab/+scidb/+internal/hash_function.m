function hash = hash_function(fcn)
%HASH_FUNCTION  Compute a SHA-256 hash identifying a MATLAB function.
%
%   hash = scidb.internal.hash_function(@my_function)
%
%   For named functions, hashes the full source code of the .m file.
%   Anonymous functions are not supported (error).
%
%   Returns a 64-character hex string.

    info = functions(fcn);

    if strcmp(info.type, 'anonymous')
        error('scidb:AnonymousFunction', ...
            ['Anonymous functions are not supported by scidb.Thunk. ' ...
             'Use a named function defined in an .m file instead.']);
    end

    % Resolve the source file path
    func_name = func2str(fcn);
    fpath = which(func_name);

    if isempty(fpath)
        error('scidb:FunctionNotFound', ...
            'Cannot locate source file for function "%s".', func_name);
    end

    % Read source and hash via Python's hashlib
    source = fileread(fpath);
    hash = char(py.hashlib.sha256(py.bytes(source, 'utf-8')).hexdigest());
end
