import numpy as np                                                                                                                                                                                  
                                                                                                                                                                                                     
# Simulate the problem:                                                                                                                                                                           
# When MATLAB table with Nx2 matrix is saved, the to_python.m code converts
# each row to a 1D array (cell array of row vectors)

# Example: MATLAB table with Nx2 matrix column
# Original: 28864 x 2 matrix
# After to_python.m conversion: cell array with 28864 elements, each is 1x2 array

# When this becomes a DataFrame column, each cell element is a 1D array
# So when saving as dict_of_arrays, the column is treated as a single 1D array of 28864 elements
# But it's actually a column of 28864 row vectors

# The metadata stores shape=(28864, 2) from original MATLAB
# But the actual stored data is flattened to 34330 elements (28864 * 1.19... or similar)

# When loading, it tries to reshape 34330 into (1, 28864) which fails

# Let's calculate what 34330 really is:
size1 = 28864
size2 = 1 + 2  # 1D array of 28864 elements, but each contains [x, y]

# The issue: when you have a cell array of row vectors in a DataFrame column,
# it becomes object dtype column with nested arrays, not a simple 1D numeric column

# When saving dict_of_arrays, the code treats this as a 1D column
# But the ndarray_keys metadata has shape=(28864, 2) from original MATLAB

print('Problem analysis:')
print('MATLAB original: 28864 x 2 matrix in table column')
print('to_python.m converts to: cell array of 28864 elements, each 1x2 array')
print('DataFrame column becomes: object dtype with nested arrays')
print('dict_of_arrays path treats as: single 1D column of 28864 elements')
print('Stored shape metadata: [28864, 2]')
print('Actual stored rows: 28864 (one per original row)')
print('When loading, tries to reshape 28864 rows into (1, 28864) or similar')
print()
print('The reshape error 34330 <- (1, 28864) suggests:')
print('- 34330 could be sum of nested array sizes when flattened')
print('- Or mismatch between what was stored vs what was expected')