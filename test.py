import sys, os
from pathlib import Path
root = Path('.')
sys.path.insert(0, str(root / 'src'))
sys.path.insert(0, str(root / 'canonical-hash' / 'src'))
sys.path.insert(0, str(root / 'thunk-lib' / 'src'))
sys.path.insert(0, str(root / 'path-gen' / 'src'))
sys.path.insert(0, str(root / 'pipelinedb-lib' / 'src'))
sys.path.insert(0, str(root / 'scirun-lib' / 'src'))
sys.path.insert(0, str(root / 'sciduck' / 'src'))

import numpy as np
import pandas as pd
from scidb.database import _flatten_struct_columns, _unflatten_struct_columns

# Test 1: DataFrame with nested struct columns (scalar leaves)
print('=== Test 1: Scalar leaves ===')
n_rows = 4
data = {
    'name': ['a', 'b', 'c', 'd'],
    'value': [1.0, 2.0, 3.0, 4.0],
    'seconds': [
        {'gaitPhases': {'start': {'leftStance': 0.1, 'leftSwing': 0.5}, 'stop': {'leftStance': 0.4, 'leftSwing': 0.8}}},
        {'gaitPhases': {'start': {'leftStance': 0.2, 'leftSwing': 0.6}, 'stop': {'leftStance': 0.5, 'leftSwing': 0.9}}},
        {'gaitPhases': {'start': {'leftStance': 0.3, 'leftSwing': 0.7}, 'stop': {'leftStance': 0.6, 'leftSwing': 1.0}}},
        {'gaitPhases': {'start': {'leftStance': 0.4, 'leftSwing': 0.8}, 'stop': {'leftStance': 0.7, 'leftSwing': 1.1}}},
    ]
}
df = pd.DataFrame(data)
print(f'Original: {df.columns.tolist()}')
print(f'Original dtypes: {dict(df.dtypes)}')

flat_df, info = _flatten_struct_columns(df)
print(f'Flattened: {flat_df.columns.tolist()}')
print(f'Struct info: {info}')
print(flat_df)

# Unflatten
restored = _unflatten_struct_columns(flat_df, info)
print(f'Restored: {restored.columns.tolist()}')
print(f'Row 0 seconds: {restored["seconds"].iloc[0]}')
assert restored['seconds'].iloc[0]['gaitPhases']['start']['leftStance'] == 0.1
assert restored['seconds'].iloc[1]['gaitPhases']['stop']['leftSwing'] == 0.9
print('Test 1 PASSED!')

# Test 2: Array leaves
print()
print('=== Test 2: Array leaves ===')
data2 = {
    'name': ['a', 'b'],
    'seconds': [
        {'gaitPhases': {'start': {'leftStance': np.array([1.0, 2.0, 3.0])}}},
        {'gaitPhases': {'start': {'leftStance': np.array([4.0, 5.0, 6.0])}}},
    ]
}
df2 = pd.DataFrame(data2)
flat_df2, info2 = _flatten_struct_columns(df2)
print(f'Flattened: {flat_df2.columns.tolist()}')
print(f'Info: {info2}')
print(flat_df2)

restored2 = _unflatten_struct_columns(flat_df2, info2)
print(f'Restored row 0: {restored2["seconds"].iloc[0]}')
arr = restored2['seconds'].iloc[0]['gaitPhases']['start']['leftStance']
print(f'Array value: {arr}, type: {type(arr).__name__}')
np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))
print('Test 2 PASSED!')

# Test 3: No struct columns (should be a no-op)
print()
print('=== Test 3: No struct columns ===')
df3 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
flat_df3, info3 = _flatten_struct_columns(df3)
assert info3 == {}
assert flat_df3.columns.tolist() == ['a', 'b']
print('Test 3 PASSED!')

print()
print('ALL TESTS PASSED!')