import sys
sys.path.insert(0, 'sciduck/src')
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from sciduck import _infer_data_columns, _dataframe_to_storage_rows, _storage_to_python

# Test 1: multi-row scalar double
df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'B': ['x', 'y', 'z']})
col_types, meta = _infer_data_columns(df)
print('multi-row scalar types:', col_types)
print('meta mode:', meta['mode'])
print('col meta A:', meta['columns']['A'])
rows = _dataframe_to_storage_rows(df, meta)
print('storage rows:', rows)

# Test 2: 1-row scalar
df1 = pd.DataFrame({'x': [42.0], 'label': ['hello']})
col_types1, meta1 = _infer_data_columns(df1)
print('1-row scalar types:', col_types1)
rows1 = _dataframe_to_storage_rows(df1, meta1)
print('1-row storage rows:', rows1)

# Test 3: multi-row vector column
df2 = pd.DataFrame({'vec': [np.array([1.0,2.0,3.0]), np.array([4.0,5.0,6.0])]})
col_types2, meta2 = _infer_data_columns(df2)
print('vector col types:', col_types2)
rows2 = _dataframe_to_storage_rows(df2, meta2)
print('vector storage rows:', rows2)

# Reconstruct
from sciduck import _storage_to_python
result = {}
for c, m in meta['columns'].items():
    result[c] = [_storage_to_python(rows[i][list(meta['columns'].keys()).index(c)], m) for i in range(3)]
print('reconstructed:', pd.DataFrame(result))