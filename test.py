import pandas as pd                                                                                                                                                             
import numpy as np                                                                                                                                                            

# Simulate the test case
df = pd.DataFrame({'force': [1.0, 2.0, 3.0], 'velocity': [4.0, 5.0, 6.0]})

# What the current code does (line 345):
col_series = df['force']
cell_val = col_series.iloc[0]  # Gets 1.0 (a single scalar)
print(f'Current approach - first cell: {cell_val}, type: {type(cell_val)}')

# What should happen (what commit fe99569 originally had):
col_data = col_series.to_numpy()  # Gets array([1., 2., 3.])
print(f'Expected approach - whole column: {col_data}, type: {type(col_data)}')