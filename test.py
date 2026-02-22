import numpy as np                                                                                                                                                              
import sys                                                                                                                                                                    
sys.path.insert(0, 'sciduck/src')
from sciduck.sciduck import _python_to_storage

# Simulate ragged vector column: meta says ndarray, ndim=1
meta = {'python_type': 'ndarray', 'numpy_dtype': 'float64', 'ndim': 1}

# Test 1: normal ndarray (should work as before)
result = _python_to_storage(np.array([1.0, 2.0, 3.0]), meta)
print(f'ndarray [1,2,3]: {result}')

# Test 2: scalar float (the bug case)
result = _python_to_storage(1.0, meta)
print(f'scalar 1.0: {result}')

# Test 3: scalar int
result = _python_to_storage(1, meta)
print(f'scalar int 1: {result}')

print('All tests passed!')