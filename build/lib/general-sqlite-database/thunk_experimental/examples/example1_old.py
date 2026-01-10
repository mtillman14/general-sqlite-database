import sys
sys.path.append("/Users/mitchelltillman/Desktop/Not_Work/Code/Python_Projects/ross_thunk/src")
from ross_thunk.thunk import Thunk
from ross_thunk.data_level import DataLevel

from ross_thunk.build_dag import build_dag

def example_fcn1(a: int, b: int) -> int:
    return a + b

def example_fcn2(total: int, a: int, b: int) -> int:
    return total - b - a

example_fcn1 = Thunk(
    fcn=example_fcn1,
    n_outputs=1
)

example_fcn2_thunk = Thunk(
    fcn=example_fcn2,
    n_outputs=1
)

# Create a simple pipeline
a = 5
# b = 3
b = Thunk
result1 = example_fcn1(a, 3)  # Should return 8
print(result1)
result2 = example_fcn2_thunk(result1, a, b)  # Should return 0
print(result2)

dag = build_dag((example_fcn1_thunk, example_fcn2_thunk))
for edge in dag.edges:
    print(edge)

# Create a more complex pipeline that dynamically loads data
def load_data(file_path_format: str, **kwargs) -> list:    
    file_path = file_path_format.format(**kwargs)
    return [i for i in range(5)]  # Simulate loading data from a file

def filter_data(data: list, threshold: int) -> list:
    return [x for x in data if x > threshold]

load_data_thunk = Thunk(
    fcn=load_data,
    n_outputs=1
)
filter_data_thunk = Thunk(
    fcn=filter_data,
    n_outputs=1
)

# Pipeline definition
file_path_format = 'data/subject_{subject}_trial_{trial}.txt'
loaded_data = load_data_thunk(file_path_format=file_path_format, subject=Thunk, trial=Thunk)  # Load data based on subject and trial
filtered_data = filter_data_thunk(loaded_data, threshold=0)  # Filter out non-positive values

data_levels = [
    {'subject': '01', 'trial': '01'}, 
    {'subject': '01', 'trial': '02'}
]
for data_level in data_levels:
    data = load_data_thunk(file_path_format, subject=data_level['subject'], trial=data_level['trial'])  # Should load data from the specified file
    saver.save(data, data_level)  # Simulate saving the processed data