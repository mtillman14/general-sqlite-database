from ross_thunk.thunk import Thunk

# Define the dataset containing all levels of all factors in the dataset
dataset = Thunk.dataset(
    data_levels_csv='data_levels.csv',
    primary_keys=['subject', 'trial'],
    root_folder='~/data'
)

# Define the database location
Thunk.set_db_path(db_path="test.db")

# Pipeline
# Function definitions. These would typically be imported from a separate file.
def example_fcn1(a: int, b: int) -> int:
    return a + b

def example_fcn2(total: int, a: int, b: int) -> int:
    return total - b - a

# Convert functions to Thunks
example_fcn1_thunk = Thunk(
    fcn=example_fcn1,
    n_outputs=1
)

example_fcn2_thunk = Thunk(
    fcn=example_fcn2,
    n_outputs=1
)

# Define the pipeline
a = 5
b = 2
for trial in dataset['trials']:
    result1 = example_fcn1_thunk(a, b)  # Should return 7
    print(f"Result of example_fcn1: {result1}")
    result2 = example_fcn2_thunk(result1, a, b)  # Should return 0
    print(f"Result of example_fcn2: {result2}")

    # If result1 needs to be saved in a bespoke fashion
    result1.dump_type = Thunk.timeseries

    # Save to the database
    trial.dump(result1)
    trial.dump(result2)