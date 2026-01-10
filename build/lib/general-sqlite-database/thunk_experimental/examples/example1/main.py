from ross_thunk.thunk import Thunk

# Define the dataset containing all levels of all factors in the dataset
dataset = Thunk.dataset(
    data_levels_csv='data_levels.csv',
    primary_keys=['subject', 'trial'],
    root_folder='~/data'
)

# Define the database location
Thunk.set_db_path(db_path="test.db")

# Import the pipeline module
import pipeline
# thunks = Thunk.get_thunks_from_module(pipeline)
# thunks = tuple([obj for obj in vars(pipeline).values() 
#                 if isinstance(obj, Thunk)])

# Run the pipeline. By default, all functions will run at the lowest data level unless specified otherwise.
# For example:
pipeline.example_fcn1_thunk.data_level = dataset.primary_keys[0] # Now this will run at the subject level.
status = Thunk.run_pipeline(pipeline, dataset=dataset)
print(status)