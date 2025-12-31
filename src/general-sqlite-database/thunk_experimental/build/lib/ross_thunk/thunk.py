from typing import Callable, Any
from hashlib import sha256

STRING_REPR_DELIMITER = "-"
STRING_REPR = "{fcn_hash}{delimiter}{n_outputs}"

class Thunk:
    """A Python adaptation of the Thunk concept in Haskell for the purposes of building a data processing pipeline.
    To create a Thunk, pass the function to be thunk-ified as the first argument, followed by the required additional metadata.
    """

    def __init__(self, 
                 fcn: Callable,
                 n_outputs: int
        ):
        self.fcn = fcn
        self.n_outputs = n_outputs
        self.pipeline_thunks = tuple()  # Will hold a reference to the PipelineThunk when called

        # Hash the function's bytecode to create a unique identifier for the thunk
        fcn_code = fcn.__code__.co_code
        fcn_hash = sha256(fcn_code).hexdigest()

        # Create a unique string representation for the thunk
        string_repr = STRING_REPR.format(
            fcn_hash=fcn_hash,
            n_outputs=n_outputs,
            delimiter=STRING_REPR_DELIMITER
        )

        # Hash the string representation to create the final hash
        self.hash = sha256(string_repr.encode()).hexdigest()    
    

    def __repr__(self):
        return f"Thunk(fcn_name={self.fcn.__name__}, n_outputs={self.n_outputs}, hash={self.hash})"
    

    def __eq__(self, other):
        if not isinstance(other, Thunk):
            return False
        return self.hash == other.hash
    

    def __hash__(self):
        return int(self.hash, 16)
    

    def __call__(self, *args, **kwargs) -> Any:
        """Calling a Thunk generates a PipelineThunk that represents the function call with the provided arguments."""
        if self.pipeline_thunk is None:
            self.pipeline_thunk = PipelineThunk(self, *args, **kwargs) # Store a reference to the pipeline thunk

        return self.pipeline_thunk(*args, **kwargs)
        

class PipelineThunk(Thunk):
    """A Thunk subclass that represents a thunked function that has been called with inputs.
    PipelineThunk enables tracking the provenance of data in a pipeline.
    """

    def __init__(self, 
                 thunk: Thunk,
                 *args,
                 **kwargs
        ):
        self.thunk = thunk
        inputs = {}
        for i, arg in enumerate(args):
            inputs[f"arg_{i}"] = arg
        inputs.update(kwargs)
        self.inputs = inputs
        outputs = tuple(OutputThunk(pipeline_thunk=self, output_num=i, is_complete=self.is_complete, value=None) for output_num in range(thunk.n_outputs))
        self.outputs = outputs

        self.hash = sha256((thunk.hash + STRING_REPR_DELIMITER + str(hash(frozenset(self.inputs.items())))).encode()).hexdigest()


    def __call__(self, *args, as_thunk: bool = True, **kwargs) -> Any:
        """If complete (i.e. all inputs specified), run the thunked function with the provided arguments.
        If not complete, return OutputThunk(s) representing the output(s) of the function."""
        is_complete = self.is_complete
        result = (None, ) if self.thunk.n_outputs > 1 else None
        if is_complete:
            result = self.fcn(*args, **kwargs)
            if not as_thunk:
                return result
            
        if self.n_outputs == 1:
            return OutputThunk(pipeline_thunk=self, output_num=0, is_complete=is_complete, value=result)
        else:
            return tuple(OutputThunk(pipeline_thunk=self, output_num=output_num, is_complete=is_complete, value = res) for output_num, res in enumerate(result))                    
    
    
    def __repr__(self) -> str:
        return f"PipelineThunk(fcn_name={self.fcn.__name__}, n_inputs={len(self.inputs)}, n_outputs={self.n_outputs}, hash={self.hash})"
    

    @property
    def is_complete(self) -> bool:
        """Check if any inputs are Thunks. If not, then the function is "complete" and ready to be evaluated."""
        for value in self.inputs.values():
            if isinstance(value, Thunk):
                return False
        return True
    

class OutputThunk(Thunk):
    """A Thunk subclass that represents the output value of a thunked function.
    This is useful for tracking the provenance of data in a pipeline.
    """

    def __init__(self, 
                 pipeline_thunk: Thunk,
                 output_num: int,
                 is_complete: bool,
                 value: Any
        ):
        self.pipeline_thunk = pipeline_thunk
        self.output_num = output_num
        self.is_complete = is_complete
        self.value = value if is_complete else None

        string_repr = f"{pipeline_thunk.hash}{STRING_REPR_DELIMITER}output{STRING_REPR_DELIMITER}{output_num}"

        self.hash = sha256(string_repr.encode()).hexdigest()

    
    def __repr__(self):
        string_repr = "OutputThunk(source_fcn_name={source_thunk}, hash={hash}, value={value})"
        try:
            return string_repr.format(
                source_thunk=self.pipeline_thunk.fcn.__name__,
                hash=self.hash
            )
        except Exception:
            return string_repr.format(
                source_thunk=self.pipeline_thunk.fcn.__name__,
                hash=self.hash
            )
    

    def __eq__(self, other):
        """Data must come from the same source thunk and have the same value to be considered equal."""
        if not isinstance(other, OutputThunk):
            return False
        return self.hash == other.hash
    

    def __str__(self):
        """Show only the value when printed."""
        return str(self.value)