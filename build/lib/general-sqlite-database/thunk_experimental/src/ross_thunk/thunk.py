from typing import Callable, Any
from hashlib import sha256

STRING_REPR_DELIMITER = "-"
THUNK_STRING_REPR = "{fcn_hash}{delimiter}{n_outputs}"

class Thunk:
    """A Python adaptation of the Thunk concept in Haskell for the purposes of building a data processing pipeline.
    To create a Thunk, pass the function to be thunked as the first argument, followed by the number of output variables.
    """

    thunk_db = None  # Placeholder for a ThunkDB instance, to persist Thunks to a database

    def __init__(self, 
                 fcn: Callable,
                 n_outputs: int
        ):
        self.fcn = fcn
        self.n_outputs = n_outputs
        self.pipeline_thunks = tuple()  # Will hold a reference to each PipelineThunk created from this Thunk

        # Hash the function's bytecode to create a unique identifier for the thunk
        fcn_code = fcn.__code__.co_code
        fcn_hash = sha256(fcn_code).hexdigest()

        # Create a unique string representation for the thunk
        string_repr = THUNK_STRING_REPR.format(
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
    

    def __call__(self, *args, as_thunk: bool = True, **kwargs) -> Any:
        """Calling a Thunk generates a PipelineThunk that represents the function call with the provided arguments."""
        pipeline_thunk = PipelineThunk(self, *args, **kwargs)
        match_found = False
        for existing_pipeline_thunk in self.pipeline_thunks:            
            # Check if this call of this PipelineThunk replaces any Thunk inputs with concrete values
            existing_keys = list(existing_pipeline_thunk.inputs.keys())
            input_keys = list(pipeline_thunk.inputs.keys())

            # Replace the inputs of the existing PipelineThunk with those of the new one (to fill in any Thunks while preserving the PipelineThunk reference)
            if pipeline_thunk._partial_or_more_match(existing_pipeline_thunk):                                
                for key_num in range(len(pipeline_thunk.inputs)):
                    existing_input_key = existing_keys[key_num]
                    input_key = input_keys[key_num]
                    existing_pipeline_thunk.inputs[existing_input_key] = pipeline_thunk.inputs[input_key]                
                match_found = True
                pipeline_thunk = existing_pipeline_thunk
                break
            
        if not match_found:
            self.pipeline_thunks = tuple(list(self.pipeline_thunks) + [pipeline_thunk])

        return pipeline_thunk(*args, as_thunk=as_thunk, **kwargs)    
        

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

        self.outputs = tuple()  # Will be populated when the PipelineThunk is called


    @property
    def hash(self) -> str:
        """Hash is a dynamic property that changes as inputs are filled in."""
        return sha256((self.thunk.hash + STRING_REPR_DELIMITER + str(hash(frozenset(self.inputs.items())))).encode()).hexdigest()  


    def __hash__(self):
        return int(self.hash, 16)      


    def __call__(self, *args, as_thunk: bool = True, **kwargs) -> Any:
        """If complete (i.e. all inputs specified), run the thunked function with the provided arguments.
        If not complete, return OutputThunk(s) representing the output(s) of the function."""
        is_complete = self.is_complete
        result = tuple(None for _ in range(self.thunk.n_outputs))
        if is_complete:
            args_list = list(args)
            for arg_num in range(len(args_list)):
                if isinstance(args_list[arg_num], OutputThunk):
                    args_list[arg_num] = args_list[arg_num].value
            result = self.thunk.fcn(*args_list, **kwargs)
            # Ensure result is a tuple of the correct length
            # If it's not a tuple, wrap it in one (means there's only one output)
            # If it is a tuple, but only one output is expected, wrap it in a single-element tuple
            if not isinstance(result, tuple) or (isinstance(result, tuple) and len(result) == 1 and self.thunk.n_outputs == 1):
                result = (result,)
            if len(result) != self.thunk.n_outputs:
                raise ValueError(f"Function {self.thunk.fcn.__name__} returned {len(result)} outputs, but {self.thunk.n_outputs} were expected.")
            if not as_thunk:
                if self.thunk.n_outputs == 1:
                    return result[0]
                return result
            
        outputs = tuple(OutputThunk(pipeline_thunk=self, output_num=output_num, is_complete=is_complete, value = res) for output_num, res in enumerate(result))
        self.outputs = outputs
        if self.thunk.n_outputs == 1:
            return outputs[0]
        else:
            return outputs                
    
    
    def __repr__(self) -> str:
        return f"PipelineThunk(fcn_name={self.thunk.fcn.__name__}, n_inputs={len(self.inputs)}, n_outputs={self.thunk.n_outputs}, hash={self.hash})"
    

    @property
    def is_complete(self) -> bool:
        """Check if any inputs are Thunks. If not, then the function is "complete" and ready to be evaluated."""
        for value in self.inputs.values():
            if value is Thunk or (isinstance(value, OutputThunk) and not value.is_complete):
                return False
        return True
    

    def _partial_or_more_match(self, other: "PipelineThunk") -> bool:
        """Determine if this PipelineThunk is an exact match (if complete) or a partial match of another PipelineThunk.
        A partial match means that all non-Thunk inputs of this PipelineThunk match the corresponding inputs of the other PipelineThunk AND
        all Thunk inputs of this PipelineThunk are also present in the other PipelineThunk (but may have different values).
        """
        if not isinstance(other, PipelineThunk):
            return False
        if self == other:
            return True
        self_inputs_keys = list(self.inputs.keys())
        other_inputs_keys = list(other.inputs.keys()) 
              
        # Check key names match, allowing for positional vs kwarg differences
        if len(self_inputs_keys) != len(other_inputs_keys):
            return False
        for count, self_arg_key in enumerate(self_inputs_keys):
            other_arg_key = other_inputs_keys[count]
            # If one is a positional arg and the other is a kwarg, make them match
            if (self_arg_key.startswith("arg_") and not other_arg_key.startswith("arg_")) or (not self_arg_key.startswith("arg_") and other_arg_key.startswith("arg_")):
                other_inputs_keys[count] = self_arg_key
            elif self_arg_key != other_arg_key:
                # If both positional or both kwarg but different names, no match
                return False
        if not self_inputs_keys == other_inputs_keys:
            return False
        
        # Check values match, allowing for Thunks to differ
        self_inputs_keys = list(self.inputs.keys())
        other_inputs_keys = list(other.inputs.keys()) 
        for key_num in range(len(self_inputs_keys)):
            self_arg_key = self_inputs_keys[key_num]
            other_arg_key = other_inputs_keys[key_num]
            value = self.inputs[self_arg_key]
            other_value = other.inputs[other_arg_key]
            if value is Thunk or other_value is Thunk:
                continue
            if value != other_value:
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
                    source_thunk=self.pipeline_thunk.thunk.fcn.__name__,
                    hash=self.hash,
                    value=str(self.value)
                )
        except Exception:
                return string_repr.format(
                    source_thunk=self.pipeline_thunk.thunk.fcn.__name__,
                    hash=self.hash,
                    value=str(self.value)
                )
    

    def __eq__(self, other):
        """Data must come from the same source thunk and have the same value to be considered equal.
        Otherwise, if compared to a non-OutputThunk, compare the contained value."""
        if not isinstance(other, OutputThunk):
            return self.value == other
        return self.hash == other.hash
    

    def __str__(self):
        """Show only the value when printed."""
        return str(self.value)
    

    def __hash__(self):
        return int(self.hash, 16)