from thunk import manual, extract_lineage, get_upstream_lineage, thunk                                                                                                              
                                                        
# Basic usage                                                                                                                                                                       
result = manual([1, 2, 3], label="outlier_removal", reason="bad sensor")                                                                                                            
print("data:", result.data)
print("function_name:", extract_lineage(result).function_name)

# Downstream chaining
@thunk
def double(x):
    return [v * 2 for v in x]

out = double(result)
print("downstream result:", out.data)
chain = get_upstream_lineage(out)
print("lineage chain:", [r["function_name"] for r in chain])

# Hash determinism
r1 = manual([1, 2, 3], label="edit")
r2 = manual([1, 2, 3], label="edit")
r3 = manual([4, 5, 6], label="edit")
print("same inputs same hash:", r1.hash == r2.hash)
print("diff data diff hash:", r1.hash != r3.hash)

# Re-export from scidb
from scidb import manual as scidb_manual
print("re-export works:", scidb_manual is manual)