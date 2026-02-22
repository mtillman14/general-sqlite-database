from scidb.filters import VariableFilter                                                                                                                                        
                                                                                                                                                                                   
# Simulate to_key() to verify format                                                                                                                                            
class MockVar:                                                                                                                                                                
    __name__ = 'Side'

f = VariableFilter(MockVar, '==', 'L')
print('to_key:', repr(f.to_key()))