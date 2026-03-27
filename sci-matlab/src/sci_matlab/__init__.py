"""MATLAB bridge for SciStack.

Provides proxy classes that satisfy the duck-typing contracts of
scilineage's LineageFcn/LineageFcnInvocation/LineageFcnResult, allowing
MATLAB functions to participate in the SciStack lineage system without
modifying any existing Python packages.

Usage from MATLAB (via py. interface):
    py.sci_matlab.bridge.MatlabLineageFcn(source_hash, 'my_func', false)
"""

from .bridge import (
    MatlabLineageFcn,
    MatlabLineageFcnInvocation,
    check_cache,
    make_lineage_fcn_result,
    register_matlab_variable,
    get_surrogate_class,
)

__all__ = [
    "MatlabLineageFcn",
    "MatlabLineageFcnInvocation",
    "check_cache",
    "make_lineage_fcn_result",
    "register_matlab_variable",
    "get_surrogate_class",
]
