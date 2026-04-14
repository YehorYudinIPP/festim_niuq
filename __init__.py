"""
FESTIM-NIUQ — Non-Intrusive Uncertainty Quantification for FESTIM.

This is the top-level package.  It re-exports selected utilities from
the :mod:`uq.util` sub-package for convenience.
"""
import sys
import os

# Add parent directory to Python path to access festim_model package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import festim_model from parent directory
try:
    import festim_model
except ImportError as e:
    print(f"Could not import festim_model: {e}")

# Import utility functions from uq.util
try:
    from .uq.util import add_timestamp_to_filename, get_festim_python, validate_execution_setup
    __all__ = ['add_timestamp_to_filename', 'get_festim_python', 'validate_execution_setup']
except ImportError as e:
    print(f"Could not import utility functions: {e}")
    __all__ = [] 