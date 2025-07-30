# festim_niuq package initialization
import sys
import os

# Add parent directory to Python path to access festim_model package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import festim_model from parent directory
try:
    import festim_model
    print("festim_model package successfully imported from parent directory")
except ImportError as e:
    print(f"Could not import festim_model: {e}")

# Import utility functions from uq.util
try:
    from .uq.util import add_timestamp_to_filename, get_festim_python, validate_execution_setup
    __all__ = ['add_timestamp_to_filename', 'get_festim_python', 'validate_execution_setup']
except ImportError as e:
    print(f"Could not import utility functions: {e}")
    __all__ = []

#from .festim_model import Model 