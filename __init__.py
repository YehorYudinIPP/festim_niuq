# Made festim_model accessible at the package level
import sys
import os

# Add parent directory to Python path to access festim_model package
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now you can import festim_model from parent directory
try:
    import festim_model
    print("festim_model package successfully imported from parent directory")
except ImportError as e:
    print(f"Could not import festim_model: {e}")

#from .festim_model import Model 