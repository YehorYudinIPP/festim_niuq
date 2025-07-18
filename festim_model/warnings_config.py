"""
Configuration for suppressing common warnings in FESTIM-NIUQ
"""
import warnings

# Suppress pkg_resources deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# Suppress other common warnings if needed
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ufl")
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
