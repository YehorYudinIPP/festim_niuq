"""
Warning filters for FESTIM-NIUQ.

Import this module early (e.g. in ``festim_model/__init__.py``) to
suppress noisy deprecation and compatibility warnings from upstream
libraries (pkg_resources, UFL) that do not affect correctness.
"""
import warnings

# Suppress pkg_resources deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# Suppress other common warnings if needed
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ufl")
warnings.filterwarnings("ignore", category=UserWarning, module="ufl")
