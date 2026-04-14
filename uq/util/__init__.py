"""
Utility sub-package for FESTIM-NIUQ.

Provides custom YAML encoders (:mod:`~uq.util.Encoder`), CSV decoders
(:mod:`~uq.util.Decoder`), parameter metadata
(:mod:`~uq.util.Parameters`), helper functions
(:mod:`~uq.util.utils`), and publication-quality plotting routines
(:mod:`~uq.util.plotting`).
"""

from . import Encoder
from . import Decoder

from . import utils
from . import plotting

# Import specific functions for easy access
from .utils import (
    load_config,
    add_timestamp_to_filename,
    get_festim_python,
    validate_execution_setup,
    save_sa_results_yaml,
)

# Make encoder classes and utility functions available at package level
__all__ = [
    "YAMLEncoder",
    "AdvancedYAMLEncoder",
    "UQPlotter",
    "load_config",
    "add_timestamp_to_filename",
    "get_festim_python",
    "validate_execution_setup",
    "save_sa_results_yaml",
]
