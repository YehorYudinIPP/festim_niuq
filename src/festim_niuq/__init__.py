"""
FESTIM-NIUQ — Non-Intrusive Uncertainty Quantification for FESTIM.

This is the top-level package.  It re-exports selected utilities from
the :mod:`festim_niuq.uq.util` sub-package for convenience.
"""

from .uq.util.utils import (
    load_config,
    add_timestamp_to_filename,
    get_festim_python,
    validate_execution_setup,
    save_sa_results,
    get_qoi_names,
    get_sobol_first,
    get_sobol_total,
    get_stat,
    integrate_statistics,
    compute_absolute_tolerance,
)

__all__ = [
    "load_config",
    "add_timestamp_to_filename",
    "get_festim_python",
    "validate_execution_setup",
    "save_sa_results",
    "get_qoi_names",
    "get_sobol_first",
    "get_sobol_total",
    "get_stat",
    "integrate_statistics",
    "compute_absolute_tolerance",
]
