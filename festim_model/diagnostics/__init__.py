"""
Post-processing diagnostics for FESTIM simulation results.

Provides the :class:`Diagnostics` class, which reads FESTIM result
files (plain-text CSV or VTX), computes derived quantities (tritium
inventory, surface fluxes), and generates transient visualisation plots.
"""

from .Diagnostics import Diagnostics

# Make Diagnostics class available at package level
__all__ = ["Diagnostics"]
