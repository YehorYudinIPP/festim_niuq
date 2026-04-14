"""
Parameter definitions and metadata for FESTIM-NIUQ UQ campaigns.

This module provides descriptors for the physical parameters that are
treated as uncertain in the UQ pipeline.  Each parameter entry records
its physical name, SI unit, and a brief description so that plotting
and reporting utilities can annotate results without hard-coding
human-readable strings elsewhere in the codebase.

The canonical parameter list is maintained here; the actual values
(mean, distribution, relative standard deviation) are read at run time
from the YAML configuration file.
"""


#: Mapping of internal parameter key to metadata used by plotting and logging.
PARAMETER_DESCRIPTORS = {
    "D_0": {
        "name": "Diffusion coefficient pre-exponential",
        "symbol": r"$D_0$",
        "unit": r"m$^{2}$/s",
    },
    "E_D": {
        "name": "Diffusion activation energy",
        "symbol": r"$E_D$",
        "unit": "eV",
    },
    "kappa": {
        "name": "Thermal conductivity",
        "symbol": r"$\kappa$",
        "unit": r"W/(m$\cdot$K)",
    },
    "G": {
        "name": "Volumetric tritium generation rate",
        "symbol": r"$G$",
        "unit": r"m$^{-3}$ s$^{-1}$",
    },
    "Q": {
        "name": "Volumetric heat source",
        "symbol": r"$Q$",
        "unit": r"W/m$^{3}$",
    },
    "E_kr": {
        "name": "Surface recombination activation energy",
        "symbol": r"$E_{kr}$",
        "unit": "eV",
    },
    "h_conv": {
        "name": "Convective heat-transfer coefficient",
        "symbol": r"$h_\mathrm{conv}$",
        "unit": r"W/(m$^{2}\cdot$K)",
    },
    "E_k": {
        "name": "Trapping energy",
        "symbol": r"$E_k$",
        "unit": "eV",
    },
    "T": {
        "name": "Temperature",
        "symbol": r"$T$",
        "unit": "K",
    },
}
