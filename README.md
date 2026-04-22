# FESTIM-NIUQ

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/YehorYudinIPP/festim_niuq/actions/workflows/tests.yml/badge.svg)](https://github.com/YehorYudinIPP/festim_niuq/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/1021648513.svg)](https://zenodo.org/badge/latestdoi/1021648513)

**FESTIM-NIUQ** is a Python package for **Non-Intrusive Uncertainty Quantification** of tritium transport simulations using the [FESTIM](https://github.com/festim-dev/FESTIM) finite-element framework.

The package couples [EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ) (from the [SEAVEA Toolkit](http://www.seaveatk.org/)) with FESTIM to propagate parametric uncertainties through tritium transport models and compute Sobol sensitivity indices via Polynomial Chaos Expansion (PCE) and Quasi-Monte Carlo (QMC) methods.

## Features

- **Non-intrusive UQ**: treats the FESTIM solver as a black box — no solver modifications needed
- **Sensitivity analysis**: first-order and total-order Sobol indices for spatially-resolved quantities
- **Multiple UQ methods**: Polynomial Chaos Expansion (PCE), Quasi-Monte Carlo (QMC), and Finite Differences
- **Correlated parameters**: supports non-diagonal covariance matrices via Cholesky decomposition
- **Bayesian inverse UQ**: PCE surrogate + MCMC inversion with `emcee`
- **Flexible configuration**: all model and UQ settings controlled via YAML files
- **HPC-ready**: QCG-PilotJob integration and SLURM scripts included
- **Automated plotting**: publication-quality uncertainty and sensitivity plots

## Installation

### Prerequisites

- Python >= 3.9
- [FESTIM](https://github.com/festim-dev/FESTIM) (requires FEniCSx / DOLFINx)

### Quick install

The easiest way to set up the environment is with conda:

```bash
# Create and activate environment with FESTIM
conda create -n festim-env
conda activate festim-env
conda install -c conda-forge festim

# Install FESTIM-NIUQ and its dependencies
pip install -r requirements.txt
```

Alternatively, install directly:

```bash
pip install -e .
```

### Development install

```bash
pip install -e ".[dev]"
```

## Usage

### Basic UQ campaign

Run the default UQ campaign with built-in test configuration:

```bash
cd uq
python3 easyvvuq_festim.py
```

This will compute statistical moments of the tritium inventory and generate Sobol sensitivity index plots.

### Custom configuration

Provide your own YAML configuration file:

```bash
python3 easyvvuq_festim.py --config config/config.uq_test_cj1959.yaml
```

### Correlated parameters

Run UQ with correlated input parameters:

```bash
python3 easyvvuq_festim_correlated.py --config config/config.uq_test_cj1959.yaml --uq-scheme pce --p-order 3
```

### Parameter scanning

Perform a parameter scan over a single parameter:

```bash
python3 festim_model_scan.py
```

### Bayesian inverse UQ

Run Bayesian inversion using a PCE surrogate and MCMC:

```bash
python3 bayesian_inverse_uq.py --config config/config_bayesian_ss.yaml --p-order 3
```

## Configuration

All settings are centralised in a YAML configuration file that controls:

- **Geometry**: domain size, coordinate system (Cartesian/cylindrical/spherical), mesh resolution
- **Materials**: transport coefficients with mean values and uncertainties
- **Boundary conditions**: Dirichlet, Neumann, or convective flux
- **Source terms**: volumetric generation rates
- **Simulation**: steady-state or transient, time stepping, solver tolerances
- **UQ settings**: parameter distributions, polynomial order, number of samples

See example configurations in [`uq/config/`](uq/config/).

## Testing

Run the test suite:

```bash
pytest tests/
```

## Project Structure

```
festim_niuq/
├── festim_model/         # FESTIM model wrapper
│   ├── Model.py          # Main model class (FESTIM 2.0 API)
│   └── diagnostics/      # Post-processing diagnostics
├── uq/                   # UQ orchestration layer
│   ├── easyvvuq_festim.py            # Main UQ campaign script
│   ├── easyvvuq_festim_correlated.py # Correlated parameters UQ
│   ├── bayesian_inverse_uq.py        # Bayesian inverse UQ
│   ├── festim_model_run.py           # Single FESTIM run wrapper
│   ├── festim_model_scan.py          # Parameter scanning
│   ├── config/                       # Example YAML configurations
│   └── util/                         # Encoders, decoders, plotting
├── tests/                # Unit and integration tests
├── docs/                 # Documentation
└── paper/                # JOSS paper
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

If you use FESTIM-NIUQ in your research, please cite it. See [CITATION.cff](CITATION.cff) for details.

## Acknowledgements

FESTIM-NIUQ was developed at the Nuclear Futures Institute, Bangor University.
