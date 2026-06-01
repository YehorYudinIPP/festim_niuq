---
title: 'FESTIM-NIUQ: Non-Intrusive Uncertainty Quantification for Tritium Transport Modelling'
tags:
  - Python
  - uncertainty quantification
  - tritium transport
  - fusion energy
  - sensitivity analysis
  - polynomial chaos expansion
authors:
  - name: Yehor Yudin
    orcid: 0000-0001-8867-3207
    affiliation: 1
  - name: Tom Griffiths
    orcid: 0000-0002-8120-9134
    affiliation: 1
  - name: Tom Smith
    orcid: 0000-0002-9956-4046
    affiliation: 1
  - name: Tessa Davey
    orcid: 0000-0002-4262-1054
  - name: Simon Middleburgh
    orcid: 0000-0003-2537-4001
    affiliation: 1
    affiliation: 1
  - name: Cillian Cockrell
    orcid: 0000-0002-8501-7287
    affiliation: 1

affiliations:
  - name: Nuclear Futures Institute, Bangor University, Bangor, LL57 1UT, United Kingdom
    index: 1
date: 14 April 2026
bibliography: paper.bib
---

# Summary

Tritium is a radioactive isotope of hydrogen that will be both the fuel and a safety-critical inventory item in future fusion power plants.
Accurate predictions of tritium transport through fusion reactor components, such as lithium-ceramic breeder blankets and plasma-facing components, are essential for assuring the performance of a fuel cycle, as well as assessing safety and reactor design.
Yet the material transport coefficients that govern these predictions carry substantial experimental uncertainty, and their *ab initio* estimates might contain uncertainties that have to be accounted for as well.

FESTIM-NIUQ is a Python package that automates non-intrusive uncertainty quantification (UQ) for tritium transport simulations performed with the FESTIM finite-element framework [@delaporte2019festim; @dark2026festim].
The package couples FESTIM with EasyVVUQ [@richardson2020easyvvuq] and ChaosPy [@feinberg2015chaospy] to propagate parametric uncertainties through diffusion–reaction models of tritium behaviour in fusion-relevant materials.
Given user-specified probability distributions on input parameters — such as the diffusion pre-exponential factor, thermal conductivity, volumetric tritium generation rate, and tritium surface recombination energy, or other paramaters for other species — FESTIM-NIUQ automatically generates parameter samples, executes FESTIM simulations in parallel, and computes statistical moments and Sobol sensitivity indices [@sobol1993sensitivity; @sobol2001global] of quantities of interest, including spatially resolved concentration profiles and integral tritium inventories.  
The entire workflow is controlled by a single YAML configuration file, which specifies model equations and theoretical terms, geometry and boundary conditions, as well as parameter values, both deterministic and uncertain, making it accessible to fusion materials scientists who have no prior UQ expertise.

The *0.2.0* version of the package is available as source code at *github.com*, installable from the *PyPI* repository, and is archived at *ZENODO*.
The repository covers basic functionality with unit tests, provides several verification cases, and allows users to adapt it to specific needs via permissive licensing.

# Statement of Need

Fusion reactor components, such as tungsten first walls and beryllium covers, interact with tritium over long operational periods.
Tritium must be carefully inventoried and managed in fusion reactor components because of its radioactivity, scarcity, and role as fuel.
Tritium transport is governed by a coupled diffusion-reaction system with Arrhenius-type coefficients whose values are often measured with uncertainties of tens of per cent [@causey2012tritium].
Reliable safety assessments and design decisions require an understanding of how these input uncertainties propagate to predicted tritium inventories and release rates [@longhurst2011verification].
Analysis that targets design and prediction should provide confidence intervals on top of the mean estimates for tritium inventory and release rates, since these directly affect safety licensing [@mirallesdolz2024uncertainty].

## Gap in Existing Tools

Although general-purpose UQ frameworks exist (see *State of the Field* below), coupling any of them to a finite-element tritium transport solver requires non-trivial engineering: writing a solver-specific parameter encoder, an output decoder, a subprocess execution harness, and post-processing and plotting utilities.
This barrier is high enough that most published tritium transport studies report only deterministic results at nominal parameter values, forgoing systematic UQ entirely.

[TODO: Cite UQ studies on hydrogen transport that used manual/ad-hoc methods to further motivate automation.]

FESTIM-NIUQ removes this barrier for the FESTIM user community [@delaporte2024festim] by providing a ready-to-use pipeline that handles every step between a YAML configuration file and publication-quality sensitivity-index plots.
The package is designed to be extended: new uncertain parameters, boundary conditions, or coordinate geometries are added by editing the configuration file rather than modifying the Python source code.
<!-- FESTIM-NIUQ has been used in ongoing research at the Nuclear Futures Institute, Bangor University, to assess parametric uncertainties in lithium-ceramic breeder blanket tritium transport simulations, and it has been presented at the UKAEA Technical Meeting [@ukaea2026meeting] and the Open-Source Software for Fusion Energy Workshop (OSSFE 2026) [@ossfe2026]. -->

Verification of the solver wrapper has been performed against Carslaw and Jaeger's analytical solutions for diffusion in a sphere [@carslaw1959conduction], and convergence of the polynomial chaos expansion (PCE) surrogate has been confirmed with increasing polynomial order.

[TODO: add PCE scaling study]

# State of the Field

Several general-purpose UQ frameworks exist, including Dakota [@adams2021dakota], OpenTURNS [@baudin2017openturns], UQLab [@marelli2014uqlab], SALib [@herman2017salib], and ChaosPy [@feinberg2015chaospy].
While powerful, these tools require users to write bespoke glue code, which includes parameter encoders, solver wrappers, and output decoders for each specific solver, which is a significant effort for finite-element tritium transport problems involving nested YAML configurations, VTX result files, and subprocess execution.
Each of the specific hydrogen and tritium transport frameworks, alternative codes including  TMAP8 [@simon2025tmap8], TESSIM-X [@schmid2012tessim], SAETTA [@hattab2025saetta] and HIIPC [@sanghiipc], each with different physical scope, model specifics and assumptions, numerical backend, and input formats.
They would require adapting UQ tools for the specific use; however, the experience of applying generic UQ methods for hydrogen transport provides a pathway to the adoption of these methods in the field.

EasyVVUQ [@richardson2020easyvvuq] from the SEAVEA Toolkit provides a flexible VVUQ workflow engine and reduces this burden, but the user must still implement solver-specific encoder and decoder classes.
FESTIM-NIUQ fills this niche by providing pre-built YAML-based encoders with deep nested parameter substitution via dot-notation paths, CSV decoders, a subprocess execution harness, and publication-quality plotting routines, all tailored to the FESTIM data model.  The result is that a user can launch a complete UQ campaign with a single command and configuration file without writing any Python code.

Contributing a generic FESTIM integration upstream to EasyVVUQ was considered but rejected because the integration requires FESTIM-specific knowledge of its configuration schema, output file formats, and coordinate-system conventions.  Maintaining it as a standalone package allows independent versioning aligned with FESTIM releases and keeps the EasyVVUQ core free of solver-specific logic.

[TODO: table summary of existing tools: FESTIM integration, PCE support, YAML config, fusion-specific QoIs]

# Software Design

FESTIM-NIUQ adopts a non-intrusive architecture in which the FESTIM solver is treated as a black box.
This design decouples the UQ layer from the solver internals, allowing users to upgrade FESTIM independently and to apply the same UQ machinery to different transport models without code changes.

The package consists of three layers:

1. **Model wrapper** (`festim_model/`): Encapsulates FESTIM model configuration, execution, and result export for both FESTIM 2.0 (DOLFINx-based) and the legacy FESTIM 1.x API.
Constructs the model: geometry, mesh, material properties, boundary conditions, solver settings.
2. **UQ orchestration** (`uq/`): Manages parameter sampling, campaign execution, and analysis using EasyVVUQ and ChaosPy.
Contains encoder/decoder classes to access generic FESTIM models.
Supports Polynomial Chaos Expansion (PCE), Quasi-Monte Carlo (QMC), and Bayesian inverse UQ via PCE surrogate and MCMC.
3. **Utilities** (`uq/util/`): Custom YAML encoders that perform deep nested parameter substitution via dot-notation paths, CSV decoders, and publication-quality plotting routines for Sobol indices and statistical profile bands.

The `AdvancedYAMLEncoder` replaces parameter values at arbitrary nesting depths in YAML configuration files without requiring Jinja templates, which simplifies the workflow and reduces the risk of template syntax errors.
Support for correlated parameters is provided via Cholesky decomposition of user-specified covariance matrices.

For high-performance computing environments, FESTIM-NIUQ integrates with QCG-PilotJob for embarrassingly parallel sample evaluation on cluster resources, and SLURM submission scripts are included for common HPC platforms.  
On workstations, the same campaign runs locally using `joblib` multiprocessing without any configuration changes, ensuring reproducibility across environments.

## Configuration Interface

All UQ settings are controlled through a YAML configuration file \autoref{lst:yaml}:

```python
parameters:
  D:
    type: Uniform
    lower: 1.5e-8
    upper: 3.5e-8     # m^2/s - diffusion coefficient
  G:
    type: Uniform
    lower: 8.0e19
    upper: 1.2e20     # /m^3/s - volumetric generation rate
  C_boundary:
    type: Uniform
    lower: 1.0e20
    upper: 5.0e20     # /m^3 - boundary concentration

uq:
  method: pce         # pce | qmc
  polynomial_order: 3
  qoi: tritium_inventory

festim:
  script: festim_model/model.py
```

: Example UQ configuration (*config.uq.yaml*). \label{lst:yaml}

# Supported UQ Methods

A number of non-intrusive parametric uncertainty quantification methods implemented in EasyVVUQ is supported by the package.

  - **Polynomial Chaos Expansion (PCE)**: Requires $\mathcal{O}(p^d)$ model evaluations for polynomial order $p$ and $d$ uncertain parameters (there is a $\binom{p+d}{d}$ method for sparse version).
  Yields analytical Sobol decomposition from the PCE coefficients [@saltelli1995about].
  - **Quasi-Monte Carlo (qMC)**: Uses Sobol sequences for low-discrepancy sampling. 
  Suitable for high-dimensional or computationally inexpensive models.

## Testing and Continuous Integration

[Describe the GitHub Actions CI pipeline, unit test coverage fraction, and any regression/verification tests. 
Reference the *github/workflows* directory.]

# Overall Workflow

Figure \autoref{fig:workflow} illustrates the end-to-end UQ pipeline.
At a high level, FESTIM-NIUQ performs five steps:

  1. **Campaign setup**: Parse a YAML configuration file specifying uncertain parameters, their probability distributions, the sampling strategy, and the FESTIM model entry point.
  2. **Ensemble generation**: Use EasyVVUQ to build a parameter ensemble. Instantiate one FESTIM input deck per sample and populate the individual run directories with varied files and links to shared files, with mapping to the original sampling plan.
  3. **Simulation execution**: Run the ensemble sequentially (PC) or in parallel on an HPC cluster via the SEAVEA Toolkit [@groen2021vecmatk].
  4. **Post-processing**: Collect outputs, compute statistical moments (mean, variance) and Sobol sensitivity indices for the selected quantity of interest (QoI). 
  Save the results of the campaign as a SQLite database of runs and a Pickle file of uncertainty analysis results.
  5. **Reporting**: Print moments to the CLI and write publication-ready Sobol-index figures.

![Schematic of the FESTIM-NIUQ uncertainty quantification pipeline.
Dashed lines indicate EasyVVUQ data flow. Solid lines indicate
FESTIM-NIUQ process calls.](figures/flowchart_v1_light.png){#fig:workflow}

# Example Application

## Test Case: Tritium Inventory in a 1-D Slab

We consider a 1-D tungsten slab of thickness $L = 2\,\mathrm{mm}$ subject to a volumetric tritium source $G$ and a fixed concentration boundary condition at $r = a$. The governing transport equation is:

$$
  \frac{\partial c}{\partial t}
  = \nabla\cdot(D\,\nabla c)
    + G
    - \sum_i k_i^+\,c\,(n_i - c_{t,i})
    + \sum_i k_i^-\,c_{t,i},
  \label{eq:transport}$$

where $c$ is the mobile hydrogen concentration, $D$ the diffusion coefficient, $G$ the generation rate, and $c_{t,i}$, $k_i^\pm$ are trap occupancy and rate constants for trap site $i$.

Three parameters are treated as uncertain (uniform distributions): $D$, $G$, and $C(r=a)$.
A PCE of order 3 requires $\binom{3+3}{3} = 20$ FESTIM evaluations to resolve.

## Results

Figure \autoref{fig:results_uncertainty} shows the first-order and total-order Sobol indices and the probability density function of the tritium inventory.
Table \autoref{tab:moments} summarises the statistical moments.

  ![Mean value, standard deviation, confidence intervals, as well as default and analytic verification values of tritium inventory.](figures/sobol_indices.png){#fig:results_uncertainty}

  ![First-order (S1) and total-order (ST) Sobol sensitivity indices for the tritium inventory.](figures/sobol_indices.png){#fig:sobol}

  ![Probability density of the tritium inventory obtained from the PCE surrogate.](figures/qoi_distribution.png){#fig:pdf}

  <!-- ![UQ results for the 1-D tungsten slab test case. [TODO: Update captions with actual quantitative findings]](figures:a.png){#fig:results} -->

: Statistical moments of the tritium inventory (QoI). \label{tab:moments}

| Statistic                | Value | Units     |
|--------------------------|:-----:|:---------:|
| Mean $\mu$               | todo  | $m^{-3}$  |
| Std. deviation, $\sigma$ | todo  | $m^{-3}$   |
| Coefficient of variation | todo  |   --        |

# Research Impact Statement

FESTIM-NIUQ was developed as part of ongoing fusion materials research at the Nuclear Futures Institute at Bangor University, where it is used to assess parametric uncertainties in lithium-ceramic breeder blanket tritium transport simulations, and is a part of the UKAEA LIBRTI programme on breeder blanket technology.
It has been presented at the LIBRTI 2026 Conference on Breeder Blanket Technology [@yudin2026librti], the Open Source Software for Fusion Energy 2026 conference [@yudin2026ossfe].
<!-- and SEAVEA summer hackathon 2025 [@seaveahack2026] -->
It forms the basis for uncertainty-aware studies of tritium trapping and release in Lithium ceramics irradiation experiments at High Flux Accelerator-Driven Neutron Facility [@bishop2024hfadnef] at the University of Birmingham, a partner project of UKAEA.

[TODO: future publications]
[TODO: GitHub activity]

# AI Usage Disclosure

Agentic AI tools (GitHub Copilot, including the copilot-swe-agent) were used during the development of FESTIM-NIUQ.
AI assistance was employed to implement additional functionality, automated testing, documentation, nd project scaffolding after the initial core software was developed by the authors.  
All AI-generated code and documentation were reviewed and validated by the human authors for correctness and scientific accuracy.
AI tools (Claude Sonet and Opus 4.7) were used for preparing, brainstorming, drafting, and reviewing the text of the manuscript.

# Acknowledgements

This work was carried out at the Nuclear Futures Institute, Bangor University.  
The work is funded by the UKAEA LIBRTI programme project.
The authors thank the FESTIM development team for their open-source solver and responsive support, as well as members of the SEAVEA consortium for maintaining the VVUQ toolkit used in the work.

# References
