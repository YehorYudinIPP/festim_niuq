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
    affiliation: 1
  - name: Cillian Cockrell
    orcid: 0000-0002-8501-7287
    affiliation: 1
  - name: Simon Middleburgh
    orcid: 0000-0003-2537-4001
    affiliation: 1
affiliations:
  - name: Nuclear Futures Institute, Bangor University, United Kingdom
    index: 1
date: 14 April 2026
bibliography: paper.bib
---

# Summary

Tritium is a radioactive isotope of hydrogen that will be both the fuel and a safety-critical inventory item in future fusion power plants.
Accurate predictions of tritium transport through fusion reactor component, such as lithium-ceramic breeder blankets and plasma-facing components, are essential for assuring performance of the fuel cycle, as well as assessing safety and reactor design.
Yet the material transport coefficients that govern these predictions carry substantial experimental uncertainty, and their *ab initio* estiamtes might contain uncertainties that ahs to be accounted as well.

FESTIM-NIUQ is a Python package that automates non-intrusive uncertainty quantification (UQ) for tritium transport simulations performed with the FESTIM finite-element framework [@delaporte2019festim; @delaporte2024festim].
The package couples FESTIM with EasyVVUQ [@richardson2020easyvvuq] and ChaosPy [@feinberg2015chaospy] to propagate parametric uncertainties through diffusion–reaction models of tritium behaviour in fusion-relevant materials.
Given user-specified probability distributions on input parameters — such as the diffusion pre-exponential factor, thermal conductivity, volumetric generation rate, and surface recombination energy — FESTIM-NIUQ automatically generates parameter samples, executes FESTIM simulations in parallel, and computes statistical moments and Sobol sensitivity indices [@sobol1993sensitivity] of quantities of interest, including spatially resolved concentration profiles and integral tritium inventories.  
The entire workflow is controlled by a single YAML configuration file, which specifies model equations and tehire terms, geometry and boundary conditions, as well as parameter values - bother determenistic and unecertain - making it accessible to fusion materials scientists who have no prior UQ expertise.

The \textit{0.2.0} version of the package is available as source code at \textit{github.com}, installable from \textit{PyPI} repository, and is archived at \textit{ZENODO}.
The repository covers basic functionality with unit tests, provides several verification cases, and allows users to adapt it to specific needs via permissive licensing.

# Statement of Need

Fusion reactor components, such as tungsten first walls and beryllium covers, interact with tritium over long operational periods.
Tritium must be carefully inventoried and managed in fusion reactor components because of its radioactivity, scarcity, and role as fuel.
Tritium transport is governed by a coupled diffusion–reaction system with Arrhenius-type coefficients whose values are often measured with uncertainties of tens of percent [@causey2012tritium]. 
Reliable safety assessments and design decisions require an understanding of how theseinput uncertainties propagate to predicted tritium inventories and release rates [@humrickhouse2011verification].
Design-basis analyses require not just best-estimate simulations but also confidence intervals on tritium inventory and release rates, since these directly affect safety licensing [{todo_ref1].

## Gap in Existing Tools

Although general-purpose UQ frameworks exist (see *State of the Field* below), coupling any of them to a finite-element tritium transport solver requires non-trivial engineering: writing a solver-specific parameter encoder, an output decoder, a subprocess execution harness, and post-processing and plotting utilities.
This barrier is high enough that most published tritium transport studies report only deterministic results at nominal parameter values, forgoing systematic UQ entirely.

[Cite UQ studies on hydrogen transport that used manual/ad-hoc methods to further motivate automation.]

FESTIM-NIUQ removes this barrier for the FESTIM user community [@delaporte2024festim] by providing a ready-to-use pipeline that handles every step between a YAML configuration file and publication-quality sensitivity-index plots.
The package is designed to be extended: new uncertain parameters, boundary conditions, or coordinate geometries are added by editing the configuration file rather than modifying Python source code.
FESTIM-NIUQ has been used in ongoing research at the Nuclear Futures Institute, Bangor University, to assess parametric uncertainties in lithium-ceramic breeder blanket tritium transport simulations, and it has been presented at the UKAEA Technical Meeting [@ukaea2026meeting] and the Open-Source Software for Fusion Energy Workshop (OSSFE 2026) [@ossfe2026].

Verification of the solver wrapper has been performed against Carslaw and Jaeger's analytical solutions for diffusion in a sphere
[@carslaw1959conduction], and convergence of the polynomial chaos expansion (PCE) surrogate has been confirmed with increasing polynomial order. [TODO: add PCE scaling study]

# State of the Field

Several general-purpose UQ frameworks exist, including Dakota [@adams2014dakota], OpenTURNS [@baudin2017openturns], UQLab [@marelli2014uqlab], SALib [@herman2017salib], and ChaosPy [@feinberg2015chaospy].
While powerful, these tools require users to write bespoke glue code, which includes parameter encoders, solver wrappers, output decoders, - for each specific solver, which is a significant effort for finite-element tritium transport problems involving nested YAML configurations, VTX result files, and subprocess execution.
Each of the specific hydrogen and tritium transport frameworks, alternative codes including  TMAP8 [@simon2025tmap8], TESSIM-X [@schmid2012tessim], SAETTA [@hattab2025saetta] and HIIPC [@sanghiipc], each with different physical scope, model specifics and assumptions, numerical backend, and input formats.
They would require adapting UQ tools for the specific use, however, the experience of applying generic UQ methods for hydrogen transport provides a pathway to adoption of these methods in the field.

EasyVVUQ [@richardson2020easyvvuq] from the SEAVEA Toolkit provides a flexible VVUQ workflow engine and reduces this burden, but the user must still implement solver-specific encoder and decoder classes.
FESTIM-NIUQ fills this niche by providing pre-built YAML-based encoders with deep nested parameter substitution via dot-notation paths, CSV decoders, a subprocess execution harness, and publication-quality plotting routines, all tailored to the FESTIM data model.  The result is that a user can launch a complete UQ campaign with a single command and configuration file without writing any Python code.

Contributing a generic FESTIM integration upstream to EasyVVUQ was considered but rejected because the integration requires FESTIM-specific knowledge of its configuration schema, output file formats, and coordinate-system conventions.  Maintaining it as a standalone package allows independent versioning aligned with FESTIM releases and keeps the EasyVVUQ core free of solver-specific logic.

[TODO: table summary of existing tools: FESTIM integration, PCE support, YAML config, fusion-specific QoIs]

# Software Design

FESTIM-NIUQ adopts a non-intrusive architecture in which the FESTIM
solver is treated as a black box.  This design decouples the UQ layer
from the solver internals, allowing users to upgrade FESTIM independently
and to apply the same UQ machinery to different transport models without
code changes.

The package consists of three layers:

1. **Model wrapper** (`festim_model/`): Encapsulates FESTIM
   [@delaporte2024festim] model configuration, execution, and result
   export for both FESTIM 2.0 (DOLFINx-based) and the legacy FESTIM 1.x
   API.
2. **UQ orchestration** (`uq/`): Manages parameter sampling, campaign
   execution, and analysis using EasyVVUQ and ChaosPy.  Supports
   Polynomial Chaos Expansion (PCE), Quasi-Monte Carlo (QMC), and
   Bayesian inverse UQ via PCE surrogate and MCMC.
3. **Utilities** (`uq/util/`): Custom YAML encoders that perform deep
   nested parameter substitution via dot-notation paths, CSV decoders,
   and publication-quality plotting routines for Sobol indices and
   statistical profile bands.

The `AdvancedYAMLEncoder` replaces parameter values at arbitrary nesting
depths in YAML configuration files without requiring Jinja templates,
which simplifies the workflow and reduces the risk of template syntax
errors.  Support for correlated parameters is provided via Cholesky
decomposition of user-specified covariance matrices.

For high-performance computing environments, FESTIM-NIUQ integrates with
QCG-PilotJob for embarrassingly parallel sample evaluation on cluster
resources, and SLURM submission scripts are included for common HPC
platforms.  On workstations the same campaign runs locally using
`joblib` multiprocessing without any configuration changes, ensuring
reproducibility across environments.

# Overall Workflow

Figure \autoref{fig:workflow} illustrates the end-to-end UQ pipeline.
At a high level, FESTIM-NIUQ performs five steps:

  1. 'Campaign setup': Parse a YAML configuration file
        specifying uncertain parameters, their probability
        distributions, the sampling strategy, and the \festim{} model
        entry point.
  2. Ensemble generation: Use \easyvvuq{} to build a
        parameter ensemble; instantiate one \festim{} input deck per
        sample.
  3. Simulation execution: Run the ensemble sequentially
        (laptop) or in parallel on an HPC cluster via the SEAVEA
        Toolkit~\cite{seavea}.
  4. Post-processing: Collect outputs, compute statistical
        moments (mean, variance) and Sobol sensitivity indices for the
        selected quantity of interest (QoI).
  5. Reporting: Print moments to the CLI and write
        publication-ready Sobol-index figures.

![Schematic of the FESTIM-NIUQ uncertainty quantification pipeline.
Dashed lines indicate EasyVVUQ data flow. Solid lines indicate
FESTIM-NIUQ process calls.](figures/flowchart_v1_light.png){#fig:workflow}

# AI Usage Disclosure

Agentic AI tools (GitHub Copilot, including the copilot-swe-agent) were
used during the development of FESTIM-NIUQ.  AI assistance was employed
to implement additional functionality, automated testing, documentation,
and project scaffolding after the initial core software was developed by
the authors.  All AI-generated code and documentation were reviewed and
validated by the human authors for correctness and scientific accuracy.

# Acknowledgements

This work was carried out at the Nuclear Futures Institute, Bangor
University.  The authors thank the FESTIM development team for their
open-source solver and responsive support.

# References
