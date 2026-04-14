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
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Tom Griffiths
    affiliation: 1
  - name: Tom Smith
    affiliation: 1
  - name: Tessa Davey
    affiliation: 1
  - name: Cillian Cockrell
    affiliation: 1
  - name: Simon Middleburgh
    affiliation: 1
affiliations:
  - name: Nuclear Futures Institute, Bangor University, United Kingdom
    index: 1
date: 14 April 2026
bibliography: paper.bib
---

# Summary

FESTIM-NIUQ is a Python package that performs non-intrusive uncertainty
quantification (UQ) on tritium transport simulations modelled with the
FESTIM finite-element framework. The package couples FESTIM with EasyVVUQ
to propagate parametric uncertainties through diffusion–reaction models
of tritium behaviour in fusion-relevant materials.  Given user-specified
probability distributions on input parameters such as diffusivity,
thermal conductivity, and source rates, FESTIM-NIUQ automatically
generates parameter samples, executes FESTIM simulations, and computes
statistical moments and Sobol sensitivity indices of quantities of
interest, including spatially resolved concentration profiles and
integral tritium inventories.

<!-- TODO: Expand this summary to 150-250 words for a non-specialist audience. -->

# Statement of Need

Tritium — a radioactive isotope of hydrogen — must be carefully
accounted for in fusion reactor components such as lithium-ceramic
breeder blankets.  The tritium concentration field is governed by a
diffusion–reaction equation with Arrhenius-type transport coefficients
whose values carry substantial experimental uncertainty.  Quantifying
how this input uncertainty propagates to predicted tritium inventories
and release rates is essential for reactor safety assessments, yet
performing systematic UQ studies on finite-element tritium transport
models remains a labour-intensive task that requires expertise in both
nuclear materials science and statistical methods.

FESTIM-NIUQ addresses this gap by providing a turnkey UQ pipeline
specifically designed for FESTIM users.  The target audience is fusion
materials scientists and nuclear engineers who need to assess the
reliability of their tritium transport predictions without modifying the
underlying solver.  By automating parameter sampling, model execution,
result collection, and sensitivity analysis, FESTIM-NIUQ lowers the
barrier to rigorous UQ in the fusion materials community.

<!-- TODO: Add references to relevant fusion safety and UQ literature. -->

# State of the Field

Several general-purpose UQ frameworks exist, including Dakota
[@adams2014dakota], OpenTURNS [@baudin2017openturns], UQLab
[@marelli2014uqlab], and SALib [@herman2017salib].  While powerful,
these tools do not provide out-of-the-box integration with FESTIM and
require significant effort to set up custom encoder/decoder pipelines
for finite-element tritium transport problems.

EasyVVUQ [@richardson2020easyvvuq] from the SEAVEA Toolkit provides a
flexible VVUQ workflow engine but still requires users to write bespoke
glue code for each solver.  FESTIM-NIUQ fills this niche by providing
pre-built YAML-based encoders, CSV decoders, and plotting utilities that
are tailored to the FESTIM data model, enabling users to run UQ campaigns
with a single command and configuration file.

<!-- TODO: Expand comparison and add "build vs. contribute" justification. -->

# Software Design

FESTIM-NIUQ adopts a non-intrusive architecture in which the FESTIM
solver is treated as a black box.  This design choice was made to
decouple the UQ layer from the solver internals, allowing users to
upgrade FESTIM independently and to apply the same UQ machinery to
different transport models without code changes.

The package consists of three layers:

1. **Model wrapper** (`festim_model/`): Encapsulates FESTIM model
   configuration, execution, and result export.
2. **UQ orchestration** (`uq/`): Manages parameter sampling, campaign
   execution, and analysis using EasyVVUQ and ChaosPy.
3. **Utilities** (`uq/util/`): Custom YAML encoders that perform deep
   nested parameter substitution via dot-notation paths, CSV decoders,
   and publication-quality plotting routines.

The `AdvancedYAMLEncoder` replaces parameter values at arbitrary nesting
depths in YAML configuration files without requiring Jinja templates,
which simplifies the workflow and reduces the risk of template syntax
errors.  Support for correlated parameters is provided via Cholesky
decomposition of user-specified covariance matrices.

<!-- TODO: Discuss HPC integration (QCG-PilotJob, SLURM) and design trade-offs. -->

# Research Impact Statement

FESTIM-NIUQ has been presented at the UKAEA Technical Meeting
[@ukaea2026meeting] and the Open-Source Software for Fusion Energy
Workshop (OSSFE 2026) [@ossfe2026].  The package is actively used in
ongoing research at the Nuclear Futures Institute, Bangor University, to
assess parametric uncertainties in lithium-ceramic breeder blanket tritium
transport simulations.

Verification of the solver wrapper has been performed against Carslaw and
Jaeger's analytical solutions for diffusion in a sphere
[@carslaw1959conduction], and convergence of the PCE surrogate has been
verified with increasing polynomial order.

<!-- TODO: Add any additional publications, external users, or community engagement evidence. -->

# AI Usage Disclosure

Agentic AI tools (GitHub Copilot, including the copilot-swe-agent) were
used during the development of FESTIM-NIUQ.  AI assistance was employed
to implement additional functionality, automated testing, documentation,
and project scaffolding after the initial core software was developed by
the authors.  All AI-generated code and documentation were reviewed and
validated by the human authors for correctness and scientific accuracy.

# Acknowledgements

<!-- TODO: Add funding sources and institutional acknowledgements. -->

This work was carried out at the Nuclear Futures Institute, Bangor
University.

# References
