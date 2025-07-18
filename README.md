# FESTIM-NIUQ

A package of tools and scripts to perform Non-Intrusive Uncertainty Quantification for FESTIM code.

The package uses [EasyVVUQ](https://github.com/UCL-CCS/EasyVVUQ) and [SEAVEA-Toolkit](http://www.seaveatk.org/) to perform parametric uncertainties propagation for a tritium transport problem modelled using [FESTIM](https://github.com/festim-dev/FESTIM) framework.

## Installation

The usage requires the following non-standard packages:

 - FESTIM
 - EasyVVUQ

 ### Virtual environments

To run the scripts you will need to create the respective virtual enviroments.

The easiest way is to create a single one with all the dependencies and run all the script in it.

To make one, run:

```
python3 -m venv festim_niuq
```

And to activate it, run:

```
source festim_niuq/bin/activate
```

### Set-up

To install the prerequisites run:

```
pip3 install FESTIM==1.4
pip3 install easyvvuq

```

Alternatively, you can install all dependencies in batch:

```
pip3 install -r requirements.txt
```

## Usage

Turn-key usage for a test is:

```
cd uq
python3 easyvvuq_festim.py 
```

This should show moments of a selected QoI (tritium inventory) in the CLI output and generate plot for Sobol indices for the respective model parameters (diffusion coefficient $$D$$, generation rate $$G$$, concentration at boundary $$C(r=a)$$)
