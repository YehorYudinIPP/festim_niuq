import argparse
import os
import sys

import numpy as np
import pickle

from itertools import product

from datetime import datetime

# consider import visualisation libraries optional
import matplotlib.pyplot as plt

# Add parent directory to path for custom encoder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom YAML encoders
from util.Encoder import AdvancedYAMLEncoder

import chaospy as cp
import easyvvuq as uq

from easyvvuq.actions import Encode, Decode, ExecuteLocal, Actions, CreateRunDirectory
from easyvvuq.actions import QCGPJPool

# local imports
from util.utils import load_config, add_timestamp_to_filename, get_festim_python, validate_execution_setup
from util.plotting import UQPlotter


def define_phys_conv_rate(results):
    """
    Define the physical convergence rate based on the results of an UQ campaign.
    The rate is considered as a function of varied (uncertain) parameters.
    """
    print(f"Convergence rate esttimate: not implemented yet!")
    return 0

def visualisation_of_results(results, distributions, qois, plot_folder_name, plot_timestamp):
    """
    Visualize the results of the correlated EasyVVUQ campaign.

    Uses bespoke plotting functions tailored for FDAnalysis results,
    which provide mean/std statistics and derivative-based sensitivity
    indices rather than quantiles or Sobol indices.

    Note on alternative correlated uncertainty propagation methods:
        - Monte Carlo with Cholesky decomposition: Generate correlated samples
          directly from the joint distribution and run full MC propagation.
          Provides exact quantiles but is computationally expensive.
        - Polynomial Chaos with Rosenblatt transform: Use cp.MvNormal with
          PCESampler to get spectral convergence while handling correlations.
          EasyVVUQ supports this via the Rosenblatt transform internally.
        - Latin Hypercube Sampling (LHS) with Iman-Conover method: Efficient
          stratified sampling that preserves rank correlations. Good for
          moderate numbers of samples.
        - Nataf transform: Maps correlated non-Gaussian inputs to independent
          standard normal space. Useful when marginals are non-Gaussian.
        - Copula-based methods: Model dependency structure separately from
          marginal distributions. Flexible for complex dependencies.

    Suggested future visualisation improvements:
        - Scatter plots of correlated input samples to verify correlation structure.
        - Correlation contribution decomposition: show independent vs correlated
          variance contributions to each QoI.
        - Tornado/waterfall charts for derivative-based sensitivity ranking.
        - 2D contour plots of QoI response surfaces over correlated parameter pairs.
    """

    print("Visualizing results...")

    # Create a common timestamp for all plots from this run, if none
    if not plot_timestamp:
        plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a folder to save plots
    if not os.path.exists("plots_festim_uq_corr_" + plot_timestamp):
        plot_folder_name = "plots_festim_uq_corr_" + plot_timestamp
        os.makedirs(plot_folder_name)

    # Read the vertices from the results
    vertices = results.describe('x', 'mean')  # Assuming 'x' is the vertex coordinate in results
    if vertices is None:
        print("No vertices found in the results. Using a simple range for plotting.")
        if len(qois) > 1:
            rs = np.linspace(0., 1., len(results.describe(qois[1], 'mean')))
        else:
            rs = np.linspace(0., 1., 10)
    else:
        rs = vertices

    # Use bespoke correlated plotting method that handles FDAnalysis results properly
    uqplotter = UQPlotter()
    uqplotter.plot_stats_correlated(results, distributions, qois[1:], plot_folder_name, plot_timestamp, rs=rs)

    print(f"Plots saved in folder: {plot_folder_name}")
    return 0

def define_parameter_uncertainty(config, sigma_norm=0.25, corr=0.1):
    """Define uncertain parameters and their distributions for FESTIM model UQ.

    This function creates probability distributions for uncertain model parameters
    and defines correlations between them for use in uncertainty quantification
    campaigns with EasyVVUQ.

    Args:
        config (dict): Configuration dictionary loaded from YAML config file.
        sigma_norm (float, optional): Normalized standard deviation for normal 
            distributions. Must be in range [0, 1]. Defaults to 0.1. Can be reused for Uniform distributions.
            For Uniform distributions, the expansion factor is sqrt(3) to maintain the specified CoV.
            For Normal distributions, the expansion factor is 1.0.
            This parameter is used to scale the uncertainty of all parameters uniformly.
        corr (float, optional): Correlation coefficient (normalised covariance) for all pairs of parameters.
            Must be in range [0, 1]. Defaults to 0.1.

    Returns:
        dict: Dictionary mapping parameter names to chaospy distribution objects.

    Raises:
        ValueError: If sigma_norm is outside the valid range [0, 1].

    Example:
        >>> distributions = define_parameter_uncertainty(config, sigma_norm=0.2)
        >>> print(len(distributions))
        2

    Note:
        The function uses sigma_norm as a coefficient of variation (CoV) to scale the uncertainty
        of all parameters uniformly. For uniform distributions, the expansion
        factor is sqrt(3) to maintain the specified CoV.

    Todo:
        * Add support for custom distribution types per parameter
        * Implement validation for correlation matrix symmetry
    """

    # define common coefficient of variation (CoV) for all parameters
    if sigma_norm < 0.0 or sigma_norm > 1.0:
        raise ValueError("Coefficient of variation (CoV) must be in the range [0, 1].")
    
    print(f"Defining uncertain parameters with CoV={sigma_norm}")

    # Define default mean values for parameters from the YAML configuration file
    material_num = config.get('geometry', None).get('domains', None)[0].get('material', None)

    means = {
        "D_0": config.get('materials', None)[material_num-1].get('D_0', None).get('mean', None),
        "thermal_conductivity": config.get('materials', None)[material_num-1].get('thermal_conductivity', None).get('mean', None),
    }
    print(f" >>> Mean values for uncertain parameters: {means}") ###DEBUG

    # Define the distributions for uncertain parameters

    # The absolute bounds of the distribution is a function of the mean and CoV
    # - for CoV of U[a,b]: STD = (b-a)/sqrt(12) , meaning it should be a=mean*(1-sqrt(3)CoV), b=mean*(1+sqrt(3)CoV)
    expansion_factor_uniform = np.sqrt(3)
    # TODO define for the Normal distribution: should be 1.0

    parameters_distributions = {
        "D_0": cp.Normal(means["D_0"], means["D_0"]*sigma_norm), # cp.Uniform(1.0e-7, 1.0e-5), # Diffusion coefficient base value - Normal distribution

        #"E_D": cp.Uniform(0.1, 1.0), # Activation energy

        # "T": cp.Uniform(means["T"]*(1.-expansion_factor_uniform*sigma_norm), means["T"]*(1.+expansion_factor_uniform*sigma_norm)), # Temperature [K]

        "thermal_conductivity": cp.Normal(means["thermal_conductivity"], means["thermal_conductivity"]*sigma_norm), # Temperature [K] - Normal distribution

        # "source_concentration_value": cp.Uniform(means["source_concentration_value"]*(1.-expansion_factor_uniform*sigma_norm), means["source_concentration_value"]*(1.+expansion_factor_uniform*sigma_norm)), 

        #"left_bc_value": cp.Uniform(means["left_bc_value"]*(1.-sigma_norm), means["left_bc_value"]*(1.+sigma_norm)),  # Boundary condition value: see on choice of left/right BC

        # "right_bc_concentration_value": cp.Uniform(means["right_bc_concentration_value"]*(1.-expansion_factor_uniform*sigma_norm), means["right_bc_concentration_value"]*(1.+expansion_factor_uniform*sigma_norm)),  # Boundary condition value at the right (outer) surface of the sample
    }

    param_names = list(parameters_distributions.keys())
    n_params = len(param_names)

    # Create a mean vector for the multivariate distribution
    mean_vector = np.zeros(n_params)
    for i, param in enumerate(param_names):
        mean_vector[i] = means[param]

    print(f" > Mean vector: {mean_vector}") ###DEBUG

    # Create a standard deviation vector for the multivariate distribution
    std_vector = np.zeros(n_params)
    for i, param in enumerate(param_names):
        std_vector[i] = means[param] * sigma_norm  # Standard deviation is CoV * mean

    print(f" > Standard deviation vector: {std_vector}") ###DEBUG

    print(f" > Number of uncertain parameters defined: {n_params}") ###DEBUG

    # Define correlations between parameters

    # Check if correlation coefficient is in the valid range
    if corr < 0.0 or corr > 1.0:
        raise ValueError("Correlation coefficient must be in the range [0, 1].")
    print(f"Defining correlations between parameters with correlation={corr}")

    # Make a dictionary for the correlations between parameters
    parameter_correlations = {(x, y): 0.0 for x, y in product(parameters_distributions.keys(), repeat=2)}

    # Define correlations between specific pairs of parameters
    parameter_correlations[('D_0', 'thermal_conductivity')] = corr  # correlation between D_0 and thermal conductivity
    parameter_correlations[('thermal_conductivity', 'D_0')] = parameter_correlations[('D_0', 'thermal_conductivity')]  # symmetric matrix

    #TODO: to assure symmetry of the correlation matrix: (1) use frozenset as key pair, (2) add a validation function to check that the matrix is symmetric

    # Create correlation matrix as numpy array
    covariance_matrix = np.zeros((n_params, n_params))
    
    # Fill the correlation matrix - TODO there is a more Pythonic way to do this, with double list comprehension and a separate diagonal matrix specification
    for i, (p1, p2) in enumerate(product(param_names, repeat=2)):
        covariance_matrix[i // n_params, i % n_params] = parameter_correlations.get((p1, p2), 1.0) * (std_vector[i // n_params] * std_vector[i % n_params])  # Covariance is correlation * std_dev1 * std_dev2

    # For the covariance matrix, the diagonal elements should be the variances of the parameters - should be covered by the loop above
    for i in range(n_params):
        covariance_matrix[i, i] = (std_vector[i])**2  # Variance is std_dev^2

    print(f" > Covariance matrix:\n{covariance_matrix}") ###DEBUG

    # Make a ChaosPy multivariate distribution
    parameters_distributions_joint = cp.MvNormal(mean_vector, covariance_matrix) # uses Rosenblatt transform to generate samples from the multivariate normal distribution down the line

    return parameters_distributions, parameters_distributions_joint

def define_festim_model_parameters(config):
    """
    Define the FESTIM model parameters and the QoI columns for EasyVVUQ decoding.
    """

    # Define the model input parameters
    parameters = {
    "D_0": {"type": "float", "default": 1.0e-7,}, # Diffusion coefficient base value [m^2/s]

    "E_D": {"type": "float", "default": 0.2,}, # Activation energy [eV]

    "T": {"type": "float", "default": 300.0,}, # Temperature [K]

    "rho": {"type": "float", "default": 1.0e+3,},  # Density of the material [g/cm^3]

    "thermal_conductivity": {"type": "float", "default": 1.0,}, # Thermal conductivity [W/(m*K)]

    "heat_capacity": {"type": "float", "default": 1.0,}, # Heat capacity [J/(kg*K)]

    "source_concentration_value": {"type": "float", "default": 1.0e20,}, # Source term value [atoms/m^3]

    "left_bc_concentration_value": {"type": "float", "default": 1.0e15},  # Boundary condition value: better to specify centre at r=0.0

    "right_bc_concentration_value": {"type": "float", "default": 1.0e15},  # Boundary condition value at the right (outer) surface of the sample
    }

    milestone_times = config.get('simulation', {}).get('milestone_times', [])
    qois = ["x"]
    qois.extend([f"t={float(t):.2e}s" for t in milestone_times])

    if len(qois) == 1:
        # Fallback for steady-state-only configurations.
        qois.append("t=steady")

    return parameters, qois

def prepare_execution_command():
    """
    Prepare the execution command for the EasyVVUQ campaign.
    """

    # Get the Python executable and script path - validate and setup environment
    python_exe, script_path = validate_execution_setup()

    # Use the filename that the encoder creates (config.yaml)
    exec_command_line = f"{python_exe} {script_path} --config config.yaml"
    
    print(f"Execution command line: {exec_command_line}")

    # Execute the script locally - prepare the ExecuteLocal action
    execute = ExecuteLocal(exec_command_line)
    
    return execute

def prepare_uq_campaign(config, config_file, fixed_params=None):
    """
    Prepare the uncertainty quantification (UQ) campaign by creating necessary steps: set-up, parameter definitions, encoders, decoders, and actions.
    """

    # Define the model input and output parameters
    parameters, qois = define_festim_model_parameters(config)

    # Set up necessary elements for the EasyVVUQ campaign

    # Before running the campaign, substitute the fixed parameters in the template YAML file
    
    # Create an Encoder object - Advanced YAML Encoder
    encoder = AdvancedYAMLEncoder(
        template_fname=config_file,
        target_filename="config.yaml",
        parameter_map={
            "D_0": "materials.D_0.mean",
            "thermal_conductivity": "materials.thermal_conductivity.mean",

            # "E_D": "materials.E_D.mean",
            # "T": "initial_conditions.temperature.value.mean",

            # "source_concentration_value": "source_terms.concentration.source_value",
            # "left_bc_concentration_value": "boundary_conditions.concentration.left.value",
            # "right_bc_concentration_value": "boundary_conditions.concentration.right.value",

            "length": "geometry.domains.length", 
        },
        type_conversions={
            "D_0": float,
            "thermal_conductivity": float,
            "E_D": float,
            "T": float,
            "source_concentration_value": float,
            "left_bc_concentration_value": float,
            "right_bc_concentration_value": float, 
            "length": float,
        },
        fixed_parameters=fixed_params,  # Pass the dictionary of parameters to be fixed to the encoder
    )

    print(f"Encoder prepared: {encoder}") ###DEBUG

    # Create a decoder object
    # The decoder will read the results from the output file and extract the quantities of interest (QoIs)
    # TODO change the output and decoder to YAML (for UQ derived quantities) or other format
    decoder = uq.decoders.SimpleCSV(
        #target_filename="output.csv", # option for synthetic diagnostics specifically chosen for UQ
        target_filename="results/results_tritium_concentration.txt",  # Results from the base data of a simulation
        output_columns=qois
        )
    
    print(f"Decoder prepared: {decoder}") ###DEBUG

    # Prepare execution command action
    # This command will be used to run the FESTIM model with the generated configuration file
    execute = prepare_execution_command()

    # Set up actions for the campaign
    actions = Actions(
        CreateRunDirectory('run_dir'), # TODO figure out how not to use it
        Encode(encoder),
        execute,
        Decode(decoder),
    )

    print(f"Sequence of actions prepared: {actions}") ###DEBUG

    # Define the campaign parameters and actions
    # This includes the model parameters, quantities of interest (QoIs), and the actions to be performed during the campaign
    campaign_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Campaign timestamp: {campaign_timestamp}")

    campaign = uq.Campaign(
        name=f"festim_campaign_corr_{campaign_timestamp}_",
        params=parameters,
        actions=actions,
    )

    # Define uncertain parameters distributions
    distributions, distributions_joint = define_parameter_uncertainty(config)
    print(f"Uncertain parameters distributions defined: {distributions}")

    # Define sampling method and create a sampler for the campaign - this sampler will generate samples based on the defined distributions

    # Here we use a Finite Difference (FD) surrogate sampler using the covariance matrix
    #p_order = 2  # Polynomial order for the expansion

    sampler = uq.sampling.FDSampler(
        vary=distributions,
        distribution=distributions_joint,  # Use the joint distribution for sampling
        relative_analysis=True,  # Use relative analysis for the sampler
    )

    campaign.set_sampler(sampler)
    print(f"Sampler prepared and set for the campaign: {sampler}")

    print(f"Campaign and its elements are prepared!")
    return campaign, qois, distributions, campaign_timestamp, sampler

def run_uq_campaign(campaign, resource_pool=None):
    """
    Run the UQ campaign using the specified resource pool.
    If no resource pool is provided, it will use the default QCGPJPool.
    """
    if resource_pool is None:
        resource_pool = QCGPJPool()

    # Execute the campaign
    with resource_pool as pool:
        campaign_results = campaign.execute(pool=pool)
        campaign_results.collate()

    # Get results from the campaign
    # results = campaign_results.get_collation_results()

    return campaign, campaign_results

def analyse_uq_results(campaign, params, qois, sampler):
    """
    Perform analysis on the UQ results.

    Note: FDAnalysis computes mean, variance, std and derivative-based sensitivity
    indices. Sobol indices and percentiles are NOT computed by FDAnalysis and will
    be zero — use derivatives_first for sensitivity information instead.
    """
    # Perform FD analysis on the campaign results
    analysis = uq.analysis.FDAnalysis(sampler=sampler, qoi_cols=qois)

    #TODO Better - get last analysis results from the campaign
    campaign.apply_analysis(analysis)

    # Get the last analysis results
    results = campaign.get_last_analysis()

    # Display the results of the analysis
    for qoi in qois[1:]:
        print(f"Results for {qoi}:")

        print(f"Mean:", results.describe(qoi, 'mean'))
        print(f"Standard Deviation:", results.describe(qoi, 'std'))

        # Note: FDAnalysis does not compute percentiles — they remain as zero arrays.
        # Use mean +/- k*std for approximate confidence intervals instead.

        for param in params.keys():
            print(f"Parameter: {param}")

            # Derivative-based sensitivity (the primary FD measure)
            print(f"Derivative first indices: {results._get_derivatives_first(qoi, param)}")

            # Note: Sobol indices from FDAnalysis are zero — they are not computed
            # by the finite-difference method. Use derivatives_first instead.

        print("\n")

    # Save the analysis results
    analysis_filename = add_timestamp_to_filename("analysis_results.hdf5")

    #print(f"Results saved to: {results_filename}")
    # TODO: specify the results filename in the campaign or save it in a specific folder
    # TODO: save the results of a campaign
    # TODO: add config files and parameters distriubtions to the saved results

    return results

def perform_uq_festim_correlated_params(config=None, fixed_params=None):
    """
    Main function to perform the UQ campaign for FESTIM with correlated parameters.
    This function orchestrates the preparation, execution, and analysis of the UQ campaign.
    """
    # EasyVVUQ script to be executed as a function
    print(" \n ! Starting FESTIM UQ campaign (correlated) !.. \n")
    print(f" time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prepare the UQ campaign
    # This includes defining parameters, encoders, decoders, and actions

    config_file_path = "config.yaml"

    # Check if config exists
    if config is None:

        print("No config file provided, reading from arguments")

        # Read the configuration file from the command line argument or use a default one
        parser = argparse.ArgumentParser(description='Run FESTIM model with correlated UQ using YAML configuration')

        parser.add_argument('--config', '-c',
                        default='config.yaml',
                        help='Path to YAML configuration file (default: config.yaml)')

        args = parser.parse_args()
        config_file_path = args.config
        print(f"> Using arguments file: {config_file_path}")

        # Load configuration from YAML file
        config = load_config(config_file_path)

        print(f" > Loaded configuration from: {config_file_path}")

    if config is None:
        print("No config file provided, quitting...")
        return

    print(f" >> Passing parameters fixed to the campaign: {fixed_params}") ###DEBUG

    campaign, qois, distributions, campaign_timestamp, sampler = prepare_uq_campaign(config, config_file=config_file_path, fixed_params=fixed_params)

    # TODO: add more parameter for Arrhenious law
    # TODO: try higher BC concentration values - does it even make sense to have such low BC (+)
    # TODO: run with higher polynomial degree (+: now 2)
    # TODO: check if there are actually negative concentrations, if yes - check model and specify correct params ranges - work out expressions for quantiles for unfiorm distribution based on PCE(p=2)

    # Run the campaign
    campaign, campaign_results = run_uq_campaign(campaign)

    # Save the results
    results_filename = add_timestamp_to_filename("results.hdf5")
    campaign.campaign_db.dump()

    # Save campaign configuration and parameters distributions to a Pickle file
    config_filename = add_timestamp_to_filename("uq_campaign_config_corr.pickle")
    pickle.dump(config, open(config_filename, "wb"))
    print(f" >> Campaign configuration saved to: {config_filename}")

    # Perform the analysis
    results = analyse_uq_results(campaign, distributions, qois, sampler)

    # Visualize the results
    visualisation_of_results(results, distributions, qois, "plots_festim_uq_corr_" + campaign_timestamp, plot_timestamp=campaign_timestamp)

    print("FESTIM UQ campaign completed successfully!")
    return 0


if __name__ == "__main__":
    """
    Main entry point for the script.
    This will execute the UQ campaign when the script is run directly.
    """
    try:
        # Simple case: run with default parameters
        # perform_uq_festim_correlated_params()

        # More complex case: run with fixed parameters
        fixed_params = {
            "length": 5.e-4,  # Length of the sample in meters
        }
        perform_uq_festim_correlated_params(fixed_params=fixed_params)

    except Exception as e:
        print(f"An error occurred during the UQ campaign: {e}")
        sys.exit(1)

##################################
# TODO:
