import argparse
import os
import sys
import subprocess
from datetime import datetime
import numpy as np

import pickle
import argparse

# consider import visualisation libraries optional
import matplotlib.pyplot as plt

# Add parent directory to path for custom encoder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import custom YAML encoders
from util.Encoder import YAMLEncoder, AdvancedYAMLEncoder

import chaospy as cp
import easyvvuq as uq

from easyvvuq.actions import Encode, Decode, ExecuteLocal, Actions, CreateRunDirectory
from easyvvuq.actions import QCGPJPool, EasyVVUQBasicTemplate, EasyVVUQParallelTemplate

# local imports
from util.utils import load_config, add_timestamp_to_filename, get_festim_python, validate_execution_setup
from util.plotting import plot_unc_vs_r, plot_unc_qoi, plot_stats_vs_r, plot_unc_vs_t, plot_sobols_vs_t, plot_stats_vs_t


def visualisation_of_results(results, distributions, qois, plot_folder_name, plot_timestamp, runs_info=None):
    """
    Visualize the results of the EasyVVUQ campaign.
    This function is a placeholder for future visualization methods.
    """

    print("Visualizing results...")
    # Plot the results: error bar for each QoI + other plots

    # Create a common timestamp for all plots from this run, if none
    if not plot_timestamp:
        plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a folder to save plots
    # Assumes that by default there is no folder with plots, and creates a new one
    if not os.path.exists("plots_festim_uq_" + plot_timestamp):
        plot_folder_name = "plots_festim_uq_" + plot_timestamp
        os.makedirs(plot_folder_name)
        # Save plots in this folder

    # Read the vertices from the results
    vertices = results.describe('x', 'mean')  # Assuming 'x' is the vertex coordinate in results
    if vertices is None:
        print("No vertices found in the results. Using a simple range for plotting.")
        rs = np.linspace(0., 1., len(qois))  # Assuming a simple range for x-axis - false, qois is number of checkpoints + 1
    else:
        rs = vertices
    # TODO mesh can be individual for each QoI, potentially each simulation, so read it from the results

    # Read runs from database to a list
    runs_info = list(runs_info) if runs_info is not None else None

    # Plotting statistics of the results as a function of radius (spatial coordinates)
    plot_stats_vs_r(results, qois[1:], plot_folder_name, plot_timestamp, rs=rs, runs_info=runs_info)

    # # Bespoke plotting of uncertainty and Sobol indices in QoI as a function of TIME - not for steady state simulations
    # plot_stats_vs_t(results, distributions, qois[1:], plot_folder_name, plot_timestamp, rs=rs)

    print(f"Plots saved in folder: {plot_folder_name}")
    return 0

def define_parameter_uncertainty(config, CoV=None, distribution=None):
    """
    Define the uncertain parameters and their distributions for the FESTIM model and EasyVVUQ uncertainty propagations.
    """

    print(f"Defining uncertain parameters with CoV={CoV} and distributions={distribution}")

    # # Trial 1) Vary parameter of Arrhenius law
    # parameters_used = ['D_0', 'E_D', 'T_0',]
    # # - add BC and Sources
    # parameters_used += ['source_concentration_value', 'right_bc_concentration_value']

    # Trial 2) Vary parameters for thermal conduction
    #parameters_used = ['thermal_conductivity']

    # Trial 3) Vary parameters for coupled gas and heat transport: 2 problems x (1 tr. coefficients + 1 const source + 1 BC reaction parameter)
    parameters_used = ['D_0', 'kappa','G', 'Q', 'E_kr', 'h_conv']

    # Define default mean values for parameters
    material_num = config.get('geometry', None).get('domains', None)[0].get('material', None)

    # - These values can be adjusted based on the specific model requirements and the physical properties of the system being simulated.
    means = {

        "D_0": config.get('materials', None)[material_num-1].get('D_0', None).get('mean', None),  # Diffusion coefficient base value
        "kappa": config.get('materials', None)[material_num-1].get('thermal_conductivity', None).get('mean', None),  # Thermal conductivity
        "G": config.get('source_terms', None).get("concentration", None).get('value', None).get('mean', None),  # Gas concentration source term
        "Q": config.get('source_terms', None).get('heat', None).get('value', None).get('mean', None),  # Heat source term
        "E_kr": config.get('boundary_conditions', None).get("concentration", None).get("right", None).get('E_kr', None).get('mean', None),  # Reaction rate coefficient
        "h_conv": config.get('boundary_conditions', None).get("temperature", None).get("right", None).get('h_conv', None).get('mean', None),  # Convective heat transfer coefficient


        # "E_D": config.get('materials', None)[material_num-1].get('E_D', None).get('mean', None),  # Activation energy
        # "T_0": config.get('initial_conditions', None).get('temperature', None).get("value", None).get('mean', None),  # Mean temperature

        # "source_concentration_value": config.get('source_terms', None).get('concentration', None).get('mean', None),  # Mean source value
        # "left_bc_concentration_value": config.get('boundary_conditions', None).get('concentration', None).get('left', None).get('mean', None),  # Mean left boundary condition value
        # "right_bc_concentration_value": config.get('boundary_conditions', None).get('concentration', None).get('right', None).get('mean', None),  # Mean right boundary condition value
    }
    # TODO read means and default from the configuration file - alternatively, parse the whole YAML UQ file and get the means from there
    print(f" >>> Mean values for uncertain parameters: {means}") ###DEBUG

    # Define standard deviations for the parameters
    if CoV is not None:
        # use a single defined Coefficient of Variation (CoV) for all parameters, e.g. for a scan
        if CoV < 0.0 or CoV > 1.0:
            raise ValueError("Coefficient of variation (CoV) must be in the range [0, 1].")
        
        relative_stds = {name: CoV for name in parameters_used}  # Create a dictionary with the same CoV for all parameters
    else:
        # parse the YAML UQ file to get the CoV for each parameter
        relative_stds = {
            "D_0": config.get('materials', None)[material_num-1].get('D_0', None).get('relative_stdev', None),  # Diffusion coefficient base value
            "kappa": config.get('materials', None)[material_num-1].get('thermal_conductivity', None).get('relative_stdev', None),  # Thermal conductivity
            "G": config.get('source_terms', None).get("concentration", None).get('value', None).get('relative_stdev', None),  # Gas concentration source term
            "Q": config.get('source_terms', None).get('heat', None).get('value', None).get('relative_stdev', None),  # Heat source term
            "E_kr": config.get('boundary_conditions', None).get("concentration", None).get("right", None).get('E_kr', None).get('relative_stdev', None),  # Reaction rate coefficient
            "h_conv": config.get('boundary_conditions', None).get("temperature", None).get("right", None).get('h_conv', None).get('relative_stdev', None),  # Convective heat transfer coefficient


            # "E_D": config.get('materials', None)[material_num-1].get('E_D', None).get('relative_stdev', None),  # Activation energy
            # "T_0": config.get('initial_conditions', None).get('temperature', None).get("value", None).get('relative_stdev', None),  # Mean temperature

            # "source_concentration_value": config.get('source_terms', None).get('concentration', None).get('relative_stdev', None),
            # "right_bc_concentration_value": config.get('boundary_conditions', None).get('right', None).get('relative_stdev', None),
        }
    print(f" >>> Relative STDs for uncertain parameters: {relative_stds}") ###DEBUG

    # Define the distributions for uncertain parameters
    if distribution is not None:
        # use a single distribution for all parameters, e.g. for a scan
        distributions = {name: distribution for name in parameters_used}
    else:
        # use the default distributions from the configuration file
        distributions = {
            "D_0": config.get('materials', None)[material_num-1].get('D_0', None).get('pdf', None),  # Diffusion coefficient base value
            "kappa": config.get('materials', None)[material_num-1].get('thermal_conductivity', None).get('pdf', None),  # Thermal conductivity
            "G": config.get('source_terms', None).get("concentration", None).get('value', None).get('pdf', None),  # Gas concentration source term
            "Q": config.get('source_terms', None).get('heat', None).get('value', None).get('pdf', None),  # Heat source term
            "E_kr": config.get('boundary_conditions', None).get("concentration", None).get("right", None).get('E_kr', None).get('pdf', None),  # Reaction rate coefficient
            "h_conv": config.get('boundary_conditions', None).get("temperature", None).get("right", None).get('h_conv', None).get('pdf', None),  # Convective heat transfer coefficient


            # "E_D": config.get('materials', None)[material_num-1].get('E_D', None).get('pdf', None),  # Activation energy
            # "T_0": config.get('initial_conditions', None).get('temperature', None).get("value", None).get('pdf', None),  # Mean temperature

            # "source_concentration_value": config.get('source_terms', None).get('concentration', None).get('pdf', 'normal'),
            # "right_bc_concentration_value": config.get('boundary_conditions', None).get('right', None).get('pdf', 'normal'),
        }
    
    print(f" >>> Distributions for uncertain parameters: {distributions}") ###DEBUG

    # Define coefficients to recalculate distribution defining parameters - absolute bounds of the (uniform) distribution is a function of the mean and CoV
    # - for COV of U[a,b]: STD = (b-a)/sqrt(12) , meaning it should be a=mean*(1-sqrt(3)CoV), b=mean*(1+sqrt(3)CoV)
    expansion_factor_lookup = {
        "normal": 1.0,
        "uniform": np.sqrt(3),
    }

    distribution_lookup = {
        "normal": cp.Normal,
        "uniform": cp.Uniform,
        "lognormal": cp.LogNormal,
        "beta": cp.Beta,
        "gamma": cp.Gamma,
        "exponential": cp.Exponential,
    }

    # Create the distributions for the parameters
    parameters_distributions = { 
            name:  distribution_lookup[distributions[name]](
                (means[name]*(1.-expansion_factor_lookup[distributions[name]]*relative_stds[name])) if distributions[name] == 'uniform' else (means[name]) if distributions[name] == 'normal' else means[name],  # lower bound for uniform, mean for normal
                (means[name]*(1.+expansion_factor_lookup[distributions[name]]*relative_stds[name])) if distributions[name] == 'uniform' else (means[name]*relative_stds[name]) if distributions[name] == 'normal' else means[name]*relative_stds[name],  # upper bound for uniform, mean for normal
                ) 
        for name in parameters_used}
    #TODO: tackle not implemented distributions, e.g. lognormal, beta, gamma, exponential
    #TODO: tackle different distribution specifications, e.g. normal with mean and std, uniform with bounds, etc.

    return parameters_distributions

def define_festim_model_parameters():
    """
    Define the FESTIM model parameters and their default values.
    """

    # Define the model input parameters
    parameters = {

    "D_0": {"type": "float", "default": 1.0e-7,},

    "kappa": {"type": "float", "default": 1.0,},

    "G": {"type": "float", "default": 1.0e+6,},

    "Q": {"type": "float", "default": 1000.0,},

    "E_kr": {"type": "float", "default": 1.0,},

    "h_conv": {"type": "float", "default": 1000.0,},

    # "E_D": {"type": "float", "default": 0.1,},

    # "T_0": {"type": "float", "default": 300.0,},

    # "source_concentration_value": {"type": "float", "default": 1.0e20,},

    # "left_bc_concentration_value": {"type": "float", "default": 0.0},  # Boundary condition value: better to specify centre at r=0.0

    # "right_bc_concentration_value": {"type": "float", "default": 1.0e15},  # Boundary condition value at the right (outer) surface of the sample
    }

    # Define  output parameters / the quantities of interest (QoIs)
    # TODO: read an (example) output file to get the QoI names
    qois = [
        #"tritium_inventory",

        "x", # Mainly for (a) reading the vertices for postprocessing, (b) checking that grid has not changed

        # "t=1.00e-01s",
        # "t=2.00e-01s",
        # "t=5.00e-01s",
        # "t=1.00e+00s",
        # "t=5.00e+00s",
        # "t=1.00e+01s",
        # "t=2.50e+01s",

        "t=steady",  # Steady state value, if simulation for performed for a stationary model

        # "t=final" # Final values extracted from the model
    ]
    
    #print(f"Model parameters defined: {parameters}") ####DEBUG
    #print(f"QoIs defined: {qois}") ####DEBUG
    return parameters, qois

def prepare_execution_command():
    """
    Prepare the execution command for the EasyVVUQ campaign.
    """

    # Get the Python executable and script path - validate and setup environment
    python_exe, script_path = validate_execution_setup()

    # Use the filename that the encoder creates (config.uq.yaml)
    config_suffix = f" --config config.yaml "

    # - Assuring correct environment: 
    # - - running activation command
    env_name = "festim2-env"  # Name of the conda environment to activate
    env_prefix = "" # f"conda activate {env_name} && "

    # - option 1) run via calling the python3 command - be sure that the correct environment is used
    exec_command_line = f"{env_prefix} {python_exe} {script_path} {config_suffix}"
    # - option 2) run using shebang in the python script (as an executable)

    print(f"Execution command line: {exec_command_line}")

    # Execute the script locally - prepare the ExecuteLocal action
    execute = ExecuteLocal(exec_command_line)
    
    return execute

def prepare_uq_campaign(config, fixed_params=None, uq_params=None):
    """
    Prepare the uncertainty quantification (UQ) campaign by creating necessary steps: set-up, parameter definitions, encoders, decoders, and actions.
    """

    # Define the model input and output parameters
    parameters, qois = define_festim_model_parameters()

    # TODO rearange FESTIM data output, figure out how to specify multiple quantities at different times, all vs a coordinate

    # Set up necessary elements for the EasyVVUQ campaign

    # TODO: provide parameters with fixed values for the entire campaign -  could be done: (1) in the file at the harddrive, (2) in the YAML object read by the encoder, (3) as an UQ parameter that has to be set to a new default value (a CopyEncoder + MultiEncoder can be applied for this...)

    # Option 1) Modify the parameters in a hard-drive file - skipping
    # Option 2) Before running the campaign, substitute the parameters in the template YAML file
    
    #print(f" >> Preparing the encoder with parameters: {fixed_params}") ###DEBUG
    
    # Create an Encoder object

    # Option 1): Simple YAML Encoder
    # encoder = YAMLEncoder(
    #     template_fname="festim_yaml.template",
    #     target_filename="config.yaml",
    #     delimiter="$"
    # )

    # Option 2): Advanced YAML Encoder
    encoder = AdvancedYAMLEncoder(
        template_fname="festim_yaml.template",
        target_filename="config.yaml",
        parameter_map={ # TODO: store the YAML schema as a separate file; ideally, read from an existing YAML config file
            # ATTENTION: for lists, pattern will always choose the first element, e.g. for materials, domains
            "D_0": "materials.D_0.mean",
            "kappa": "materials.thermal_conductivity.mean",
            "G": "source_terms.concentration.value.mean",
            "Q": "source_terms.heat.value.mean",
            "E_kr": "boundary_conditions.concentration.right.E_kr.mean",
            "h_conv": "boundary_conditions.temperature.right.h_conv.mean",

            # "E_D": "materials.E_D.mean",
            # "T": "initial_conditions.temperature.value.mean",

            # "source_concentration_value": "source_terms.concentration.source_value",
            # "left_bc_concentration_value": "boundary_conditions.concentration.left.value",
            # "right_bc_concentration_value": "boundary_conditions.concentration.right.value",
            
            "length": "geometry.domains.length", 
        },
        type_conversions={
            "D_0": float,
            "kappa": float,
            "G": float,
            "Q": float,
            "E_kr": float,
            "h_conv": float,

            "E_D": float,
            "T": float,

            "source_concentration_value": float,
            "left_bc_concentration_value": float,
            "right_bc_concentration_value": float, 

            "length": float,
        },
        fixed_parameters=fixed_params,  # Pass the dictionary of parameters to be fixed to the encoder
    )

    # Option 3): Use built-in EasyVVUQ encoder
    # encoder = uq.encoders.JinjaEncoder(
    #     template_fname="festim.template", 
    #     target_filename="config.yaml"
    # )

    # TODO modify the YAML more arbitrartly - pass a parameter value from this function e.g. for the sample length scan

    #print(f"Using encoder: {encoder.__class__.__name__}") ###DEBUG
    print(f"Encoder prepared: {encoder}")

    # Create a decoder object
    # The decoder will read the results from the output file and extract the quantities of interest (QoIs)

    # TODO change the output and decoder to YAML (for UQ derived quantities) or other format (?)

    decoder = uq.decoders.SimpleCSV(
        #target_filename="output.csv", # option for synthetic diagnostics specifically chosen for UQ
        target_filename="results/results_tritium_concentration.txt",  # Results from the base data of a simulation, FESTIM1.4 version of TXT outputs for 1D data
        # target_filename="results/results_tritium_concentration.csv",  # Results from the base data of a simulation, FESTIM 2.0 version of CSV outputs for 1D data
        output_columns=qois
        )
    
    print(f"Decoder prepared: {decoder}")

    # Prepare execution command action
    # This command will be used to run the FESTIM model with the generated configuration file
    execute = prepare_execution_command()

    # Set up actions for the campaign
    actions = Actions(
        CreateRunDirectory('run_dir'),
        Encode(encoder),
        execute,
        Decode(decoder),
    )

    print(f"Sequence of actions prepared: {actions}")

    # Define the campaign parameters and actions
    # This includes the model parameters, quantities of interest (QoIs), and the actions to be performed during the campaign
    # Note: The campaign name can be customized as needed
    # Define the campaign
    campaign_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Campaign timestamp: {campaign_timestamp}")

    campaign = uq.Campaign(
        name=f"festim_campaign_{campaign_timestamp}_",
        params=parameters,
        #qois=qois,
        actions=actions,
    )

    # Define uncertain parameters distributions
    distributions = define_parameter_uncertainty(config)
    print(f"Uncertain parameters distributions defined: {distributions}")

    # Define sampling method and create a sampler for the campaign
    # This sampler will generate samples based on the defined distributions

    if uq_params is not None:
        if 'uq_scheme' in uq_params:
            if uq_params['uq_scheme'] == 'pce':
                # Option A) Polynomial Chaos Expansion (PCE) sampler
                # - define polynomial order for PC expansion
                if 'p_order' in uq_params:
                    p_order = uq_params['p_order']
                else:
                    p_order = 1

                print(f"Using UQ scheme: {uq_params['uq_scheme']} with polynomial order: {p_order}") ###DEBUG
                
                sampler = uq.sampling.PCESampler(
                    vary=distributions,
                    polynomial_order=p_order,
                )

            elif uq_params['uq_scheme'] == 'qmc':
                # Option B) quasi-Monte Carlo sampler
                # - define number of samples
                if 'n_samples' in uq_params:
                    n_samples = uq_params['n_samples']
                else:
                    n_samples = 128  # Default number of samples if not specified

                print(f"Using UQ scheme: {uq_params['uq_scheme']} with number of samples: {n_samples}") ###DEBUG

                sampler = uq.sampling.QMCSampler(
                    vary=distributions,
                    n_mc_samples=n_samples,  # Number of samples to generate
                )

            else:
                raise ValueError(f"Unsupported UQ scheme: {uq_params['uq_scheme']}. Supported schemes are 'pce' and 'qmc'.")
        else:
            raise ValueError("UQ scheme not specified in uq_params. Please provide 'uq_scheme' as either 'pce' or 'qmc'.")
    else:
        # Default to PCE sampler with polynomial order 1 if no UQ parameters are provided
        print("No UQ parameters provided, defaulting to PCE sampler with polynomial order 1.")
        sampler = uq.sampling.PCESampler(
            vary=distributions,
            polynomial_order=1,
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
    # If no pool is generated priorly and passed - create a new one
    if resource_pool is None:

        # Make sure the right parameters are passed to the pool: virtual environment, working directory, etc.
        template = EasyVVUQBasicTemplate()
        template_params = {
            "venv": "/home/yhy25yyp/anaconda3/envs/festim2-env",
            #"venv": "/home/yhy25yyp/workspace/festim2-venv/",
            # "working_directory": "/path/to/working/directory"
        }

        # By default, run with resource pool by QCG-PJ
        resource_pool = QCGPJPool(
            template=template,
            template_params=template_params,
        )

    # Execute the campaign
    with resource_pool as pool:

        print(f"> Running the campaign with resource pool: {pool}")

        campaign_results = campaign.execute(pool=pool)

        print("> Execution completed! Collating the results...")

        campaign_results.collate()

    # Get results from the campaign
    # results = campaign_results.get_collation_results()

    return campaign, campaign_results

def analyse_uq_results(campaign, qois, sampler, uq_params=None):
    """
    Perform analysis on the UQ results.
    This function is a placeholder for future analysis methods.
    """
    if uq_params is not None:
        if 'uq_scheme' in uq_params:
            print(f"Performing analysis for UQ scheme: {uq_params['uq_scheme']}") ###DEBUG
            if uq_params['uq_scheme'] == 'pce':
                # Perform PCE analysis on the campaign results
                analysis = uq.analysis.PCEAnalysis(sampler=sampler, qoi_cols=qois)
            elif uq_params['uq_scheme'] == 'qmc':
                # Perform QMC analysis on the campaign results
                analysis = uq.analysis.QMCAnalysis(sampler=sampler, qoi_cols=qois)
            else:
                raise ValueError(f"Unsupported UQ scheme: {uq_params['uq_scheme']}. Supported schemes are 'pce' and 'qmc'.")
        else:
            print("UQ scheme not specified in uq_params. Proceeding with default analysis.")
    else:
        print("No UQ parameters provided. Proceeding with default analysis.")

    #TODO Probably get last analysis results from the campaign
    campaign.apply_analysis(analysis)

    # Get the last analysis results
    results = campaign.get_last_analysis()

    print(f"\n >>> Analysis completed. Results:\n{results}") ###DEBUG

    # Save the analysis results to a file
    result_filename_base = "analysis_results_uq_campaign.pickle"
    results_filename = add_timestamp_to_filename(result_filename_base)
    print(f">> Saving the campaign results into {results_filename}") ###DEBUG
    pickle.dump(results, open(results_filename, "wb"))

    # Display the results of the analysis
    for qoi in qois[1:]:
        print(f"Results for {qoi}:")
        print(results.describe(qoi))
        print("\n")

    #TODO extract more data, in particular, on the individual trajectories of the QoI

    # TODO: specify the results filename in the campaign or save it in a specific folder
    # TODO: save the results of a campaign
    # TODO: add config files and parameters distriubtions to the saved results

    return results

def perform_uq_festim(fixed_params=None):
    """
    Main function to perform the UQ campaign for FESTIM.
    This function orchestrates the preparation, execution, and analysis of the UQ campaign.
    """
    # EasyVVUQ script to be executed as a function
    print(" \n ! Starting FESTIM UQ campaign !.. \n")

    # Prepare the UQ campaign
    # This includes defining parameters, encoders, decoders, and actions

    parser = argparse.ArgumentParser(description='Run FESTIM model with YAML configuration')
    
    parser.add_argument('--config', '-c', 
                       default='config.yaml',
                       help='Path to YAML configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    print(f"> Using arguments file: {args.config}")

    # Load configuration from YAML file
    config = load_config(args.config)
    if config is None:
        print("No config file provided, quitting...")
        return
    print(f" > Loaded configuration from: {args.config}")

    # Read the configuration file from the command line argument or use a default one
    
    print(f" >> Passing parameters fixed to the campaign: {fixed_params}") ###DEBUG

    # Define UQ parameters for the campaign
    uq_params = {
        'uq_scheme': 'pce',  # 'pce' or 'qmc'
        'p_order': 1,        # for PCE
        'n_samples': 8,   # for QMC
    }

    campaign, qois, distributions, campaign_timestamp, sampler = prepare_uq_campaign(config, fixed_params=fixed_params, uq_params=uq_params)

    # TODO: add more parameter for Arrhenious law (+)
    # TODO: try higher BC concentration values - does it even make sense to have such low BC (+)
    # TODO: run with higher polynomial degree (+: now 2)
    # TODO: check if there are actually negative concentrations, if yes - check model and specify correct params ranges - work out expressions for quantiles for unfiorm distribution based on PCE(p=2) (+)

    # Run the campaign
    campaign, campaign_results = run_uq_campaign(campaign)

    #campaign.campaign_db.save(results_filename)
    campaign.campaign_db.dump()

    # Perform the analysis - also saves a Pickle file with results
    results = analyse_uq_results(campaign, qois, sampler, uq_params=uq_params)

    # Save campaign configuration and parameters distributions to a YAML file
    config_filename = add_timestamp_to_filename("uq_campaign_config.pickle")
    pickle.dump(config, open(config_filename, "wb"))
    print(f" >> Campaign configuration saved to: {config_filename}")

    # Get the individual results from the campaign
    runs = campaign.campaign_db.runs() # return an iterator over runs in the campaign

    # print(f">> Iterating over runs in the campaign DB") ###DEBUG
    # for run in runs:
    #     print(f" >> Runs: {run}") ###DEBUG

    # Visualize the results
    visualisation_of_results(results, distributions, qois, "plots_festim_uq_" + campaign_timestamp, plot_timestamp=campaign_timestamp, runs_info=runs)

    print("FESTIM UQ campaign completed successfully!")


if __name__ == "__main__":
    """
    Main entry point for the script.
    This will execute the UQ campaign when the script is run directly.
    """
    try:
        perform_uq_festim()
    except Exception as e:
        print(f"An error occurred during the UQ campaign: {e}")
        sys.exit(1)

##################################
# TODO:

# 0) Double check everything and clean up the code

# 6*) Add convergence (to the steady state) as a QoI