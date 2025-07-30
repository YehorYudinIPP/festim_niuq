import os
import sys
import subprocess
from datetime import datetime
import numpy as np

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
from easyvvuq.actions import QCGPJPool

# local imports
from util.utils import add_timestamp_to_filename, get_festim_python, validate_execution_setup
from util.plotting import plot_unc_vs_r, plot_unc_qoi, plot_stats_vs_r, plot_unc_vs_t, plot_sobols_vs_t, plot_stats_vs_t


def visualisation_of_results(results, distributions, qois, plot_folder_name, plot_timestamp):
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

    # Plotting statistics of the results as a function of radius (spatial coordinates)
    plot_stats_vs_r(results, qois[1:], plot_folder_name, plot_timestamp, rs=rs)

    # Bespoke plotting of uncertainty and Sobol indices in QoI as a function of TIME
    plot_stats_vs_t(results, distributions, qois[1:], plot_folder_name, plot_timestamp, rs=rs)

    print(f"Plots saved in folder: {plot_folder_name}")
    return 0

def define_parameter_uncertainty(CoV=0.1):
    """
    Define the uncertain parameters and their distributions for the FESTIM model and EasyVVUQ uncertainty propagations.
    """

    # define common coefficient of variation (CoV) for all parameters
    if CoV < 0.0 or CoV > 1.0:
        raise ValueError("Coefficient of variation (CoV) must be in the range [0, 1].")
    
    print(f"Defining uncertain parameters with CoV={CoV}")

    # Define default mean values for parameters
    # - These values can be adjusted based on the specific model requirements and the physical properties of the system being simulated.
    means = {
        "T": 300.0,  # Mean temperature
        "source_concentration_value": 1.0e18, #1.0e19, #,1.0e20,  # Mean source value
        "left_bc_concentration_value": 1.0e15,  # Mean boundary condition value: better to keep right side as the domain boundary and left as centre
        "right_bc_concentration_value": 1.0e17, #1.0e16, #1.0e15,  # Mean boundary condition value
    }

    # Define the distributions for uncertain parameters
    # -  These distributions can be adjusted based on the specific model requirements and the physical properties of the system being simulated.

    # The absolute bounds of the distribution is a function of the mean and CoV
    #  For COV of U[a,b]: STD = (b-a)/sqrt(12) , meaning it should be a=mean*(1-sqrt(3)CoV), b=mean*(1+sqrt(3)CoV)
    expansion_factor_uniform = np.sqrt(3)

    parameters_distributions = {
        #"D_0": cp.Uniform(1.0e-7, 1.0e-5), # Diffusion coefficient base value

        #"E_D": cp.Uniform(0.1, 1.0), # Activation energy

        "T": cp.Uniform(means["T"]*(1.-expansion_factor_uniform*CoV), means["T"]*(1.+expansion_factor_uniform*CoV)), # Temperature [K]

        "source_concentration_value": cp.Uniform(means["source_concentration_value"]*(1.-expansion_factor_uniform*CoV), means["source_concentration_value"]*(1.+expansion_factor_uniform*CoV)), 

        #"left_bc_value": cp.Uniform(means["left_bc_value"]*(1.-CoV), means["left_bc_value"]*(1.+CoV)),  # Boundary condition value: see on choice of left/right BC

        "right_bc_concentration_value": cp.Uniform(means["right_bc_concentration_value"]*(1.-expansion_factor_uniform*CoV), means["right_bc_concentration_value"]*(1.+expansion_factor_uniform*CoV)),  # Boundary condition value at the right (outer) surface of the sample
}
    
    return parameters_distributions

def define_festim_model_parameters():
    """
    Define the FESTIM model parameters and their default values.
    """

    # Define the model input parameters
    parameters = {
    "D_0": {"type": "float", "default": 1.0e-7,},

    "E_D": {"type": "float", "default": 0.2,},

    "T": {"type": "float", "default": 300.0,},

    "source_concentration_value": {"type": "float", "default": 1.0e20,},

    #"left_bc_concentration_value": {"type": "float", "default": 1.0e15},  # Boundary condition value: better to specify centre at r=0.0

    "right_bc_concentration_value": {"type": "float", "default": 1.0e15},  # Boundary condition value at the right (outer) surface of the sample
    }

    # Define  output parameters / the quantities of interest (QoIs)
    # TODO: read an (example) output file to get the QoI names
    qois = [
        #"tritium_inventory",
        "x", # Mainly for (a) reading the vertices for postprocessing, (b) checking that grid has not changed
        "t=1.00e-01s",
        "t=2.00e-01s",
        "t=5.00e-01s",
        "t=1.00e+00s",
        "t=5.00e+00s",
        "t=1.00e+01s",
        "t=2.50e+01s",
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

    # Use the filename that the encoder creates (config.yaml)
    exec_command_line = f"{python_exe} {script_path} --config config.yaml"
    
    print(f"Execution command line: {exec_command_line}")

    # Execute the script locally - prepare the ExecuteLocal action
    execute = ExecuteLocal(exec_command_line)
    
    return execute

def prepare_uq_campaign(fixed_params=None):
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
        parameter_map={
            "D_0": "materials.D_0",
            "E_D": "materials.E_D",
            "T": "model_parameters.T_0",
            "source_concentration_value": "source_terms.source_concentration_value",
            "left_bc_value": "boundary_conditions.left_bc_value",
            "right_bc_concentration_value": "boundary_conditions.right_bc_concentration_value",
            "length": "geometry.length", 
        },
        type_conversions={
            "D_0": float,
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
    # TODO change the output and decoder to YAML (for UQ derived quantities) or other format
    decoder = uq.decoders.SimpleCSV(
        #target_filename="output.csv", # option for synthetic diagnostics specifically chosen for UQ
        target_filename="results/results_tritium_concentration.txt",  # Results from the base data of a simulation
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
    distributions = define_parameter_uncertainty()
    print(f"Uncertain parameters distributions defined: {distributions}")

    # Define sampling method and create a sampler for the campaign
    # This sampler will generate samples based on the defined distributions

    # Here we use a Polynomial Chaos Expansion (PCE) sampler!
    p_order = 2  # Polynomial order for PC expansion

    sampler = uq.sampling.PCESampler(
        vary=distributions,
        polynomial_order=p_order,
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

def analyse_uq_results(campaign, qois, sampler):
    """
    Perform analysis on the UQ results.
    This function is a placeholder for future analysis methods.
    """
    # Perform PCE analysis on the campaign results
    analysis = uq.analysis.PCEAnalysis(sampler=sampler, qoi_cols=qois)
    #TODO Probably get last analysis results from the campaign
    campaign.apply_analysis(analysis)

    # Get the last analysis results
    results = campaign.get_last_analysis()

    # Display the results of the analysis
    for qoi in qois[1:]:
        print(f"Results for {qoi}:")
        print(results.describe(qoi))
        print("\n")

    # Save the analysis results
    analysis_filename = add_timestamp_to_filename("analysis_results.hdf5")

    #print(f"Results saved to: {results_filename}")
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
    print("Starting FESTIM UQ campaign...")

    # Prepare the UQ campaign
    # This includes defining parameters, encoders, decoders, and actions
    
    #print(f" >> Passing parameters to the campaign: {fixed_params}") ###DEBUG
    campaign, qois, distributions, campaign_timestamp, sampler = prepare_uq_campaign(fixed_params=fixed_params)

    # TODO: add more parameter for Arrhenious law
    # TODO: try higher BC concentration values - does it even make sense to have such low BC (+)
    # TODO: run with higher polynomial degree (+: now 2)
    # TODO: check if there are actually negative concentrations, if yes - check model and specify correct params ranges - work out expressions for quantiles for unfiorm distribution based on PCE(p=2)

    # Run the campaign
    campaign, campaign_results = run_uq_campaign(campaign)

    # Save the results
    results_filename = add_timestamp_to_filename("results.hdf5")
    campaign.campaign_db.dump()

    # Perform the analysis
    results = analyse_uq_results(campaign, qois, sampler)

    # Visualize the results
    visualisation_of_results(results, distributions, qois, "plots_festim_uq_" + campaign_timestamp, plot_timestamp=campaign_timestamp)

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