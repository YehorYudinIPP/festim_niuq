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

# Import custom YAML encoder
from util.Encoder import YAMLEncoder, AdvancedYAMLEncoder

import chaospy as cp
import easyvvuq as uq

from easyvvuq.actions import Encode, Decode, ExecuteLocal, Actions, CreateRunDirectory
from easyvvuq.actions import QCGPJPool

def add_timestamp_to_filename(filename, timestamp=None):
    """
    Add timestamp to filename before the extension.
    
    Args:
        filename (str): Original filename
        timestamp (str, optional): Custom timestamp string. If None, uses current datetime.
    
    Returns:
        str: Filename with timestamp
    
    Example:
        add_timestamp_to_filename("results.hdf5") -> "results_20250718_143025.hdf5"
    """
    if timestamp is None:
        # Different timestamp formats:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        # 20250718_143025
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # 2025-07-18_14-30-25
        # timestamp = datetime.now().strftime("%Y%m%d")               # 20250718 (date only)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M")          # 20250718_1430 (no seconds)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"

def get_festim_python():
    """Get the correct Python executable for FESTIM environment."""
    # Method 1: Check if specific conda environment exists
    conda_python = "/home/yehor/miniconda3/envs/festim-env/bin/python3"
    if os.path.exists(conda_python):
        return conda_python
    
    # Method 2: Try to find conda environment dynamically
    try:
        result = subprocess.run(['conda', 'info', '--envs'], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if 'festim-env' in line:
                env_path = line.split()[-1]
                python_path = os.path.join(env_path, 'bin', 'python')
                if os.path.exists(python_path):
                    return python_path
    except:
        pass
    
    # Method 3: Fallback to current Python
    print("Warning: FESTIM environment not found, using current Python")
    return sys.executable

def validate_execution_setup():
    """Validate that the execution environment is properly configured."""
    runnable_script = "festim_model_run.py"
    script_path = os.path.join(os.getcwd(), runnable_script)
    
    # Check script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    # Check script is executable
    if not os.access(script_path, os.X_OK):
        print(f"Making script executable: {script_path}")
        os.chmod(script_path, 0o755)
    
    # Check Python executable
    python_exe = get_festim_python()
    if not os.path.exists(python_exe):
        raise FileNotFoundError(f"Python executable not found: {python_exe}")
    
    print(f"✓ Script validation passed: {script_path}")
    print(f"✓ Python executable: {python_exe}")
    return python_exe, script_path

def define_phys_conv_rate(results):
    """
    Define the physical convergence rate based on the results of an UQ campaign.
    The rate is considered as a function of varied (uncertain) parameters.
    """
    print(f"Convergence rate esttimate: not implemented yet!")
    return 0

def plot_unc_vs_r(r, y, sy, y10, y90, qoi_name, foldername="", filename=""):
    """
    Plot uncertainty in the results as a function of radius (spatial coordinates).
    """
    fig, ax = plt.subplots()

    ax.plot(r, y, label=f'<y> at {qoi_name}')
    ax.fill_between(r, y - sy, y + sy, alpha=0.3, label='+/- STD')
    ax.fill_between(r, y10, y90, alpha=0.1, label='10% - 90%')

    ax.set_title(f"Uncertainty at {qoi_name} as a function of radius")
    ax.set_xlabel("#Radius, fraction of length")
    ax.set_ylabel(f"Concentration [m^-3] at {qoi_name}")
    ax.legend()
    ax.grid(True)

    fig.savefig(f"{foldername}/bespoke_{filename}")

    plt.close()  # Close the plot to avoid display issues in some environments

    return 0

def plot_unc_qoi(stats_dict_s, qoi_name, foldername="", filename="", r_ind=0):
    """
    Plot uncertainty in the specific scalar QoIs.
    """

    # Specific to bespoke plot for a list of QoIs
    fig, ax = plt.subplots()

    #Boxplotting the mean and std at a single radius

    #ax.plot(qoi_name, y[r_ind], 'o', label=f"<y> at r={0} and {qoi_name}")

    #ax.errorbar(qoi_name, y[r_ind], yerr=sy[r_ind], fmt='o', label=f"+/- STD at r_ind={r_ind} and {qoi_name}")

    ax.bxp(
        stats_dict_s,
        patch_artist=True,
        showmeans=True,
        shownotches=True,
        #meanline=True,  # Show mean line
        label=f"QoIs at r.ind {r_ind}",
        #label=f"Mean, 95% CI, 10% - 90%, min - max",
    )

    #ax.fill_betweenx([y10[r_ind], y90[r_ind]], qoi - 0.01, qoi + 0.01, alpha=0.1, label=f"10% - 90% at r={0} and {qoi_name}")

    ax.set_ylabel(f"Concentration [m^-3] at radius r index [{r_ind}]")  # Assuming all QoIs have the same units
    ax.set_xlabel(f"Different times of a simulation")
    ax.set_title(f"Uncertainty in QoIs: mean, median, 95% CI, 10% - 90%, min - max")
    ax.legend(loc='best')
    ax.grid(axis='y')

    #fig_qoi.suptitle("Uncertainty in QoIs at selected radius")
    
    fig.savefig(f"{foldername}/{filename}")

    return 0

def plot_stats_vs_r(results, qois, plot_folder_name, plot_timestamp, rs=None):
    """
    Plot statistics of the results as a function of radius (spatial coordinates).
    """

    # Specific for common boxplot for QoIs
    stats_dict_s = []
    r_ind_qoi = 0  # Select the max radius values: r=0.0 should be physical centre of domain
    
    # Rund over QoIs in analysis results object
    for qoi in qois:
        # Generate filenames with timestamp
        moments_vsr_filename = add_timestamp_to_filename(f"{qoi}_moments_vs_r.png", plot_timestamp)
        sobols_treemap_filename = add_timestamp_to_filename(f"{qoi}_sobols_treemap.png", plot_timestamp)
        sobols_filename = add_timestamp_to_filename(f"{qoi}_sobols_first_vs_r.png", plot_timestamp)
        
        # Default plotting of the moments
        results.plot_moments(
            qoi=qoi,
            ylabel=f"Concentration [m^-3], {qoi}",
            xlabel=f"Radius, #vertices",
            filename=f"{plot_folder_name}/{moments_vsr_filename}",
        )

        # Plotting Sobol indices as a treemap
        #TODO: figure out how to plot treemaps at arbitrary locations
        # results.plot_sobols_treemap(
        #     qoi=qoi,
        #     filename=f"{plot_folder_name}/{sobols_treemap_filename}",
        # )

        # Read out the arrays of stats from the results object
        y = results.describe(qoi, 'mean')
        ymed = results.describe(qoi, 'median')
        sy = results.describe(qoi, 'std')
        y01 = results.describe(qoi, '1%')
        y10 = results.describe(qoi, '10%')
        y90 = results.describe(qoi, '90%')
        y99 = results.describe(qoi, '99%')
        ymin = results.describe(qoi, 'min')
        ymax = results.describe(qoi, 'max')

        # Filling in the values for the list of dicts for a common boxplot
        stats_dict_s.append({
            'mean': [y[r_ind_qoi]],
            'med': [ymed[r_ind_qoi]],
            'q1': [y10[r_ind_qoi]],
            'q3': [y90[r_ind_qoi]],
            'cilo': [y[r_ind_qoi] - 1.95* sy[r_ind_qoi]],
            'cihi': [y[r_ind_qoi] + 1.95* sy[r_ind_qoi]],
            'whislo': [y01[r_ind_qoi]],
            'whishi': [y99[r_ind_qoi]],
            'fliers': [ymin[r_ind_qoi], ymax[r_ind_qoi]],
            'label': f"{qoi}",
        })

        # Define a simple range for x-axis
        #rs = np.linspace(0., 1., len(y))  # Should be done outside of scope of current function

        # Bespoke plotting of uncertainty in QoI (vs. radius)
        plot_unc_vs_r(rs, y, sy, y10, y90, qoi_name=qoi, foldername=plot_folder_name, filename=moments_vsr_filename)

        # Plotting Sobol indices as a function of radius
        results.plot_sobols_first(
            qoi=qoi,
            withdots=False,  # Show dots for each Sobol index
            xlabel=f"Radius, #vertices",
            ylabel=f"Sobol Index (first) at {qoi}",
            filename=f"{plot_folder_name}/{sobols_filename}",  # Save with bespoke prefix
        )

        print(f"Plots (for spatially resolved functions) saved: {moments_vsr_filename}, {sobols_treemap_filename}, {sobols_filename}")
        #TODO compare those in absolute values - fix the y axis limits?

    # Save plot common for QoIs: specific for bespoke QoI uncertainty plotting
    #  - bespoke plotting of uncertainty in QoI (at selected radius)
    plot_unc_qoi(stats_dict_s, qoi_name=qoi, foldername=plot_folder_name, filename=add_timestamp_to_filename("qoi_uncertainty_vs_r.png", plot_timestamp),r_ind=r_ind_qoi)

    return 0

def plot_unc_vs_t(r_at_r, t_s, y_at_r, sy_at_r, y10_at_r, y90_at_r, foldername="", filename=""):
    """
    Plot uncertainty in the results as a function of time.
    """
    fig, ax = plt.subplots()
    # print(f"Shapes of the lists: y_s: {len(y_s)}, sy_s: {len(sy_s)}, y10_s: {len(y10_s)}, y90_s: {len(y90_s)}") ###DEBUG

    ax.plot(t_s, y_at_r, label=f'<y> at r={r_at_r}')
    ax.fill_between(t_s, np.array(y_at_r) - np.array(sy_at_r), np.array(y_at_r) + np.array(sy_at_r), alpha=0.3, label='+/- STD')
    ax.fill_between(t_s, y10_at_r, y90_at_r, alpha=0.1, label='10% - 90%')

    ax.set_title(f"Uncertainty as a function of time at r={r_at_r}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Concentration [m^-3] at {r_at_r}")
    ax.legend(loc='best')
    ax.grid(True)

    fig.savefig(f"{foldername}/{filename}")

    plt.close()

    return 0

def plot_sobols_vs_t(r_s, t_s, s1_s, distributions, foldername="", filename="", r_ind=0):
    """
    Plot Sobol indices as a function of time.
    """
    fig, ax = plt.subplots()
    # print(s1_s[-1]) ### DEBUG

    for i, param_name in enumerate(distributions.keys()):
        # Extract r_ind-th element from each Sobol array to get time series at fixed radius
        s1_at_r = [s1_timestep[r_ind] for s1_timestep in s1_s[i]]
        ax.plot(t_s, s1_at_r, label=f'Sobol Index (first) for {param_name}')

    ax.set_title(f"Sobol Indices as a function of time at r={r_s}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Sobol Index (first), fraction of unity")
    ax.legend()

    fig.savefig(f"{foldername}/{filename}")

    plt.close()

    return 0

def plot_stats_vs_t(results, distributions, qois, plot_folder_name, plot_timestamp, rs=None):
    """
    Plot statistics of the results as a function of time.
    """

    # Select - uncertainty at the depth of the specimen (r=0.)
    r_ind_selected = [0, -1] # Select the first and last radius index (or any other index)

    # Read the results for all times and align data for plotting against time
    y_s = []
    sy_s = []
    y10_s = []
    y90_s = []
    r_s = []

    s1_s = [[] for _ in range(len(distributions))]  # Assuming one first Sobol index per distribution

    # op1) Extract time from QoI names
    #t_s = [float(qoi.split('=')[1].strip()) for qoi in qois]
    # op2) read from results
    t_s = []

    # Run over QoIs in analysis results object and read statistics
    for qoi in qois:
        # Every element in qois list is a single time step
        # Read every time step from results in a list-of-lists [n_timesteps x n_elements]
        t_s.append(float(qoi.split('=')[1].strip()[:-1]))  # Extract time from QoI names, strp 's' at the end and '='
        y_s.append(results.describe(qoi, 'mean'))
        sy_s.append(results.describe(qoi, 'std'))
        y10_s.append(results.describe(qoi, '10%'))
        y90_s.append(results.describe(qoi, '90%'))
        s1 = results.sobols_first(qoi) # returing a dict {input_param: (list of) Sobol index values}
        for i,param_name in enumerate(distributions.keys()):
            # Assuming each distribution is a valid QoI descriptor
            s1_s[i].append(s1[param_name])  # Assuming 'first' is a valid QoI descriptor
        #r_s.append(np.linspace(0., 1., len(y_s[-1])))  
        r_s.append(rs) # assuming we read the readius values from outside, and they are the same for all QoIs

    # Run over selected radius indices
    for r_ind in r_ind_selected:
        # Generate filenames with timestamp for time series plots
        moments_vst_filename = add_timestamp_to_filename(f"moments_vs_t_at_{r_ind}.png", plot_timestamp)
        sobols_vst_filename = add_timestamp_to_filename(f"sobols_first_vs_t_at_{r_ind}.png", plot_timestamp)

        # Extract r_ind-th element from each time step (array) to get time series at fixed radius
        y_at_r = [y_timestep[r_ind] for y_timestep in y_s]
        sy_at_r = [sy_timestep[r_ind] for sy_timestep in sy_s]
        y10_at_r = [y10_timestep[r_ind] for y10_timestep in y10_s]
        y90_at_r = [y90_timestep[r_ind] for y90_timestep in y90_s]
        r_at_r = [r_timestep[r_ind] for r_timestep in r_s]  # Assuming r_s is a list of lists with radius values
        #TODO check if r_at_r changes with time, or is constant

        # Plotting of moments as a function of time
        plot_unc_vs_t(r_at_r[0], t_s, y_at_r, sy_at_r, y10_at_r, y90_at_r, foldername=plot_folder_name, filename=moments_vst_filename)

        # Plotting Sobol indices as a function of time
        plot_sobols_vs_t(r_at_r[0], t_s, s1_s, distributions, foldername=plot_folder_name, filename=sobols_vst_filename, r_ind=r_ind)

        print(f"Plots (for time series) saved: {moments_vst_filename}, {sobols_vst_filename}")

    return 0

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
    if not os.path.exists("plots_festim_uq_" + plot_timestamp):
        plot_folder_name = "plots_festim_uq_" + plot_timestamp
        os.makedirs(plot_folder_name)
        # Save plots in this folder
    #TODO: assumes that by default there is no folder with plots, and creates a new one

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
        "source_value": 1.0e18, #1.0e19, #,1.0e20,  # Mean source value
        "left_bc_value": 1.0e15,  # Mean boundary condition value: better to keep right side as the domain boundary and left as centre
        "right_bc_value": 1.0e17, #1.0e16, #1.0e15,  # Mean boundary condition value
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

        "source_value": cp.Uniform(means["source_value"]*(1.-expansion_factor_uniform*CoV), means["source_value"]*(1.+expansion_factor_uniform*CoV)), 

        #"left_bc_value": cp.Uniform(means["left_bc_value"]*(1.-CoV), means["left_bc_value"]*(1.+CoV)),  # Boundary condition value: see on choice of left/right BC

        "right_bc_value": cp.Uniform(means["right_bc_value"]*(1.-expansion_factor_uniform*CoV), means["right_bc_value"]*(1.+expansion_factor_uniform*CoV)),  # Boundary condition value at the right (outer) surface of the sample
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

    "source_value": {"type": "float", "default": 1.0e20,},

    #"left_bc_value": {"type": "float", "default": 1.0e15},  # Boundary condition value: better to specify centre at r=0.0

    "right_bc_value": {"type": "float", "default": 1.0e15},  # Boundary condition value at the right (outer) surface of the sample
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

def prepare_uq_campaign(*args, **kwargs):
    """
    Prepare the uncertainty quantification (UQ) campaign by creating necessary steps: set-up, parameter definitions, encoders, decoders, and actions.
    """

    # Define the model input and output parameters
    parameters, qois = define_festim_model_parameters()

    # TODO rearange FESTIM data output, figure out how to specify multiple quantities at different times, all vs a coordinate

    # Set up necessary elements for the EasyVVUQ campaign

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
            "source_value": "source_terms.source_value",
            "left_bc_value": "boundary_conditions.left_bc_value",
            "right_bc_value": "boundary_conditions.right_bc_value"
        },
        type_conversions={
            "D_0": float,
            "E_D": float,
            "T": float,
            "source_value": float,
            "left_bc_value": float,
            "right_bc_value": float, 
        }
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
        target_filename="results/results.txt",  # Results from the base data of a simulation
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

def perform_uq_festim(*args, **kwargs):
    """
    Main function to perform the UQ campaign for FESTIM.
    This function orchestrates the preparation, execution, and analysis of the UQ campaign.
    """
    # EasyVVUQ script to be executed as a function
    # TODO single out parts into function / methods of a class - test
    print("Starting FESTIM UQ campaign...")

    # Prepare the UQ campaign
    # This includes defining parameters, encoders, decoders, and actions
    campaign, qois, distributions, campaign_timestamp, sampler = prepare_uq_campaign()

    # TODO: add more parameter for Arrhenious law
    # TODO: try higher BC concentration values - does it even make sense to have such low BC
    # TODO: run with higher polynomial degree (+: now 2)
    # TODO: check if there are actually negative concentrations, if yes - check model and specify correct params ranges

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