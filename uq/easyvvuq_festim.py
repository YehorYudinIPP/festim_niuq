import os
import sys
import subprocess
from datetime import datetime
import numpy as np

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
    conda_python = "/home/yehor/miniconda3/envs/festim-env/bin/python"
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

def plot_unc_vs_r(r, y, sy, y10, y90, qoi_name, foldername, filename):
    """
    Plot uncertainty in the results as a function of radius (spatial coordinates).
    """
    fig, ax = plt.subplots()

    ax.plot(r, y, label=f'<y> at {qoi_name}')
    ax.fill_between(r, y - sy, y + sy, alpha=0.3, label='+/- STD')
    ax.fill_between(r, y10, y90, alpha=0.1, label='10% - 90%')

    ax.set_title(f"Uncertainty in {qoi_name} as a function of radius")
    ax.set_xlabel("#Radius, vertices")
    ax.set_ylabel(f"Concentration at {qoi_name}")
    ax.legend()

    fig.savefig(f"{foldername}/bespoke_{filename}")

    plt.close()  # Close the plot to avoid display issues in some environments

    return 0

def plot_unc_qoi(r, y, sy, y10, y90, qoi_name, foldername, filename):
    """
    Plot uncertainty in the specific scalar QoIs.
    """
    fig, ax = plt.subplots()

    #Boxplotting the mean and std at a single radius
    r_ind = -1  # Select the first radius (or any other index)

    ax.plot(r[r_ind], y[r_ind], 'o', label=f'<y> at r={r[r_ind]:.2f} and {qoi_name}')
    ax.errorbar(r[r_ind], y[r_ind], yerr=sy[r_ind], fmt='o', label=f"+/- STD at r={r[r_ind]:.2f} and {qoi_name}")
    ax.fill_betweenx([y10[r_ind], y90[r_ind]], r[r_ind] - 0.01, r[r_ind] + 0.01, alpha=0.1, label=f"10% - 90% at r={r[r_ind]:.2f} and {qoi_name}")

    ax.set_title(f"Uncertainty in {qoi_name} at radius {r[r_ind]:.2f}")
    ax.set_ylabel(f"Concentration in {qoi_name} at radius {r[r_ind]:.2f}")
    ax.legend()

    fig.savefig(f"{foldername}/bespoke_qoi_{filename}")

    plt.close()

    return 0

def plot_stats_vs_r(results, distributions, qois, plot_folder_name, plot_timestamp):
    """
    Plot statistics of the results as a function of radius (spatial coordinates).
    """
    
    # Rund over QoIs in analysis results object
    for qoi in qois:
        # Generate filenames with timestamp
        moments_vsr_filename = add_timestamp_to_filename(f"{qoi}_moments_vs_r.png", plot_timestamp)
        sobols_treemap_filename = add_timestamp_to_filename(f"{qoi}_sobols_treemap.png", plot_timestamp)
        sobols_filename = add_timestamp_to_filename(f"{qoi}_sobols_first_vs_r.png", plot_timestamp)
        
        # Default plotting of the moments
        results.plot_moments(
            qoi=qoi,
            filename=f"{plot_folder_name}/{moments_vsr_filename}",
        )

        # Plotting Sobol indices as a treemap
        results.plot_sobols_treemap(
            qoi=qoi,
            filename=f"{plot_folder_name}/{sobols_treemap_filename}",
        )
        plt.close()  # Close the plot to avoid display issues in some environments

        # Read out the arrays of stats from the results object
        y = results.describe(qoi, 'mean')
        sy = results.describe(qoi, 'std')
        y10 = results.describe(qoi, '10%')
        y90 = results.describe(qoi, '90%')

        # Define a simple range for x-axis
        r = np.linspace(0., 1., len(y))  # Assuming a simple range for x-axis
        #TODO read 'x' column from results to get actual radius values

        # Bespoke plotting of uncertainty in QoI (vs. radius)
        plot_unc_vs_r(r, y, sy, y10, y90, qoi_name=qoi, foldername=plot_folder_name, filename=moments_vsr_filename)

        # Bespoke plotting of uncertainty in QoI (at selected radius)
        plot_unc_qoi(r, y, sy, y10, y90, qoi_name=qoi, foldername=plot_folder_name, filename=moments_vsr_filename)

        # Plotting Sobol indices as a function of radius
        results.plot_sobols_first(
        qoi=qoi,
        withdots=False,  # Show dots for each Sobol index
        xlabel='Radius, #vertices',
        ylabel='Sobol Index (first)',
        filename=f"{plot_folder_name}/{sobols_filename}",  # Save with bespoke prefix
    )

    print(f"Plots (for spatially resolved functions) saved: {moments_vsr_filename}, {sobols_treemap_filename}, {sobols_filename}")

    return 0

def plot_unc_vs_t(t_s, y_s, sy_s, y10_s, y90_s, foldername="", filename="", r_ind=0):
    """
    Plot uncertainty in the results as a function of time.
    """
    fig, ax = plt.subplots()
    # print(f"Shapes of the lists: y_s: {len(y_s)}, sy_s: {len(sy_s)}, y10_s: {len(y10_s)}, y90_s: {len(y90_s)}") ###DEBUG
    
    # Extract r_ind-th element from each time step (array) to get time series at fixed radius
    y_at_r = [y_timestep[r_ind] for y_timestep in y_s]
    sy_at_r = [sy_timestep[r_ind] for sy_timestep in sy_s]
    y10_at_r = [y10_timestep[r_ind] for y10_timestep in y10_s]
    y90_at_r = [y90_timestep[r_ind] for y90_timestep in y90_s]
    
    ax.plot(t_s, y_at_r, label=f'<y> at r_index={r_ind}')
    ax.fill_between(t_s, np.array(y_at_r) - np.array(sy_at_r), np.array(y_at_r) + np.array(sy_at_r), alpha=0.3, label='+/- STD')
    ax.fill_between(t_s, y10_at_r, y90_at_r, alpha=0.1, label='10% - 90%')

    ax.set_title(f"Uncertainty as a function of time at radius index {r_ind}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Concentration")
    ax.legend()

    fig.savefig(f"{foldername}/{filename}")

    plt.close()

    return 0

def plot_sobols_vs_t(t_s, s1_s, foldername="", filename="", r_ind=0):
    """
    Plot Sobol indices as a function of time.
    """
    fig, ax = plt.subplots()
    # print(s1_s[-1]) ### DEBUG
    for i, param_name in enumerate(distributions.keys()):
        # Extract r_ind-th element from each Sobol array to get time series at fixed radius
        s1_at_r = [s1_timestep[r_ind] for s1_timestep in s1_s[i]]
        ax.plot(t_s, s1_at_r, label=f'Sobol Index (first) for {param_name}')
    ax.set_title(f"Sobol Indices as a function of time at radius index {r_ind}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Sobol Index (first)")
    ax.legend()
    fig.savefig(f"{foldername}/{filename}")
    plt.close()

    return 0

def plot_stats_vs_t(results, distributions, qois, plot_folder_name, plot_timestamp):
    """
    Plot statistics of the results as a function of time.
    """

    # Select - uncertainty at the depth of the specimen (r=0.)
    r_ind_selected = [-1] # Select the first radius (or any other index)

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

    # Run over QoIs in analysis results object
    for qoi in qois:
        # Every element in qois list is a single time step
        # Read every time step from results in a list-of-lists [n_timesteps x n_elements]
        t_s.append(float(qoi.split('=')[1].strip()[:-1]))  # Extract time from QoI names
        y_s.append(results.describe(qoi, 'mean'))
        sy_s.append(results.describe(qoi, 'std'))
        y10_s.append(results.describe(qoi, '10%'))
        y90_s.append(results.describe(qoi, '90%'))
        s1 = results.sobols_first(qoi) # returing a dict {input_param: (list of) Sobol index values}
        for i,param_name in enumerate(distributions.keys()):
            # Assuming each distribution is a valid QoI descriptor
            s1_s[i].append(s1[param_name])  # Assuming 'first' is a valid QoI descriptor
        r_s.append(np.linspace(0., 1., len(y_s[-1])))  # Assuming a simple range for x-axis

    for r_ind in r_ind_selected:
        # Generate filenames with timestamp for time series plots
        moments_vst_filename = add_timestamp_to_filename(f"{qoi}_at{r_ind}_moments_vs_t.png", plot_timestamp)
        sobols_vst_filename = add_timestamp_to_filename(f"{qoi}_at{r_ind}_sobols_first_vs_t.png", plot_timestamp)

        # Plotting of moments as a function of time
        # TODO: plot those in a single figure
        plot_unc_vs_t(t_s, y_s, sy_s, y10_s, y90_s, foldername=plot_folder_name, filename=moments_vst_filename, r_ind=r_ind)

        # Plotting Sobol indices as a function of time
        plot_sobols_vs_t(t_s, s1_s, foldername=plot_folder_name, filename=sobols_vst_filename, r_ind=r_ind)

    print(f"Plots (for time series) saved: {moments_vst_filename}, {sobols_vst_filename}")

    return 0

parameters = {
    "D_0": {"type": "float", "default": 1.0e-7,},
    "E_D": {"type": "float", "default": 0.2,},
    "T": {"type": "float", "default": 300.0,},
    "source_value": {"type": "float", "default": 1.0e20,},
    "left_bc_value": {"type": "float", "default": 1e15},  # Boundary condition value
}

# TODO rearange FESTIM data output, figure out how to specify multiple quantities at different times, all vs a coordinate
# TODO: read an (example) output file to get the QoI names
qois = [
    #"tritium_inventory",
    "t=5.00e-01s",
    "t=1.00e+00s",
    "t=2.00e+00s",
    "t=5.00e+00s",
]

# Set up necessary elements for the EasyVVUQ campaign

# Option 1: Simple YAML Encoder
# encoder = YAMLEncoder(
#     template_fname="festim_yaml.template",
#     target_filename="config.yaml",
#     delimiter="$"
# )

# Option 2: Advanced YAML Encoder (alternative)
encoder = AdvancedYAMLEncoder(
    template_fname="festim_yaml.template",
    target_filename="config.yaml",
    parameter_map={
        "D_0": "materials.D_0",
        "E_D": "materials.E_D",
        "T": "materials.T",
        "source_value": "source_terms.source_value",
        "left_bc_value": "boundary_conditions.left_bc_value"
    },
    type_conversions={
        "D_0": float,
        "E_D": float,
        "T": float,
        "source_value": float,
        "left_bc_value": float
    }
)
print(f"Using encoder: {encoder.__class__.__name__}") ###DEBUG

# Option 3: Use built-in EasyVVUQ encoder
# encoder = uq.encoders.JinjaEncoder(
#     template_fname="festim.template", 
#     target_filename="config.yaml"
# )

# Decoder
# TODO change the output and decoder to YAML
decoder = uq.decoders.SimpleCSV(
    #target_filename="output.csv", # option for synthetic diagnostics specifically chosen for UQ
    target_filename="results/results.txt",  # Results from the base data of a simulation
    output_columns=qois
      )

# Execution script - validate and setup environment
python_exe, script_path = validate_execution_setup()

# Use the filename that the encoder creates (config.yaml)
exec_command_line = f"{python_exe} {script_path} --config config.yaml"
print(f"Execution command line: {exec_command_line}")

# Execute the script locally
execute = ExecuteLocal(exec_command_line)

# Set up actions for the campaign
actions = Actions(
    CreateRunDirectory('run_dir'),
    # Copy config file to each run directory (alternative approach)
    # CopyFile(config_file, 'config.yaml'),  # Uncomment if you want to copy config to each run
    Encode(encoder),
    execute,
    Decode(decoder),
)

# Define the campaign
campaign_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
campaign = uq.Campaign(
    name=f"festim_campaign_{campaign_timestamp}_",
    params=parameters,
    #qois=qois,
    actions=actions,
)

# Define uncertain parameters distributions
CoV = 0.25
means = {
    "T": 300.0,  # Mean temperature
    "source_value": 1.0e20,  # Mean source value
    "left_bc_value": 1.0e15,  # Mean boundary condition value
}
distributions = {
    #"D_0": cp.Uniform(1.0e-7, 1.0e-5), # Diffusion coefficient base value
    #"E_D": cp.Uniform(0.1, 1.0), # Activation energy
    "T": cp.Uniform(means["T"]*(1.-CoV), means["T"]*(1.+CoV)), # Temperature [K]
    "source_value": cp.Uniform(means["source_value"]*(1.-CoV), means["source_value"]*(1.+CoV)),  # Assuming source_value is also a parameter
    "left_bc_value": cp.Uniform(means["left_bc_value"]*(1.-CoV), means["left_bc_value"]*(1.+CoV)),  # Boundary condition value
}

# Define sampling method
p_order = 1  # Polynomial order for PC expansion

sampler = uq.sampling.PCESampler(
        vary=distributions,
        polynomial_order=p_order,
)

campaign.set_sampler(sampler)

# Run the campaign
with QCGPJPool() as qcjpool:
    campaign_results = campaign.execute(pool=qcjpool)
    campaign_results.collate()

# Get results from the campaign
# results = campaign_results.get_collation_results()

# Save the results
results_filename = add_timestamp_to_filename("results.hdf5")
campaign.campaign_db.dump()

print(f"Results saved to: {results_filename}")

# Perform the analysis
analysis = uq.analysis.PCEAnalysis(sampler=sampler,
                                   qoi_cols=qois,
)
campaign.apply_analysis(analysis)

results = campaign.get_last_analysis()

# Display the results of the analysis
for qoi in qois:
    print(f"Results for {qoi}:")
    print(results.describe(qoi))
    print("\n")

# Save the analysis results
analysis_filename = add_timestamp_to_filename("analysis_results.hdf5")

print(f"Analysis results saved to: {analysis_filename}")

### PLOTTING RESULTS ###

# Plot the results: error bar for each QoI + other plots
import matplotlib.pyplot as plt

# Create a common timestamp for all plots from this run
plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create a folder to save plots
if not os.path.exists("plots_festim_uq_" + plot_timestamp):
    plot_folder_name = "plots_festim_uq_" + plot_timestamp
    os.makedirs(plot_folder_name)
    # Save plots in this folder

# Plotting statistics of the results as a function of radius (spatial coordinates)
plot_stats_vs_r(results, distributions, qois, plot_folder_name, plot_timestamp)

# Bespoke plotting of uncertainty and Sobol indices in QoI as a function of TIME
plot_stats_vs_t(results, distributions, qois, plot_folder_name, plot_timestamp)

print(f"Plots saved in folder: {plot_folder_name}")

print("FESTIM UQ campaign completed successfully!")


##################################
# TODO:
# add legends to bespoke codes
# make sobol plots more clear; figure out disparity!

# 0) Double check everything and clean up the code

# 1) Plot uncertainty in single scalar QoI (+/-)
# 2) Plot uncertainty in profile QoI as a function of radius (+)
# 3) Plot uncertainty in scalar QoI as a function of time (+)
# 4) Plot Sobol indices as a function of radius (+)
# 5) Plot Sobol indices as a function of time
# 6*) Add convergence (to the steady state) as a QoI