import os
import sys
import subprocess

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


parameters = {
    "D_0": {"type": "float", "default": 1.0e-7,},
    "E_D": {"type": "float", "default": 0.2,},
    "T": {"type": "float", "default": 300.0,},
    "source_value": {"type": "float", "default": 1.0e20,},
    "left_bc_value": {"type": "float", "default": 1e15},  # Boundary condition value
}

qois = [
    "tritium_inventory",
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
decoder = uq.decoders.SimpleCSV(target_filename="output.csv", output_columns=qois)

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
campaign = uq.Campaign(
    name="festim_campaign_",
    params=parameters,
    #qois=qois,
    actions=actions,
)

# Define uncertain parameters distributions
distributions = {
    #"D_0": cp.Uniform(1.0e-7, 1.0e-5), # Diffusion coefficient base value
    #"E_D": cp.Uniform(0.1, 1.0), # Activation energy
    "T": cp.Uniform(200.0, 400.0), # Temperature
    "source_value": cp.Uniform(1.0e19, 1.0e21),  # Assuming source_value is also a parameter
    "left_bc_value": cp.Uniform(1e14, 1e16),  # Boundary condition value
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
results_filename = "results.hdf5"
campaign.campaign_db.dump()

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
analysis_filename = "analysis_results.hdf5"

# Plot the results: error bar for each QoI
import matplotlib.pyplot as plt

for qoi in qois:
    results.plot_moments(
        qoi=qoi,
        filename=f"{qoi}_moments.png",
    )
    results.plot_sobols_treemap(
        qoi=qoi,
        filename=f"{qoi}_sobols_treemap.png",
    )

# TODO:
# 0) Double chekc everything and clean up the code
# 1) Plot uncertainty in single scalar QoI
# 2) Plot uncertainty in profile QoI as a function of radius
# 3) Plot uncertainty in scalar QoI as a function of time
# 4) Plot Sobol indices as a function of radius
# 5) Plot Sobol indices as a function of time