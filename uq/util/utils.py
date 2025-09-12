import os
import sys
import subprocess
import yaml
from pathlib import Path
import numpy as np

from datetime import datetime

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# from serializer import serialize_yaml
# from serializer import deserialize_yaml
from joblib import dump, load

def load_config(config_file):
    """Load configuration from YAML file specific for UQ"""

    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        print(f" >> Configuration loaded: {config}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None
    
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
    env_python = "/home/yhy25yyp/anaconda3/envs/festim2-env/bin/python3"
    #env_python = "/home/yhy25yyp/workspace/festim2-venv/bin/python3"
    
    if os.path.exists(env_python):
        return env_python
    
    # Method 2: Try to find conda environment dynamically
    try:
        result = subprocess.run(['conda', 'info', '--envs'], 
                              capture_output=True, text=True, check=True)
        for line in result.stdout.split('\n'):
            if 'festim2-env' in line:
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

def save_sa_results_yaml():
    """
    Save sensitivity analysis results to a YAML file. - example
    """
    results = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'description': 'Sensitivity analysis results',
        'data': {
            # Example data structure
            'sensitivity_indices': [0.1, 0.2, 0.3],
            'parameters': ['param1', 'param2', 'param3']
        }
    }
    
    filename = add_timestamp_to_filename("sa_results.yaml")
    # serialize_yaml(results, filename) # TODO - implement, or copy

    print(f"✓ Sensitivity analysis results saved to: {filename}")
    return filename

def integrate_statistics(uq_resuls):
    """
    A function to integrate statistics over the quantities it is conditioned on:
    s_int = int_X s(x)dx
    12/09/2025: basic functionality is to integrate Sobol indices over the domain of radius values: r e [0., R_max]
    """

    # Integrating Sobol indices over the domain of radius values: r e [0., R_max]

    # Get first sobol indices for all quantities of interest
    qois = uq_resuls.get_qoi_names()
    print(f"Quantities of interest: {qois}")

    for qoi in qois[1:]:  # Skip the first 'run' entry
        print(f"\n>>> Integrating statistics for quantity of interest: {qoi}")

        # Get the first-order Sobol indices for this quantity of interest
        sobol_first = uq_resuls.get_sobol_first(qoi)
        print(f"Sobol first-order indices shape: {sobol_first.shape}")  # (n_samples, n_params)

        # Assuming the first dimension corresponds to the varying parameter (e.g., radius)
        # and the second dimension corresponds to different parameters

        # Integrate over the first dimension (e.g., radius)
        stat_integrated = {}
        for param_idx in range(sobol_first.shape[1]):
            param_name = f"param_{param_idx+1}"
            param_values = uq_resuls.get_param_values(param_name)
            if param_values is None:
                print(f"Parameter values for {param_name} not found, skipping integration.")
                continue

            # Simple trapezoidal integration over the parameter values
            integrated_value = np.trapz(sobol_first[:, param_idx], x=param_values)
            stat_integrated[param_name] = integrated_value
            print(f"Integrated Sobol index for {param_name}: {integrated_value}")

    return stat_integrated