#!/home/yehor/miniconda3/envs/festim-env/bin/python3
import sys
import os

# Set environment variables to suppress Qt warnings
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-' + str(os.getuid())

import numpy as np
import yaml
import argparse
from pathlib import Path

from festim_model_run import load_config  # Import the load_config function from festim_model_run

# Add parent directory to Python path to import festim_model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# Now we can import festim_model from parent directory
from festim_model import Model
from festim_model.diagnostics import Diagnostics

# Import UQ functions from easyvvuq_festim
from easyvvuq_festim import perform_uq_festim, run_uq_campaign, analyse_uq_results, visualisation_of_results

def parameter_scan(config, param_name='length', level_variation=3):
    """
    Perform parameter scan for the FESTIM model.
    This function is a placeholder for actual parameter scanning logic.
    """
    # Load the base configuration
    if not config:
        print("No configuration provided for parameter scan.")
           
        print("Trying to load YAML configuration file...")
        parser = argparse.ArgumentParser(description='Run FESTIM model with YAML configuration')
    
        parser.add_argument('--config', '-c', 
                       default='config.yaml',
                       help='Path to YAML configuration file (default: config.yaml)')
    
        args = parser.parse_args()

        config = load_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    
    print(f"Performing parameter scan for {param_name}...")

    # Get the default value of the parameter from the configuration
    param_def_val = config.get(param_name, 1.0)

    if param_def_val is None:
        print(f"Warning: Default value for {param_name} not found in configuration, using 1.0")
        param_def_val = 1.0 
    
    # specifically, for the length parameter
    if param_name == 'length':
        param_def_val = float(config['geometry'].get('length', 1.0))
        print(f"Using default length value: {param_def_val} m")

    # Specify the list of parameter values to scan

    # Option 1) use logarithmic scale, specify range of magnitudes
    log_base = 10.0  # Base for logarithmic scale
    log_scale_range = level_variation # try log_scale_range number of orders of magnitude towards higher and lower values #TODO think of a better way to specify the range
    param_value_lo_bound = param_def_val * 10**(-log_scale_range)
    param_value_hi_bound = param_def_val * 10**(log_scale_range)

    n_runs = 2*log_scale_range + 1  # Number of runs for the scan

    param_values = param_def_val * np.logspace(
        np.log(param_value_lo_bound) / np.log(log_base),
        np.log(param_value_hi_bound) / np.log(log_base), 
        num=n_runs, 
        base=log_base
    )

    # Option 2) use logarithmic scale, specify range of values

    # Option 3) use linear scale, specify range of values

    results = []

    for value in param_values:
        print(f"Running simulation with {param_name} = {value}")

        # Update the configuration with the specific results folder name
        config['simulation']['output_directory'] = f"results_{param_name}_{value:.2e}"

        # Update the configuration with the current parameter value
        config[param_name] = value #TODO make fall back if config does not have this parameter or the structure is different
        
        # Specifically, for the length parameter
        if param_name == 'length':
            config['geometry']['length'] = value
        
        # Create a model instance with the updated configuration
        model = Model(config=config)
        
        # Run the model and collect results
        result = model.run()
        results.append(result)

        # Visualise results
        diagnostics = Diagnostics(model, results=result, result_folder=model.result_folder)
        diagnostics.visualise()

        print(f"Simulation with {param_name} = {value} completed.")

    print("Parameter scan completed!")

    # Plot the results as a dependency on the parameter value
    print("Plotting results...")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(param_values, results, marker='o')
    plt.xscale('log')  # Use logarithmic scale for x-axis
    plt.xlabel(param_name)
    plt.ylabel('Result')
    plt.title(f'Parameter Scan: {param_name}')
    plt.grid(True)
    plt.savefig(f'parameter_scan_{param_name}.png')
    plt.close()
    #TODO for the length scan specifically, map the results on the [0,1] normalised coordinate system, e.g. r = r/length

    return results

def param_scan_sensitivity_analysis(config, param_name='length', level_variation=3):
    """
    Perform sensitivity analysis for the FESTIM model with a given parameter being scanned.
    This function is a placeholder for actual sensitivity analysis logic.
    """
    print("Performing sensitivity analysis SCAN ...")

    # Load the configuration
    if not config:
        # TODO read *_yaml.template file to get the default configuration
        print("No configuration provided for sensitivity analysis.")
        return

    # Get the default value of the parameter from the configuration
    param_def_val = config.get(param_name, 1.0)

    if param_def_val is None:
        print(f"Warning: Default value for {param_name} not found in configuration, using 1.0")
        param_def_val = 1.0 
    
    # specifically, for the length parameter
    if param_name == 'length':
        param_def_val = float(config['geometry'].get('length', 1.0))
        print(f"Using default length value: {param_def_val} m")
        param_name = 'length'  # Use the full path to the parameter in the config

    # Specify the list of parameter values to scan

    # 1) use logarithmic scale, specify range of magnitudes
    log_base = 10.0  # Base for logarithmic scale
    log_scale_range = level_variation # try log_scale_range number of orders of magnitude towards higher and lower values #TODO think of a better way to specify the range
    param_value_lo_bound = param_def_val * 10**(-log_scale_range)
    param_value_hi_bound = param_def_val * 10**(log_scale_range)

    n_runs = 2*log_scale_range + 1  # Number of runs for the scan

    param_values = param_def_val * np.logspace(
        # np.log(param_value_lo_bound) / np.log(log_base),
        # np.log(param_value_hi_bound) / np.log(log_base), 
        -log_scale_range,
        +log_scale_range,
        num=n_runs, 
        base=log_base,
    )
    
    if param_values is None or len(param_values) == 0:
        print("No sensitivity parameters found in configuration, no parameter range was generated, aborting scan")
        return

    # Perform sensitivity analysis logic here
    for param_value in param_values:
        # For now, just print the parameters
        print(f"Parameter: {param_name}, Value: {param_value}")

        #TODO pass new parameter values fixes for the whole campaign
        perform_uq_festim({param_name: param_value})
        #TODO save as display the modified parameter in the scan

    # TODO make a plot of Sensitivity indices (at r=0) as a function of the parameter value (here: sample length); maybe a surface 3D plot


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run FESTIM model SCAN with a default YAML configuration')

    parser.add_argument('--config', '-c',
                       default='config.yaml',
                       help='Path to YAML configuration file (default: config.yaml)'
                       )

    args = parser.parse_args()

    # Load the configuration from the specified file
    config = load_config(args.config)

    if config:

        # # Perform parameter scan
        # parameter_scan(config, param_name='length', level_variation=3)

        # Perform sensitivity analysis scan
        param_scan_sensitivity_analysis(config, param_name='length', level_variation=3)
    else:
        print("Failed to load configuration. Exiting.")