#!/home/yehor/miniconda3/envs/festim-env/bin/python3
import sys
import os

from util.utils import compute_absolute_tolerance

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

def make_parameter_values_list(param_def_val, level_variation=3, scale='log'):
    """
    Generate a list of parameter values to scan based on a default value and level of variation.
    
    Args:
        param_def_val (float): Default value of the parameter.
        level_variation (int): Level of variation (number of orders of magnitude).
        Returns:
        param_values (list): List of parameter values to scan.
    """
    if scale not in ['log', 'linear']:
        raise ValueError("Scale must be either 'log' or 'linear'")
    if level_variation < 1:
        raise ValueError("Level of variation must be at least 1")   
    
    if scale == 'log':
        # 1) use logarithmic scale, specify range of magnitudes
        log_base = 10.0  # Base for logarithmic scale
        log_scale_range = level_variation # try log_scale_range number of orders of magnitude towards higher and lower values #TODO think of a better way to specify the range

        param_value_lo_bound = param_def_val * 10**(-log_scale_range)
        param_value_hi_bound = param_def_val * 10**(log_scale_range)

        n_runs = 2*log_scale_range + 1  # Number of runs for the scan

        param_values = param_def_val * np.logspace(
            -log_scale_range,
            +log_scale_range,
            num=n_runs, 
            base=log_base,
        )

    elif scale == 'linear':
        # 2) use linear scale, specify range of values
        vary_factor = 0.1
        linear_scale_range = level_variation * vary_factor * param_def_val # try level_variation * 10% of the default value towards higher and lower values

        param_value_lo_bound = param_def_val - linear_scale_range
        param_value_hi_bound = param_def_val + linear_scale_range 

        n_runs = 2*level_variation + 1  # Number of runs for the scan

        param_values = np.linspace(param_value_lo_bound, param_value_hi_bound, num=n_runs)

    else:
        # Should not reach here due to earlier check
        raise ValueError("Scale must be either 'log' or 'linear'")

    return param_values

def parameter_scan(config, param_name='length', level_variation=3, target_dir="./", scale='log'):
    """
    Perform parameter scan for the FESTIM model.
    This function varies a given parameter over a specified range and runs the model for each value.
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
    
    print(f"\n ..!Performing parameter scan for {param_name}!..\n")

    # Get the default value of the parameter from the configuration
    # TODO write and store pathes to parameters!
    # For the length parameter

    param_explanation = {
        'length': 'Size of the physical sample',
        'G': 'Tritium generation rate, volumetric',
        'T_in': 'Temperature inside the sample',
    }

    param_units = {
        'length': f"$ m $",
        'G': f"$ m^{{-3}} s^{{-1}}$",
        'T_in': f"$ K $",
    }

    if param_name == 'length':
        param_def_val = float(config.get('geometry', {}).get("domains", [{}])[0].get('length', 1.0))
    elif param_name == 'G':
        param_def_val = float(config.get('source_terms', {}).get('concentration', {}).get('value', {}).get('mean', 0.0))
    elif param_name == 'T_in':
        param_def_val = float(config.get('boundary_conditions', {}).get('temperature', {}).get('left', {}).get('value', {}).get('mean', 0.0))
    else:
        raise NotImplementedError(f"Parameter {param_name} cannot be find in the config")

    # param_def_val = config.get(param_name, 1.0)

    if param_def_val is None:
        print(f"Warning: Default value for {param_name} not found in configuration, using 1.0")
        param_def_val = 1.0 
    
    print(f"Using default {param_name} value: {param_def_val:.3E} [{param_units[param_name]}]")
    # print(f"> Default value of {param_name} = {param_def_val}") ###DEBUG

    # Specify the list of parameter values to scan
    # Option 1) use logarithmic scale, specify range of magnitudes
    param_values = make_parameter_values_list(param_def_val, level_variation, scale=scale)

    # Option 2) use logarithmic scale, specify range of values

    # Option 3) use linear scale, specify range of values

    print(f"> List of {param_name} values for the scan:\n{param_values}") ###DEBUG

    results = []

    for value in param_values:
        print(f"\nRunning simulation with {param_name} = {value:.3E}")

        # Update the configuration with the specific results folder name
        config['simulation']['output_directory'] = f"{target_dir}/results_{param_name}_{value:.2e}"

        # Update the configuration with the current parameter value
        config[param_name] = value #TODO make fall back if config does not have this parameter or the structure is different
        
        # Specifically, for the length parameter
        if param_name == 'length':
            old_value = config['geometry']['domains'][0]['length']
            config['geometry']['domains'][0]['length'] = value
        elif param_name == 'G':
            old_value = config['source_terms']['concentration']['value']['mean']
            config['source_terms']['concentration']['value']['mean'] = value
        elif param_name == 'T_in':
            old_value = config['boundary_conditions']['temperature']['left']['value']['mean']
            config['boundary_conditions']['temperature']['left']['value']['mean'] = value
        else:
            raise NotImplementedError(f"Parameter {param_name} cannot be located in the config")

        # Change the solver tolarence according to the passes parameter values
        def_tt_atol = config['simulation']['tolerances']['absolute_tolerance']['tritium_transport']
        old_params = {param_name: old_value}
        new_params = {param_name: value}
        new_tt_atol = compute_absolute_tolerance(def_tt_atol, old_params, new_params)
        
        config['simulation']['tolerances']['absolute_tolerance']['tritium_transport'] = new_tt_atol
        print(f" > Changing tritium transport solver absolute tolerance from {def_tt_atol:.2E} to {new_tt_atol:.2E}") ###DEBUG

        # ATTENTION: overlaoding atol to 1.e0 for a test
        config['simulation']['tolerances']['absolute_tolerance']['tritium_transport'] = 1.0e+0

        # Create a model instance with the updated configuration
        model = Model(config=config)
        
        # Run the model and collect results
        result = model.run()

        print(f"Run for {param_name}={value:.3E} completed!")
        print(f"Result:\n{result}") ###DEBUG

        results.append(result)

        # Visualise results
        diagnostics = Diagnostics(model, results=result, result_folder=model.result_folder)
        diagnostics.visualise()

        print(f"Simulation with {param_name} = {value:.3E} completed.")

    print("Parameter scan completed!")

    # Plot the results as a dependency on the parameter value
    print("Plotting results...")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))

    # Select scalars to plot
    # Tritium concentration in centre
    results_toplot = [r['tritium_concentration'][0] for r in results]
    y_qoi_name = r"$C_T(r=0) [m^{{-3}}]$"

    # Compute the log-log slope of the scale
    if scale == 'log':
        r_scale_slope = (np.log10(results_toplot[-1]) - np.log10(results_toplot[0])) / (np.log10(param_values[-1]) - np.log10(param_values[0]))
    # Compute the lin-log slope of the scale    
    elif scale == 'linear':
        r_scale_slope = (np.log10(results_toplot[-1]) - np.log10(results_toplot[0])) / (param_values[-1] - param_values[0])
    else:
        raise NotImplementedError(f"Scale {scale} for scan no implemented")
    print(f"Slope of the log-log scale for {param_name}={r_scale_slope}")

    plt.plot(param_values, results_toplot, marker='o', label=y_qoi_name)

    plt.xscale(scale)  # Use logarithmic scale for x-axis
    plt.yscale('log')  # Use logarithmic scale for y-axis

    plt.xlabel(f"${param_name}$ [{param_units[param_name]}]")
    plt.ylabel(f"Value of QoI")
    plt.title(f"Parameter Scan: {param_name} - {param_explanation[param_name]}")

    plt.legend(loc='best')
    plt.grid(True)

    plt.savefig(f"{target_dir}/parameter_scan_{param_name}.png")
    plt.close()
    #TODO for the length scan specifically, map the results on the [0,1] normalised coordinate system, e.g. r = r/length

    print(f"Finished plotting results!")
    return results

def param_scan_sensitivity_analysis(config, param_name='length', level_variation=3, scale='log'):
    """
    Perform sensitivity analysis for the FESTIM model with a given parameter being scanned.
    This function is a placeholder for actual sensitivity analysis logic.
    """
    print(" !!! Performing sensitivity analysis SCAN ...")

    # Load the configuration
    if not config:
        # TODO read *_yaml.template file to get the default configuration
        print("No configuration provided for sensitivity analysis.")
        return

    # # Get the default value of the parameter from the configuration
    # param_def_val = config.get(param_name, 1.0)

    # if param_def_val is None:
    #     print(f"> Warning: Default value for {param_name} not found in configuration, using 1.0")
    #     param_def_val = 1.0 
    
    # Specifically, for the length parameter
    if param_name == 'length':
        param_def_val = float(config.get('geometry', {}).get('domains', [{}])[0].get('length', 1.0))
        print(f"Using default length value: {param_def_val} m")
        param_name = 'length'  # Use the full path to the parameter in the config

    # Specify the list of parameter values to scan

    # 1) use logarithmic scale, specify range of magnitudes
    param_values = make_parameter_values_list(param_def_val, level_variation, scale=scale)

    # Convert to list of floats
    param_values = [float(value) for value in param_values]

    # Custom list of parameter values (for a restart after an error)
    # param_values = [param_def_val * x for x in [1.0e+0, 1.0e+1, 1.0e+2, 1.0e+3]]
    # param_values = [param_def_val * x for x in [1.0e+3]]

    print(f"Parameter values for sensitivity analysis: {param_values}")

    if param_values is None or len(param_values) == 0:
        print("No sensitivity parameters found in configuration, no parameter range was generated, aborting scan")
        return

    # Perform sensitivity analysis logic here
    for i, param_value in enumerate(param_values):
        print(f"\n Iteration {i} of sensitivity analysis scan for {param_name} = {param_value} ...")
        # For now, just print the parameters

        # Change the solver tolarence according to the passes parameter values
        def_tt_atol = config['simulation']['tolerances']['absolute_tolerance']['tritium_transport']
        old_params = {param_name: param_def_val}
        new_params = {param_name: param_value}
        new_tt_atol = compute_absolute_tolerance(def_tt_atol, old_params, new_params)
        config['simulation']['tolerances']['absolute_tolerance']['tritium_transport'] = new_tt_atol
        # print(f" > Changing tritium transport solver absolute tolerance from {def_tt_atol:.2E} to {new_tt_atol:.2E}") ###DEBUG
        print(f"\n Computing new solver tolerances:..\n  Old parameter values: {param_name}={old_params[param_name]:.2E}\n  New parameter values: {param_name}={new_params[param_name]:.2E}\n    (Log difference: {np.log10(new_params[param_name] / old_params[param_name]):.2E})\n  Old tolerance value: {def_tt_atol:.2E}\n Computed tritium transport absolute tolerance: {new_tt_atol:.2E}\n")

        #ATTENTION

        #TODO make sure type conversion during iteration over numpy array is correct
        print(f" > Next: calling a UQ campaign")
        try:
            perform_uq_festim(
                config=config,
                fixed_params={
                    param_name: param_value,
                    "tritium_transport_absolute_tolerance": new_tt_atol, # to avoid solver issues during the scan
                }
            )
        except Exception as e:
            print(f"Error occurred while calling UQ campaign: {e}")
            # sys.exit(1)

        #TODO save and display the modified parameter in the scan
        print(f" > Sensitivity analysis iteration {i} for {param_name} = {param_value} completed.\n")
    
    return 0    

    # TODO make a plot of Sensitivity indices (at r=0) as a function of the parameter value (here: sample length); maybe a surface 3D plot

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run FESTIM model SCAN with a default YAML configuration')

    parser.add_argument('--config', '-c',
                       default='config/config.uq.yaml',
                       help='Path to YAML configuration file (default: config.uq.yaml)'
                       )

    parser.add_argument('--type', '-t',
                       default='single',
                       help='Options to perform scans: single (run a single simulation per parameter value) or uq (run a UQ campaign per parameter value) (default: single)'
                       )
    
    parser.add_argument('--parameter', '-p',
                       default='length',
                       help='Parameter which to vary during the scan (default: length))'
                       )

    parser.add_argument('--targetdir', '-d',
                       default='./',
                       help='Target directory where to save the results of runs'
                       )
    
    parser.add_argument('--scale', '-s',
                       default='log',
                       help='Scale to apply to the varying parameter'
                       )
    
    parser.add_argument('--steps', '-l',
                       default=1,
                       help='Steps to make in each direction of the scale, in basic untis of the scale(log: x10, linear: +/-10%)'
                       )
    
    args = parser.parse_args()

    # Load the configuration from the specified file
    config = load_config(args.config)

    # Get the type of scan from arguments
    scan_type = args.type.lower()

    # Get the paramter to scan
    param_name = args.parameter

    # Get the target directory to store the results
    target_dir = args.targetdir

    # Get the scale for the varying parameter
    scale = args.scale

    # Get the steps to make in scale, ot level of variation
    level_variation = int(args.steps)

    if config:

        if scan_type == "single":
            # Perform parameter scan
            parameter_scan(config, param_name=param_name, level_variation=level_variation, target_dir=target_dir, scale=scale)

        elif scan_type == "uq":
            # Perform UQ campaign scan
            param_scan_sensitivity_analysis(config, param_name=param_name, level_variation=level_variation)
        else:
            print(f"Unknown scan type: {scan_type}. Please use 'single' or 'uq'.")

    else:
        print("Failed to load configuration. Exiting.")