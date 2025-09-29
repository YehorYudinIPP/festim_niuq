#!/home/yhy25yyp/anaconda3/envs/festim2-env/bin/python3
import sys
import os

# Set environment variables to suppress Qt warnings
os.environ['QT_LOGGING_RULES'] = '*.debug=false;qt.qpa.*=false'
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-' + str(os.getuid())

import numpy as np
import yaml
import argparse
from pathlib import Path

# Add parent directory to Python path to import festim_model
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    
# Now we can import festim_model from parent directory
from festim_model import Model, Model_legacy
from festim_model.diagnostics import Diagnostics

def load_config(config_file):
    """Load configuration from YAML file."""

    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None

def main():

    # Set up command line argument parsing
    print(f"\n\n ! Entering the main function: EasyVVUQ FESTIM model wrapper ! \n")

    parser = argparse.ArgumentParser(description='Run FESTIM model with YAML configuration')
    
    parser.add_argument('--config', '-c', 
                       default='config.yaml',
                       help='Path to YAML configuration file (default: config.yaml)')
    
    args = parser.parse_args()
    
    print(f"Using Python executable: {sys.executable}")
    print(f"Using FESTIM model from: {Path(__file__).parent.parent / 'festim_model'}")
    print(f"Using configuration file: {args.config}")
    print(f"Current working directory: {os.getcwd()}")

    # Load configuration from YAML file
    config = load_config(args.config)
    if config is None:
        print("No config file provided, quitting...")
        return
    
    print(f"Loaded configuration from: {args.config}")

    # print("Configuration parameters:")
    
    # # Access configuration values
    # model_params = config.get('model_parameters', {})
    # geometry = config.get('geometry', {})
    # materials = config.get('materials', {})
    # simulation = config.get('simulation', {})
    # boundary_conditions = config.get('boundary_conditions', {})
    
    # # Print some key parameters
    # print(f"  Temperature: {model_params.get('T_0', 'Not specified')} K")
    # print(f"  Time step: {simulation.get('time_step', 'Not specified')} s")
    # print(f"  Total time: {model_params.get('total_time', 'Not specified')} s")
    # print(f"  Material: {materials.get('material_name', 'Not specified')}")
    # print(f"  Sample length: {geometry.get('length', 'Not specified')} m")
    # print(f"  Number of elements: {simulation.get('n_elements', 'Not specified')}")

    festim_version = "2.0" # "1.4"
    
    # Create an instance of the FESTIM model with configuration
    if festim_version == "1.4":
        model = Model_legacy(config=config)
    else:
        model = Model(config=config)

    # Run the FESTIM model with configuration
    results = model.run()

    # n_elem_print = 3
    # print(f">>> festim_model_run: Printing last {n_elem_print} elements of the results for last time of {model.milestone_times[-1]}: {results[-n_elem_print:, -1]}")  # Print last n elements of the results for the last time step ###DEBUG
    print(f" >> Model run completed, resutls:\n{results}") ###DEBUG

    # Save results to a file (for EasyVVUQ integration)
    save_results_for_uq(results, model)
    #print(f">>> festim_model_run: Print results to the console: {results}") ###DEBUG

    # Visualise results
    # TODO make into a separate script such that if this fails, the model results are saved and run passes
    # TODO single out run and post-process scripts and run a single BASH script 

    # Option 1 - pass data from model to the diagnostics
    # diagnostics = Diagnostics(model=model, results=results, result_folder=model.result_folder, derived_quantities_flag=False)

    # Option 2 - read all the required information from results files
    qoi_names = ['tritium_concentration']
    diagnostics = Diagnostics(result_folder=config['simulation']['output_directory'], qoi_names=qoi_names, derived_quantities_flag=False, result_format='bp')

    diagnostics.visualise()
    
    print("FESTIM simulation completed successfully!")
    return results

def save_results_for_uq(results, model):
    """Save results in format expected by EasyVVUQ."""
    import json
    import csv
    
    # Extract quantities of interest (QoIs)
    # TODO: double-check the implementation; think of a good integration scheme
    #tritium_inventory = extract_tritium_inventory(results, model)
    tritium_inventory = 0.0 #ATTENTION: workaround for heat DEBUG

    # Save as CSV (with 0d scalar properties; here: tritium inventory) for EasyVVUQ decoder
    output_file = "output.csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['tritium_inventory'])  # Header
        writer.writerow([tritium_inventory])    # Data
    
    print(f"Results saved to {output_file}")
    print(f"Tritium inventory: {tritium_inventory:.2e}")

    # Save TXT files with 1D profiles (if available) from results
    # ATTENTION: mimics TXTExport from FESTIM1.4
    for qoi_name, qoi_values in results.items():

        profile_folder_name = model.result_folder
        profile_file_name = f"results_{qoi_name}.txt"
        profile_file_path = os.path.join(profile_folder_name, profile_file_name)

        grid_values = model.vertices  # Assuming model has an attribute 'vertices' for the spatial grid
        data = np.column_stack((grid_values, qoi_values))

        np.savetxt(profile_file_path, data, header=f"x,t=steady", delimiter=',', comments='')

        print(f" > Profile {qoi_name} saved to {profile_file_path}") ###DEBUG

def extract_tritium_inventory(results, model):
    """Extract tritium inventory from FESTIM results."""
    # This is a most primitive version - implement based on your specific FESTIM model
    # You might need to:
    # 1. Read the output files generated by FESTIM
    # 2. Integrate concentration over the domain
    # 3. Calculate total inventory
    
    if results is not None and model is not None:
        # Implement extraction logic using results and model
        print("Extracting tritium inventory from results in the model ...")
        data = results.get('tritium_concentration', None)
    elif results is None or model is None:
        try:
            # Example: Read from the results file
            print(f" Reading tritium concentration from: {result_file} ...")
            result_folder = model.result_folder
            result_file_name = "results_tritium_concentration.txt"
            result_file = os.path.join(result_folder, result_file_name)

            if os.path.exists(result_file):
                # Read and process the results file
                data = np.genfromtxt(result_file, skip_header=1, delimiter=',')

            else:
                print(f"Warning: Results file not found: {result_file}")
                inventory = 1.0e20  # Default value
                return inventory
            
        except Exception as e:
            print(f"Error extracting tritium inventory: {e}")
            inventory = 1.0e20  # Default value

    # Option 1) simple example: sum of final concentrations
    #TODO: Replace with a better inventory calculation

    ds = 1.0e-12 # a test area of a squared micron [m^2]
    length_elem_s = model.vertices[1:] - model.vertices[:-1]  # length of each (1D) element [m]

    # # Option 1.1) assume flat geometry and uniform (1D) mesh (~hexahedral in 3D)
    # length_elem = model.config['geometry']['length'] / model.config['simulation']['n_elements']  # Example length element [m]
    # volume_elem = ds * length_elem  # Volume of an element [m^3]
    # if len(data.shape) > 1 and data.shape[0] > 0:
    #     final_concentrations = data[:, -1]  # Last time step
    #     inventory = np.sum(final_concentrations) * volume_elem  
    # else:
    #     inventory = 1.0e20  # Default value

    # # Option 1.2) assume flat geometry and non-uniform mesh
    # # Calculate volume of an actual local element - important for a) sph. geometry and b) non-uniform mesh
    # #TODO test this
    # volume_elem_s = ds * length_elem_s  # Volume of each element [m^3]
    
    # #TODO figure out correct dimensionality of the output data
    # if len(data.shape) > 1 and data.shape[0] > 0:
    #     final_concentrations = data[:-1, -1]  # Last time step
    #     #TODO find correct resolution beteen vertices and elements
    #     inventory = np.sum(final_concentrations * volume_elem_s)
    # else:
    #     inventory = 1.0e20  # Default value

    # Option 1.3) assume spherical geometry and uniform mesh
    #TODO: test this

    radius_loc_s = model.vertices[:-1]  # Local radius of each spherical element [m]

    volume_elem_s =  4. * np.pi * length_elem_s * (radius_loc_s)**2  # Volume of a spherical layer element [m^3]

    if len(data.shape) > 0 and data.shape[0] > 0:

        #final_concentrations = data[:-1, -1]  # Last time step

        if len(data.shape) == 1:
            final_concentrations = data[-1]
        elif len(data.shape) > 1:
            final_concentrations = data[:, -1]
        else:
            # fall back option
            print("Warning: Unexpected data shape, using default tritium inventory value.")
            unit_concentration = 1.0e10  # Default value
            final_concentrations = unit_concentration * np.ones(data.shape[0])  # Default to ones if data is empty

        inventory = np.trapz(final_concentrations * volume_elem_s)
    else:
        print("Using default value for tritium inventory")
        inventory = 1.0e20  # Default value

    # TODO: figure out how to use FESTIM's DerivedQuantities to compute inventory
    #diagnostics = Diagnostics(model, results=results, result_folder=model.result_folder)
    #tritium_inventory_obj = diagnostics.compute_total_tritium_inventory()
    
    return inventory


if __name__ == "__main__":
    main()
