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

    try:
        # Option 2 - read all the required information from results files
        qoi_names = ['tritium_concentration']
        diagnostics = Diagnostics(result_folder=config['simulation']['output_directory'], qoi_names=qoi_names, derived_quantities_flag=False, result_format='bp')

        diagnostics.visualise()
    except Exception as e:
        print(f"Warning: Diagnostics visualisation failed: {e}")
        print("Results are saved; UQ pipeline can continue.")
    
    print("FESTIM simulation completed successfully!")
    return results

def save_results_for_uq(results, model):
    """Save scalar QoIs and preserve transient profile exports for EasyVVUQ."""
    import csv

    # ---- scalar QoIs in output.csv ----
    total_release = results.get('total_tritium_release', None)
    if isinstance(total_release, np.ndarray):
        final_release = float(total_release[-1]) if len(total_release) > 0 else 0.0
    elif total_release is not None:
        final_release = float(total_release)
    else:
        final_release = 0.0

    total_trapping = results.get('total_tritium_trapping', None)
    if isinstance(total_trapping, np.ndarray):
        final_trapping = float(total_trapping[-1]) if len(total_trapping) > 0 else 0.0
    elif total_trapping is not None:
        final_trapping = float(total_trapping)
    else:
        final_trapping = extract_tritium_inventory(results, model)

    output_file = "output.csv"
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['total_tritium_release', 'total_tritium_trapping'])
        writer.writerow([final_release, final_trapping])

    print(f"Results saved to {output_file}")
    print(f"Total tritium release (final): {final_release:.2e}")
    print(f"Total tritium trapping (final): {final_trapping:.2e}")

    # ---- profile CSV (should already exist from Model._export_results) ----
    milestone_times = getattr(model, 'milestone_times', []) or []

    for qoi_name, qoi_values in results.items():
        if qoi_name == 'total_tritium_release':
            continue

        profile_folder_name = model.result_folder
        profile_file_name = f"results_{qoi_name}.txt"
        profile_file_path = os.path.join(profile_folder_name, profile_file_name)

        if os.path.exists(profile_file_path):
            print(f" > Keeping existing profile export: {profile_file_path}")
            continue

        # Fallback: create profile file from in-memory results
        grid_values = np.asarray(model.vertices)
        qoi_array = np.asarray(qoi_values)

        if qoi_array.ndim == 1:
            qoi_array = qoi_array.reshape(-1, 1)
        elif qoi_array.ndim > 2:
            qoi_array = qoi_array.reshape(qoi_array.shape[0], -1)

        data = np.column_stack((grid_values, qoi_array))

        if qoi_array.shape[1] == 1 and not milestone_times:
            header = "x,t=steady"
        else:
            if len(milestone_times) == qoi_array.shape[1]:
                time_headers = [f"t={float(t):.2e}s" for t in milestone_times]
            else:
                time_headers = [f"t_idx_{i}" for i in range(qoi_array.shape[1])]
            header = "x," + ",".join(time_headers)

        os.makedirs(profile_folder_name, exist_ok=True)
        np.savetxt(profile_file_path, data, header=header, delimiter=',', comments='')

        print(f" > Fallback profile {qoi_name} saved to {profile_file_path}")

    # ---- plot concentration vs time for this individual run ----
    plot_concentration_vs_time(results, model)

def plot_concentration_vs_time(results, model):
    """Plot concentration at selected spatial positions as a function of time.

    Creates one plot with concentration vs time at the sample centre, middle,
    and right boundary.  If total tritium release data is available it is
    plotted on a second figure.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    milestone_times = getattr(model, 'milestone_times', []) or []
    if not milestone_times:
        print("No milestone times defined, skipping concentration vs time plots.")
        return

    tritium_conc = results.get('tritium_concentration', None)
    if tritium_conc is None:
        print("No tritium concentration data in results, skipping plots.")
        return

    conc_array = np.asarray(tritium_conc)
    if conc_array.ndim == 1:
        print("Only steady-state data available, skipping concentration vs time plots.")
        return

    vertices = np.asarray(model.vertices)

    # Positions of interest along the sample axis
    r_indices = {
        'center (r=0)': 0,
        'middle': len(vertices) // 2,
        'right boundary': -1,
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, r_idx in r_indices.items():
        c_vs_t = conc_array[r_idx, :]
        r_val = vertices[r_idx]
        ax.plot(milestone_times, c_vs_t, 'o-', label=f'{label} (r={r_val:.4e} m)')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Tritium Concentration [m$^{-3}$]')
    ax.set_title('Tritium Concentration at Sample Axis vs Time')
    ax.legend(loc='best')
    ax.grid(True)
    if all(t > 0 for t in milestone_times):
        ax.set_xscale('log')

    plot_folder = model.result_folder
    os.makedirs(plot_folder, exist_ok=True)
    plot_path = os.path.join(plot_folder, 'concentration_vs_time.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f" > Concentration vs time plot saved to {plot_path}")

    # Total tritium release vs time
    total_release = results.get('total_tritium_release', None)
    if total_release is not None:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        ax2.plot(milestone_times, total_release, 'o-', color='red',
                 label='Total Tritium Release')
        ax2.set_xlabel('Time [s]')
        ax2.set_ylabel('Total Tritium Release')
        ax2.set_title('Total Tritium Release vs Time')
        ax2.legend(loc='best')
        ax2.grid(True)
        if all(t > 0 for t in milestone_times):
            ax2.set_xscale('log')
        release_plot = os.path.join(plot_folder, 'total_tritium_release_vs_time.png')
        fig2.savefig(release_plot, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f" > Total tritium release vs time plot saved to {release_plot}")


def extract_tritium_inventory(results, model):
    """Extract tritium inventory from FESTIM results."""

    if results is not None and model is not None:
        data = results.get('tritium_concentration', None)
    else:
        try:
            result_folder = model.result_folder
            result_file_name = "results_tritium_concentration.txt"
            result_file = os.path.join(result_folder, result_file_name)
            print(f" Reading tritium concentration from: {result_file} ...")

            if os.path.exists(result_file):
                data = np.genfromtxt(result_file, skip_header=1, delimiter=',')
            else:
                print(f"Warning: Results file not found: {result_file}")
                return 1.0e20
        except Exception as e:
            print(f"Error extracting tritium inventory: {e}")
            return 1.0e20

    if data is None:
        print("No concentration data available, returning default inventory.")
        return 1.0e20

    # Integrate concentration over the domain
    vertices = np.asarray(model.vertices)
    conc = np.asarray(data)

    # Use the last time-step if data is 2-D
    if conc.ndim > 1:
        conc = conc[:, -1]

    coordinate_system = getattr(model, 'coordinate_system_type', 'cartesian')
    if coordinate_system == "spherical":
        weight = 4.0 * np.pi * vertices ** 2
    elif coordinate_system == "cylindrical":
        weight = 2.0 * np.pi * vertices
    else:
        weight = np.ones_like(vertices)

    if len(conc) == len(vertices):
        inventory = float(np.trapz(conc * weight, x=vertices))
    else:
        print("Warning: size mismatch between concentration and vertices, using default.")
        inventory = 1.0e20

    return inventory


if __name__ == "__main__":
    main()
