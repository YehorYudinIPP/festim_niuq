
# TODO make a class to compute qualities of interest using FESTIM results

import numpy as np
import os
import sys

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt

class Diagnostics:

    def __init__(self, model, results=None, result_folder=None):
        """
        Initialize Diagnostics with FESTIM results.
        
        :param results: Results from the FESTIM simulation.
        """
        self.model = model
        self.results = results
        self.result_folder = result_folder if result_folder else './results'

        # If result is none, read from the result folder
        if self.results is None:
            result_file = os.path.join(self.result_folder, 'results.txt')
            if os.path.exists(result_file):
                self.results = np.genfromtxt(result_file, skip_header=True, delimiter=',')
            else:
                print(f"Warning: Results file not found: {result_file}")
                self.results = None

        # Check if results are now present
        self.result_flag = None  # Flag to check if results are available
        if self.results is not None:   
            self.result_flag = True
        else:
            self.result_flag = False

        # Read the milestone times from the configuration
        self.milestone_times = self.model.config.get('simulation', {}).get('milestone_times', [])
        if not self.milestone_times:
            print("Warning: No milestone times found in configuration. Using default values.")
            self.milestone_times = [0, 10, 20, 30]
        # Option 2) for milestone times: read from result file if available
        if self.milestone_times is None and self.results is not None and self.results.shape[1] > 1:
            # they are in the header of the results file 
            self.milestone_times = np.genfromtxt(result_file, max_rows=1, delimiter=',')[1:].tolist()

    def compute_qoi(self, qoi_name):
        """
        Compute a quality of interest (QoI) from the results.
        
        :param qoi_name: Name of the QoI to compute.
        :return: Computed QoI value.
        """
        # Placeholder for actual QoI computation logic
        return self.results.get(qoi_name, None)
    
    #TODO function to compute total tritium inverntory inside simualted volume
    def compute_total_tritium_inventory(self):
        """
        Compute the total tritium inventory in the simulated volume.
        
        :return: Total tritium inventory.
        """
        # Assuming results has a 'tritium_inventory' key with the inventory data
        # ATTENTION: Placeholeder (there is an implementation in a wrapper)
        return self.results.get('tritium_inventory', 0.0)
    
    def visualise(self):
        """
        Visualize the results of the FESTIM simulation.
        This method can be extended to include specific visualization logic.
        """
        print("Visualizing results...")

        if self.result_flag is not True:
            # Attempt to fall back and read from results.txt
            print("No results found in the object during visualisation, reading from file...")
            print(f"Reading results from {self.result_folder}/results.txt")
            self.results = np.genfromtxt(self.result_folder+"/results.txt", skip_header=True, delimiter=',')

        if self.result_flag is not None:

            print("> Visualizing results")
   
            for (i,time) in enumerate(self.milestone_times):
                print(f"> Reading results for time {time} s")
                plt.plot(self.model.vertices[:], self.results[:, i], label=f"t={time} s")

            plt.xlabel("r [m]")
            plt.ylabel("Concentration [m^-3]")
            plt.title(f"Concentration vs Radius at different times. \n Param-s: T={self.model.config['materials']['T']:.2f} K, G={float(self.model.config['source_terms']['source_value']):.2e} m^-3s^-1, C(a)={float(self.model.config['boundary_conditions']['left_bc_value']):.2e} m^-3")
            plt.legend([f"t={time}" for time in self.milestone_times])
            #plt.show()

            plt.savefig(f"{self.result_folder}/results.png")

        else:
            print("No results to visualize. Please run the simulation first.")
