
# A class to compute qualities of interest using FESTIM results

import numpy as np
import os
import sys

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt

import festim as F

class Diagnostics:

    def __init__(self, model, results=None, result_folder=None):
        """
        Initialize Diagnostics with FESTIM results.
        
        :param results: Results from the FESTIM simulation.
        """
        self.model = model
        self.results = results
        self.result_folder = result_folder if result_folder else './results' # TODO by default, try to read results from the model attribute

        # If result is none, read from the result folder
        if self.results is None:
            print(f"Diagnostics: No results provided, trying to read from {self.result_folder}/")
            self.results = {}

            #print(f">>> Diagnostics.model.quantities_of_interest: {self.model.quantities_of_interest}")  ###DEBUG print available quantities of interest

            # Read results for each quantity of interest
            for qoi,_ in self.model.quantities_of_interest.items():
                # Read results from the file if available
                print(f"Reading results for {qoi} from {self.result_folder}")

                result_file = os.path.join(self.result_folder, f'results_{qoi}.txt')

                # Check if the result file exists
                if os.path.exists(result_file):
                    self.results[qoi] = np.genfromtxt(result_file, skip_header=True, delimiter=',')
                    print(f"Results for {qoi} loaded from {result_file}")
        else:
            print(f"Warning: No results for {qoi} file found in {self.result_folder}. Please run the simulation first.")
            self.results[qoi] = None

        # Check if results are now present
        # TODO think if the flag is needed
        # self.result_flag = None  # Flag to check if results are available
        # if self.results is not None:   
        #     self.result_flag = True
        # else:
        #     self.result_flag = False

        # Read the milestone times from the configuration
        self.milestone_times = self.model.config.get('simulation', {}).get('milestone_times', [])
        if not self.milestone_times:
            print("Warning: No milestone times found in configuration. Using default values.")
            self.milestone_times = [1., 2., 3., 4., 5.]  # Default values for milestone times
        # Option 2) for milestone times: read from result file if available
        if self.milestone_times is None and self.results is not None and self.results.shape[1] > 1:
            # they are in the header of the results file 
            self.milestone_times = np.genfromtxt(result_file, max_rows=1, delimiter=',')[1:].tolist()

        # n_elem_print = 3
        # print(f">>> Diagnostics.__init__: Printing last {n_elem_print} elements of the results for last time of {self.milestone_times[-1]}: {self.results[-n_elem_print:, -1]}")  # Print last n elements of the results for the last time step ###DEBUG

        # Define data structure (dict) to keep naming, units etc. for quantities of interest
        self.quantities_of_interest_descriptor = {
            'tritium_inventory': {
                'name': 'Tritium Inventory',
                'unit': 'T',
                'description': 'Total tritium inventory in the sample'
            },
            'tritium_concentration': {
                'name': 'Tritium Concentration',
                'unit': 'm^-3',
                'description': 'Tritium concentration in the volume'
            },
            'temperature': {
                'name': 'Temperature',
                'unit': 'K',
                'description': 'Temperature distribution at a point'
            },
        }

    def compute_qoi(self, qoi_name):
        """
        Compute a quality of interest (QoI) from the results.
        
        :param qoi_name: Name of the QoI to compute.
        :return: Computed QoI value.
        """
        # Placeholder for actual QoI computation logic
        return self.results.get(qoi_name, None)

    # Function to compute total tritium inventory inside simulated volume - TODO try out different integration schemes
    def compute_total_tritium_inventory(self):
        """
        Compute the total tritium inventory in the simulated volume.

        :return: Total tritium inventory (FESTIM DerivedQuantities object).
        """
        # ATTENTION: Placeholeder - need to be passed during the model initialisation
        # there is an implementation in a wrapper

        derived_quantity_types = [
            F.TotalVolume(
                field=0, 
                volume=1,
                #name='tritium_inventory', 
                #description='Total tritium inventory in the volume'
            )
        ]

        derived_quantities = F.DerivedQuantities(
            derived_quantity_types,
            show_units=True,
        )

        return derived_quantities

    def _visualise_transient_quantity(self, qoi_name, qoi_values):
        """
        Visualize a specific quantity of interest.
        
        :param qoi_name: Name of the quantity to visualize.
        :param quantity: The quantity data to visualize.
        """
        if qoi_values is None:
            print(f"No data available for {qoi_name}. Skipping visualization.")
            return

        plt.figure(figsize=(10, 6))

        # iterate over the milestone times and plot each
        for (i,time) in enumerate(self.milestone_times):
            # The first column is the radial coordinate values
            # TODO: read file as a CSV file
            # TODO: consider that there might be no result data for a given milestone time
            print(f"> Plotting results for time {time} [s]")

            plt.plot(
                self.model.vertices[:], 
                qoi_values[:, i+1], 
                label=f"t={time:.2f} s", 
                #marker='o',
                )

            # n_el_print = 3
            # print(f"Last {n_el_print} elements at time {time} s: {self.results[-n_el_print:, i+1]}") ### DEBUG

        # Set plot labels and title
        plt.xlabel('Radial Coordinate [m]')
        plt.ylabel(f"{self.quantities_of_interest_descriptor[qoi_name]['name']} [{self.quantities_of_interest_descriptor[qoi_name]['unit']}]") # Use the name and unit from the descriptor

        plt.title(f"{self.quantities_of_interest_descriptor[qoi_name]['name']} vs Radius (at different times) \n Param-s: T={self.model.config['model_parameters']['T_0']:.2f} [K], G={float(self.model.config['source_terms']['source_concentration_value']):.2e} [m^-3s^-1], C(a)={float(self.model.config['boundary_conditions']['right_bc_concentration_value']):.2e} [m^-3]")

        plt.grid('both')
        plt.legend(loc='best')
        #plt.legend([f"t={time}" for time in self.milestone_times])

        #plt.show()
        plt.savefig(f"{self.result_folder}/results_{qoi_name}.png")
        plt.close('all')  # Close all figures after plotting
            
    def _visualise_steady_quantity(self, qoi_name, qoi_values):
        """
        Visualize a specific steady-state quantity of interest.
        
        :param qoi_name: Name of the quantity to visualize.
        :param qoi_values: The quantity data to visualize.
        """
        if qoi_values is None:
            print(f"No data available for {qoi_name}. Skipping visualization.")
            return

        plt.figure(figsize=(10, 6))

        # Plot the steady-state results
        plt.plot(self.model.vertices[:], qoi_values, label=f"Steady State", marker='o')

        # Set plot labels and title
        plt.xlabel('Radial Coordinate [m]')
        plt.ylabel(f"{self.quantities_of_interest_descriptor[qoi_name]['name']} [{self.quantities_of_interest_descriptor[qoi_name]['unit']}]")  # Use the name and unit from the descriptor
        plt.title(f"{self.quantities_of_interest_descriptor[qoi_name]['name']} vs Radius (in Steady State) \n Param-s: T={self.model.config['model_parameters']['T_0']:.2f} [K], G={float(self.model.config['source_terms']['source_value']):.2e} [m^-3s^-1], C(a)={float(self.model.config['boundary_conditions']['right_bc_value']):.2e} [m^-3]")

        plt.grid('both')
        plt.legend(loc='best')

        #plt.show()
        plt.savefig(f"{self.result_folder}/results_{qoi_name}_steady.png")
        plt.close('all')

    def visualise(self):
        """
        Visualize the results of the FESTIM simulation.
        This method can be extended to include specific visualization logic.
        """
        print("Visualizing results...")

        # if self.result_flag is not True:
        #     # Attempt to fall back and read from results.txt
        #     print("No results found in the object during visualisation, reading from file...")
        #     print(f">>> Diagnostics.visualise: Reading results from {self.result_folder}/results.txt")
        #     self.results = np.genfromtxt(self.result_folder+"/results.txt", skip_header=True, delimiter=',')

        if self.results is not None:

            print("> Visualizing results")

            # TODO possibility to specify subset of quantities to visualise
            #  - For now, we assume the first column is radial coordinates and the rest are concentrations at different times
            
            # Example: for basic temperature visualization at the last time step
            # from fenics import plot
            # plot(model.T.T, title="Temperature Distribution", mode='color', interactive=True)

            if self.model.model.settings.transient:
                print("Transient simulation detected, plotting results over time.")

                # Iterate over each quantity of interest in the results dictionary and plot
                for qoi_name, qoi_values in self.results.items():
                    print(f"Visualising quantity of interest: {qoi_name}")

                    if qoi_values is not None:
                        self._visualise_transient_quantity(qoi_name, qoi_values)
                    else:
                        print(f"No results available for {qoi_name}. Skipping visualization.")
            else:
                print("Steady-state simulation detected, only entry plotted.")
                # Plot the single plot (t=steady)
                for qoi_name, qoi_values in self.results.items():
                    print(f"Visualising quantity of interest: {qoi_name}")

                    # Check if the results for this quantity are available
                    if qoi_values is not None:
                        self._visualise_steady_quantity(qoi_name, qoi_values)
                    else:
                        print(f"No results available for {qoi_name}. Skipping visualization.")

                    #plt.plot(self.model.vertices[:], self.results[:, -1], label=f"t=steady")
                    #plt.semilogy(self.model.vertices[:], self.results[:, -1], label=f"t=steady")

        else:
            print("No results to visualize. Please run the simulation first.")
