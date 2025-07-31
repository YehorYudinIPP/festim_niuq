
# A class to compute qualities of interest using FESTIM results

import numpy as np
import pandas as pd
import os
import sys

import csv

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for headless environments
import matplotlib.pyplot as plt

import festim as F

class Diagnostics:

    def __init__(self, model, results=None, result_folder=None, derived_quantities_flag=True):
        """
        Initialize Diagnostics with FESTIM results.
        
        :param results: Results from the FESTIM simulation.
        """
        self.model = model
        self.results = results
        self.result_folder = result_folder if result_folder else './results' # TODO by default, try to read results from the model attribute
        self.mesh = {}  # Dictionary to store mesh coordinates for each quantity of interest

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

                    # TODO (1) read using pandas (2) keep only the last column of the results, which is the time series
                    self.mesh[qoi] = self.results[qoi][:, 0]  # Assuming the first column is the mesh coordinates
                    self.results[qoi] = self.results[qoi][:, 1:]  # Keep only the results data (excluding mesh coordinates)
        else:
            print(f"Warning: No results for {qoi} file found in {self.result_folder}. Please run the simulation first.")
            self.results[qoi] = None

        # Additionally, read results from derived quantities file if available
        if derived_quantities_flag:
            derived_quantities_file = os.path.join(self.result_folder, 'derived_quantities.csv')

            alternative_names = {
                'Average solute volume 1 (H m-3)': 'tritium_inventory',
            }

            if os.path.exists(derived_quantities_file):
                print(f"Reading derived quantities from {derived_quantities_file}")

                # Read entire derived quantities file
                derived_quantities = np.genfromtxt(derived_quantities_file, delimiter=',', skip_header=1)
                #print(f" >> Derived quantities shape: {derived_quantities.shape}")  ###DEBUG print shape of derived quantities

                # If the resulting numpy array is 1D, reshape it to 2D
                if derived_quantities.ndim == 1:
                    derived_quantities = derived_quantities.reshape(-1, 1)

                # Read header to get column names - TODO replace with pandas for better handling
                with open(derived_quantities_file, 'r') as f:
                    header = f.readline().strip().split(',')
                
                # Add each column of derived_quantities to the results dictionary
                for i, qoi in enumerate(header):
                    # Check if the quantity name is in the alternative names mapping and replace it
                    print(f" >> Processing derived quantity: {qoi}") ###DEBUG
                    if qoi in alternative_names:
                        qoi = alternative_names[qoi]
                        # ATTENTION: so far, only the quantities specified in the alternative_names mapping are added
                        if i < derived_quantities.shape[1]:  # Check if column exists
                            self.results[qoi] = derived_quantities[:, i]
                            print(f"Loaded derived quantity: {qoi}")

                # Store the times for derived quantities separately
                self.derived_quantities_times = derived_quantities[:, 0] if derived_quantities.shape[1] > 0 else []
                print(f" > Derived quantities times: {self.derived_quantities_times}") ###DEBUG print derived quantities times

            else:
                print(f"No derived quantities file found at {derived_quantities_file}.")
                self.derived_quantities = None

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
                'dimensionality': '0d',
                'description': 'Total tritium inventory in the sample'
            },
            'tritium_concentration': {
                'name': 'Tritium Concentration',
                'unit': 'm^-3',
                'dimensionality': '1d',
                'description': 'Tritium concentration in the volume'
            },
            'temperature': {
                'name': 'Temperature',
                'unit': 'K',
                'dimensionality': '1d',
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
    @staticmethod
    def compute_total_tritium_inventory(result_folder):
        """
        Compute the total tritium inventory in the simulated volume.

        :return: Total tritium inventory (FESTIM DerivedQuantities object).
        """
        # ATTENTION: Placeholder - need to be passed during the model initialisation
        # Also, there is an implementation to recompute quantities (gas inventory) based on the raw outputs in the UQ wrapper

        derived_quantity_types = [
            F.TotalVolume(
                field="retention", 
                volume=1,
                #name='tritium_inventory', 
                #description='Total tritium inventory in the volume'
            ),
            F.HydrogenFlux(
                surface=1, 
            ),
            F.HydrogenFlux(
                surface=2, 
            ),
            F.TotalVolume(
                field="solute",
                volume=1,
            ),
            F.AverageVolume(
                field="solute",
                volume=1,
            ),
        ]

        # Make a derived quantities object
        derived_quantities = F.DerivedQuantities(
            derived_quantity_types,
            show_units=True,
            filename=result_folder+'/derived_quantities.csv',
        )
        
        # Append the new exports
        #model.exports.append(derived_quantities)

        return derived_quantities

    def _visualise_transient_0d_quantity(self, qoi_name, qoi_values, times):
        """
        Visualise a specific scalar (0D) quantity of interest resolved as a function of time.
        """
        if qoi_values is None:
            print(f"No data available for {qoi_name}. Skipping visualization.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the quantity of interest over time
        ax.plot(times, qoi_values, label=qoi_name)

        # Set plot labels and title
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f"{self.quantities_of_interest_descriptor[qoi_name]['name']} [{self.quantities_of_interest_descriptor[qoi_name]['unit']}]")
        ax.set_title(f"{self.quantities_of_interest_descriptor[qoi_name]['name']} vs Time")

        ax.grid('both')
        ax.legend(loc='best')

        # Save the plot to the result folder
        fig.savefig(f"{self.result_folder}/results_{qoi_name}.png")
        plt.close('all')

    def _visualise_transient_1d_quantity(self, qoi_name, qoi_values):
        """
        Visualize a specific quantity of interest resolved as a function of a single spatial coordinate (1D).
        
        :param qoi_name: Name of the quantity to visualize.
        :param quantity: The quantity data to visualize.
        """
        if qoi_values is None:
            print(f"No data available for {qoi_name}. Skipping visualization.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # iterate over the milestone times and plot each
        for (i,time) in enumerate(self.milestone_times):
            # The first column is the radial coordinate values
            # TODO: read file as a CSV file
            # TODO: consider that there might be no result data for a given milestone time
            print(f"> Plotting results for time {time} [s]")

            ax.plot(
                self.model.vertices[:], 
                qoi_values[:, i], 
                label=f"t={time:.2f} s", 
                #marker='o',
                )

            # n_el_print = 3
            # print(f"Last {n_el_print} elements at time {time} s: {self.results[-n_el_print:, i+1]}") ### DEBUG

        # Set plot labels and title
        ax.set_xlabel('Radial Coordinate [m]')
        ax.set_ylabel(f"{self.quantities_of_interest_descriptor[qoi_name]['name']} [{self.quantities_of_interest_descriptor[qoi_name]['unit']}]") # Use the name and unit from the descriptor

        title_string = f"{self.quantities_of_interest_descriptor[qoi_name]['name']} vs Radius (at different times)"

        # TODO make a descriptor file in a separate package for YAML parsing; storing and specifying its structure
        title_string += f"\n Param-s: T={self.model.config['model_parameters']['T_0']:.2f} [K], G={float(self.model.config['source_terms']['source_concentration_value']):.2e} [m^-3s^-1], C(a)={float(self.model.config['boundary_conditions']['right_bc_concentration_value']):.2e} [m^-3]"

        ax.set_title(title_string)

        ax.grid('both')
        ax.legend(loc='best')
        #plt.legend([f"t={time}" for time in self.milestone_times])

        #plt.show()
        fig.savefig(f"{self.result_folder}/results_{qoi_name}.png")
        plt.close('all')  # Close all figures after plotting
            
    def _visualise_steady_1d_quantity(self, qoi_name, qoi_values):
        """
        Visualize a specific steady-state quantity of interest resolved as a function of a single spatial coordinate (1D).

        :param qoi_name: Name of the quantity to visualize.
        :param qoi_values: The quantity data to visualize.
        """
        if qoi_values is None:
            print(f"No data available for {qoi_name}. Skipping visualization.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        print(f" >> qoi_values.shape: {qoi_values.shape}")  # DEBUG print shape of qoi_values

        # Plot the steady-state results
        ax.plot(
            self.model.vertices[:], 
            qoi_values,
            label=f"{self.quantities_of_interest_descriptor[qoi_name]['name']} in Steady State", 
            #marker='o'
            )

        # Set plot labels and title
        ax.set_xlabel('Radial Coordinate [m]')
        ax.set_ylabel(f"{self.quantities_of_interest_descriptor[qoi_name]['name']} [{self.quantities_of_interest_descriptor[qoi_name]['unit']}]")  # Use the name and unit from the descriptor

        title_string = f"{self.quantities_of_interest_descriptor[qoi_name]['name']} vs Radius (in Steady State)"
        title_string += f"\n Param-s: T={self.model.config['model_parameters']['T_0']:.2f} [K], G={float(self.model.config['source_terms']['source_concentration_value']):.2e} [m^-3s^-1], C(a)={float(self.model.config['boundary_conditions']['right_bc_concentration_value']):.2e} [m^-3]"
        ax.set_title(title_string)

        ax.grid('both')
        ax.legend(loc='best')

        #plt.show()
        fig.savefig(f"{self.result_folder}/results_{qoi_name}_steady.png")
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
                        if self.quantities_of_interest_descriptor[qoi_name]['dimensionality'] == '1d':
                            self._visualise_transient_1d_quantity(qoi_name, qoi_values)
                        elif self.quantities_of_interest_descriptor[qoi_name]['dimensionality'] == '0d':
                            self._visualise_transient_0d_quantity(qoi_name, qoi_values, self.derived_quantities_times)
                        else:
                            print(f"Quantity {qoi_name} is not 1D or 0D, or unspecified, skipping visualisation.")
                            # Optionally, handle other dimensionalities or skip
                    else:
                        print(f"No results available for {qoi_name}. Skipping visualization.")
            else:
                print("Steady-state simulation detected, only single entry plotted.")
                # Plot the single plot (t=steady)
                for qoi_name, qoi_values in self.results.items():
                    print(f"Visualising quantity of interest: {qoi_name}")

                    # Check if the results for this quantity are available
                    if qoi_values is not None:
                        if self.quantities_of_interest_descriptor[qoi_name]['dimensionality'] == '1d':
                            # Visualise as a 1D quantity
                            self._visualise_steady_1d_quantity(qoi_name, qoi_values)
                        else:
                            print(f"Quantity {qoi_name} is not 1D, skipping steady-state visualisation.")
                            # Optionally, handle other dimensionalities or skip
                    else:
                        print(f"No results available for {qoi_name}. Skipping visualization.")

                    #plt.plot(self.model.vertices[:], self.results[:, -1], label=f"t=steady")
                    #plt.semilogy(self.model.vertices[:], self.results[:, -1], label=f"t=steady")

        else:
            print("No results to visualize. Please run the simulation first.")
