#!/home/yehor/miniconda3/envs/festim-env/bin/python3
import numpy as np

import matplotlib.pyplot as plt

# Import FESTIM library
import festim as F


class Model():
    """
    Model class for the FESTIM simulation.
    """

    def __init__(self, config=None):

        #super().__init__(**kwargs)

        self.name = "FESTIM Model"
        self.description = "Model for FESTIM simulation"

        # Create a FESTIM model instance
        self.model = F.Simulation()

        # Define model geometry and its mesh
        # 1D case
        # TODO figure out if there is sperical geometry support in FESTIM and apply for an isotropic case
        self.n_elements = config['simulation']['n_elements']
        self.length = config['geometry']['length']  # Length of the geometry in meters
        self.vertices = np.linspace(0., self.length, self.n_elements + 1)
        self.model.mesh = F.MeshFromVertices(vertices=self.vertices)

        self.results = None # Placeholder for results

        # Define model parameters: material properties
        self.model.materials = F.Material(
                id=1,
                D_0=config['materials']['D_0'],  # diffusion coefficient
                E_D=config['materials']['E_D'],  # activation energy
            )

        # Define model parameters: temperature
        self.model.T = config['materials']['T']

        # Define Boundary conditions
        #TODO for a sperical case, apply DirichletBC at boundary, NeumannBC at the center
        self.model.boundary_conditions = [
            F.DirichletBC(
                surfaces=[1],  # Assuming a single surface at the start of the mesh
                value=config['boundary_conditions']['left_bc_value'],  # boundary value
                field=0,
            ),
            F.FluxBC(
                surfaces=[2],  # Assuming a single surface at the end of the mesh
                value=config['boundary_conditions']['right_bc_value'],  # boundary value
                field=0,
            ),
        ]

        # Define model parameters: source terms
        self.model.source_terms = [
            F.Source(
                volume=1,
                value=config['source_terms']['source_value'],  # source term value
                field=0
            ),
        ]

        # Define model settings: solver and time
        self.model.settings = F.Settings(
            final_time=config['model_parameters']['total_time'],  # final time
            absolute_tolerance=config['simulation']['absolute_tolerance'],  #  absolute tolerance
            relative_tolerance=config['simulation']['relative_tolerance'],  #  relative tolerance
        )

        # Define exports: result format
        self.result_folder = config['simulation']['output_directory']
        self.milestone_times = config['simulation']['milestone_times']  # List of times to export results

        self.model.exports = [
            # F.XDMFExport(
            #     field=0,
            #     filename=f"{self.result_folder}/results.xdmf",
            #     checkpoint=True,
            # )
            F.TXTExport(
                field=0,
                filename=f"{self.result_folder}/results.txt",
                times=self.milestone_times,
            )
        ]
    
        # Define time step
        self.model.dt = F.Stepsize(config['simulation']['time_step'])  # time step size
        milestones=self.milestone_times  

    def run(self):
        """
        Run the FESTIM simulation.
        This method initializes the model, runs the simulation, and stores the results.
        """
        # Initialize the model
        self.model.initialise()

        # Run the simulation
        self.results = self.model.run()

        # Export results
        # self.model.export_results()

        return self.results

    def visualise(self):
        """
        Visualize the results of the FESTIM simulation.
        This method can be extended to include specific visualization logic.
        """
        if self.results is not None:

            print("> Visualizing results")

            data = np.genfromtxt(self.result_folder+"/results.txt", skip_header=1, delimiter=',')
            
            for (i,time) in enumerate(self.milestone_times):
                print(data[:, 0, data[:,i+1] == time])

            plt.xlabel("r [m]")
            plt.ylabel("Concentration [m^-3]")
            plt.title("Concentration vs Radius at Different Times")
            plt.legend([f"t={time}" for time in self.milestone_times])
            plt.show()

        else:
            print("No results to visualize. Please run the simulation first.")

        