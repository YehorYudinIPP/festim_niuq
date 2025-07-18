#!/home/yehor/miniconda3/envs/festim-env/bin/python3
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import numpy as np

# Import FESTIM library
import festim as F

# Local imports
from .diagnostics import Diagnostics

class Model():
    """
    Model class for the FESTIM simulation.
    """

    def __init__(self, config=None):

        #super().__init__(**kwargs)

        self.name = "FESTIM Model"
        self.description = "Model for FESTIM simulation"

        self.config = config if config else {}

        # Create a FESTIM model instance
        self.model = F.Simulation()

        # Define model geometry and its mesh
        # 1D case
        # TODO figure out if there is sperical geometry support in FESTIM and apply for an isotropic case
        self.n_elements = int(config['simulation']['n_elements'])
        self.length = float(config['geometry']['length'])  # Length of the geometry [m]
        # Create vertices for the mesh
        # Assuming a 1D geometry, vertices are evenly spaced along the length
        self.vertices = np.linspace(0., self.length, self.n_elements + 1)
        self.model.mesh = F.MeshFromVertices(vertices=self.vertices)

        self.results = None # Placeholder for results

        # Define model parameters: material properties
        self.model.materials = F.Material(
                id=1,
                D_0=float(config['materials']['D_0']),  # diffusion coefficient
                E_D=float(config['materials']['E_D']),  # activation energy
            )

        # Define model parameters: temperature
        self.model.T = config['materials']['T']

        print(f"Using material properties: D_0={self.model.materials[0].D_0}, E_D={self.model.materials[0].E_D}, T={self.model.T.__dict__}") ###DEBUG

        # Define Boundary conditions
        #TODO for a sperical case, apply DirichletBC at boundary, NeumannBC at the center
        self.model.boundary_conditions = [
            F.DirichletBC(
                surfaces=[1],  # Assuming a single surface at the start of the mesh
                value=float(config['boundary_conditions']['left_bc_value']),  # boundary value
                field=0,
            ),
            F.FluxBC(
                surfaces=[2],  # Assuming a single surface at the end of the mesh
                value=float(config['boundary_conditions']['right_bc_value']),  # boundary value
                field=0,
            ),
        ]

        # Define model parameters: source terms
        print(f"Passing the source term value: {config['source_terms']['source_value']}") ###DEBUG
        
        self.model.sources = [
            F.Source(
                value=float(config['source_terms']['source_value']),  # source term value [m^-3 s^-1]
                #value=1.0e20,  ###DEBUG
                volume=1,
                field=0,
            ),
        ]

        print(f"Using boundary value at outer surfaces: {self.model.boundary_conditions[0].__dict__}") ###DEBUG
        print(f"Using constant volumetric source term with values: {self.model.sources[0].__dict__}") ###DEBUG

        # Define model settings: solver and time
        self.model.settings = F.Settings(
            final_time=float(config['model_parameters']['total_time']),  # final time
            absolute_tolerance=float(config['simulation']['absolute_tolerance']),  #  absolute tolerance
            relative_tolerance=float(config['simulation']['relative_tolerance']),  #  relative tolerance
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
                field="solute",
                filename=f"{self.result_folder}/results.txt",
                times=self.milestone_times,
            )
        ]
    
        # Define time step
        self.model.dt = F.Stepsize(
            float(config['simulation']['time_step']),
            milestones=self.milestone_times,
        )  # time step size
        #milestones=self.milestone_times  

    def run(self):
        """
        Run the FESTIM simulation.
        This method initializes the model, runs the simulation, and stores the results.
        """
        # Initialize the model
        self.model.initialise()

        #  self.inspect_object_structure()  # Print the structure of the model for debugging

        # Run the simulation
        self.results = self.model.run()

        print("FESTIM simulation completed successfully!")
        #print(f"Results: {self.results}")
        self.result_flag = True

        # Export results
        # self.model.export_results()

        return self.results

    def inspect_object_structure(self):
        """Print detailed structure of the FESTIM model object."""
        print("FESTIM Model Structure:")
        print("=" * 50)
        
        # Model object structure
        print(f"Model type: {type(self.model)}")
        print(f"Model attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
        
        # Materials
        if hasattr(self.model, 'materials'):
            print(f"Materials: {len(self.model.materials)} materials")
            for i, mat in enumerate(self.model.materials):
                print(f"  Material {i}: {type(mat)} - {[attr for attr in dir(mat) if not attr.startswith('_')]}")
        
        # Boundary conditions
        if hasattr(self.model, 'boundary_conditions'):
            print(f"Boundary conditions: {len(self.model.boundary_conditions)} conditions")
            for i, bc in enumerate(self.model.boundary_conditions):
                print(f"  BC {i}: {type(bc)}")
        
        # Sources
        if hasattr(self.model, 'sources'):
            print(f"Sources: {len(self.model.sources)} sources")
            for i, source in enumerate(self.model.sources):
                print(f"  Source {i}: {type(source)}")

