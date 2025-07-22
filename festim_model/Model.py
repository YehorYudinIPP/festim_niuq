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

        # Specifying numerical parameters of mesh: physical size and number of elements
        self.n_elements = int(config['simulation']['n_elements'])
        self.length = float(config['geometry']['length'])  # Length of the geometry [m]

        # Create vertices for the mesh
        # Assuming a 1D geometry, vertices are evenly spaced along the length

        # Option 1) for vertices: uniform mesh
        # self.vertices = np.linspace(0., self.length, self.n_elements + 1)

        # When the BC uncertainty influence fall-off quickly, try refined mesh at the domain boundary
        # Option 2) mesh refined at the boundary (right side)
        refined_length_fraction = 0.1  # Fraction of the length to refine
        refined_elements_fraction = 0.25  # Fraction of elements to refine

        refined_elements_count = int(self.n_elements * refined_elements_fraction)
        self.vertices = np.concatenate((
            np.linspace(0., self.length * (1. - refined_length_fraction), self.n_elements - refined_elements_count + 1), # inner (larger) part of the mesh
            np.linspace(self.length * (1. - refined_length_fraction), self.length, refined_elements_count + 1)[1:] # refined (smaller outer) part of the mesh
        ))
        #TODO mind round-off errors in the mesh size

        print(f"Using vertices: {self.vertices}")  ### Debugging output

        # Create a Mesh object from the vertices

        # Option 1) for mesh: use FESTIM's MeshFromVertices
        self.model.mesh = F.MeshFromVertices(
            vertices=self.vertices,
            #type="cylindrical",  # Specify cylindrical mesh type
            type="spherical",  # Specify spherical mesh type
            )
        
        # Option 2) for mesh: use FESTIM's Mesh  - and FeniCS objects - specific for spherical geometry
        # self.model.mesh = F.Mesh(
        #     type="spherical",  # Specify spherical mesh type
        # )

        self.results = None # Placeholder for results

        # Define model parameters: material properties
        self.model.materials = F.Material(
                id=1,
                D_0=float(config['materials']['D_0']),  # diffusion coefficient
                E_D=float(config['materials']['E_D']),  # activation energy
            )
        # TODO: fetch data from HTM DataBase - LiO2 as an example

        # Define model parameters: temperature
        self.model.T = config['materials']['T']

        print(f"Using material properties: D_0={self.model.materials[0].D_0}, E_D={self.model.materials[0].E_D}, T={self.model.T.__dict__}") ###DEBUG

        # Define Boundary conditions
        # Sperical case, apply DirichletBC at boundary (in relative terms, r=1.0), NeumannBC (FluxBC) at the center (r=0.0)
        self.model.boundary_conditions = [
            F.FluxBC(
                surfaces=[1],  # Assuming a single surface at the end of the mesh
                value=float(config['boundary_conditions']['left_bc_value']),  # boundary value
                field=0,
            ),
            F.DirichletBC(
                surfaces=[2],  # Assuming a single surface at the start of the mesh
                value=float(config['boundary_conditions']['right_bc_value']),  # boundary value
                field=0,
            ),
        ]

        # Define model parameters: source terms
        #print(f"Passing the source term value: {config['source_terms']['source_value']}") ###DEBUG
        
        self.model.sources = [
            F.Source(
                value=float(config['source_terms']['source_value']),  # source term value [m^-3 s^-1]
                #value=1.0e20,  ###DEBUG
                volume=1,
                field=0,
            ),
        ]

        print(f"Using boundary value at outer surfaces: {self.model.boundary_conditions[1].__dict__}") ###DEBUG
        print(f"Using constant volumetric source term with values: {self.model.sources[0].__dict__}") ###DEBUG

        #TODO: Add model for temperature!

        # Define model numerical settings for a simulation: solver and time
        self.model.settings = F.Settings(
            final_time=float(config['model_parameters']['total_time']),  # final time
            absolute_tolerance=float(config['simulation']['absolute_tolerance']),  #  absolute tolerance
            relative_tolerance=float(config['simulation']['relative_tolerance']),  #  relative tolerance
        )
        #TODO: estimate good simulation time via diffusion and other transport coefficients, dimensions of domain, and source term etc.

        # Define exports: result format
        self.result_folder = config['simulation']['output_directory']
        self.milestone_times = config['simulation']['milestone_times']  # List of times to export results

        self.model.exports = [
            # F.XDMFExport(
            #     field=0,
            #     filename=f"{self.result_folder}/results.xdmf",
            #     checkpoint=True,
            # )
            # TODO figure out Python API for XDMF export and import
            # TODO look up / brush up Pythonic VTK API
            # TODO figure out HDF5 export and import
            # TODO chose result data format to store numerical tensor of data of dimansions [quantity x time x elements]
            F.TXTExport(
                field="solute",
                filename=f"{self.result_folder}/results.txt",
                times=self.milestone_times,
            )
        ]
    
        # Define time step
        self.model.dt = F.Stepsize(
            #float(config['simulation']['time_step']), # op1) fixed time step size

            initial_value=float(config['simulation']['time_step']),  # op2) initial time step size for adaptive dt
            stepsize_change_ratio=float(config['simulation']['stepsize_change_ratio']),  # ratio for adaptive time stepping
            #min_value=float(config['simulation']['min_time_step']),  # minimum time step size
            max_stepsize=float(config['simulation']['max_stepsize']),  # maximum time step size
            dt_min=1e-05,  # minimum time step size for adaptive time stepping

            milestones=self.milestone_times, # check points for results export
        ) 
        #TODO: since model convergence so quickly make time step adaptive, based on the model parameters and mesh size - exam the influence of the time step on the results

        #milestones=self.milestone_times  

        print(f"Initialisation finished! Model initialized with {self.n_elements} elements")  ###DEBUG

    def run(self):
        """
        Run the FESTIM simulation.
        This method initializes the model, runs the simulation, and stores the results.
        """
        # Initialize the model
        self.model.initialise()

        # Input and initalisation verification
        # self.inspect_object_structure()  ### Print the structure of the model for debugging

        # Run the simulation
        self.results = self.model.run()

        print("FESTIM simulation completed successfully!")

        # Output verification
        #print(f"Results: {self.results}") ### DEBUG
        self.result_flag = True

        # Export results
        # self.model.export_results()

        #TODO: Think of better BCs
        #TODO: Read Lithium data from HTM DataBase
        #TODO: Use proprietary visualisation and diagnostics tools
        #TODO: Explore cartesian/cylindrical/spherical geometries/coordinates/curvatures
        #TODO: Add important physical effects
        #TODO: Couple with heat conductivity and temperature

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

