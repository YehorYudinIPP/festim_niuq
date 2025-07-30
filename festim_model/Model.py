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

        self.results = None # Placeholder for results
        # TODO find a way to fill this in from FESTIM Model object

        self.quantities_of_interest = {
            'tritium_concentration': None,  # Placeholder for tritium concentration
        }  # Dictionary to store quantities of interest (QoI)

        # Define model geometry and its mesh
        self._specify_geometry(config)

        # Define material properties
        self._specify_materials(config)

        # Define if the model is transient or steady-state - includes model numerical setting definition
        if 'transient' in config['model_parameters'] and config['model_parameters']['transient'] is True:
            
            # Define model numerical settings for a simulation: solver and time
            #TODO: estimate good simulation time via diffusion and other transport coefficients, dimensions of domain, and source term etc.

            self.model.settings = F.Settings(
                transient=True,  # Enable transient simulation
                final_time=float(config['model_parameters']['total_time']),  # final time of the simulation [s]
                absolute_tolerance=float(config['simulation']['absolute_tolerance']),  #  absolute tolerance
                relative_tolerance=float(config['simulation']['relative_tolerance']),  #  relative tolerance
            )
                    
            print("Model is set to transient simulation.")
        else:

            self.model.settings = F.Settings(
                transient=False,  # Enable transient simulation
                absolute_tolerance=float(config['simulation']['absolute_tolerance']),  #  absolute tolerance
                relative_tolerance=float(config['simulation']['relative_tolerance']),  #  relative tolerance
            )

            print("Model is set to steady-state simulation.")

        # Define model parameters: temperature (and heat transfer model)
        if 'heat_model' in config['model_parameters'] and config['model_parameters']['heat_model'] == 'heat_transfer':
            # Use heat transfer model
            self._add_heat_conduction(config)
            #TODO test thoroughly!
        else:
            self.model.T = config['model_parameters']['T_0'] # set fixed background temperature

        print (f" >> Using heat transfer model: {self.model.T.__dict__}") ###DEBUG

        # Define Boundary conditions
        self._specify_boundary_conditions(config)

        # Define source terms
        self._add_source_terms(config)

        # Define derived quantities - TODO implement
        #self.derived_quantities = self.define_derived_quantities()

        # Define exports: result format
        self._specify_outputs(config)

        # Define time stepping (for transient simulation)
        if self.model.settings.transient:
            self._specify_time_integration_settings(config)
        else:
            self.model.dt = None
        #TODO: since model convergence so quickly make time step adaptive, based on the model parameters and mesh size - exam the influence of the time step on the results

        # Added derived quantities to the model exports
        self._add_derived_quantities(['tritium_inventory'])

        print(f" > Initialisation finished! Model initialized with {self.n_elements} elements")  ###DEBUG

    def _specify_geometry(self, config):
        """
        Specify the geometry of the FESTIM model.
        This method can be extended to include specific geometry logic.
        """
        print("Specifying geometry...")
        # 1D case

        # Specifying numerical parameters of mesh: physical size and number of elements
        self.n_elements = int(config['simulation']['n_elements'])
        self.length = float(config['geometry']['length'])  # Length of the geometry [m]

        self.coordinate_system_type = config['geometry'].get('coordinate_system', 'spherical')  # Coordinate system type (default: spherical)

        # Create vertices for the mesh
        # Assuming a 1D geometry, vertices are evenly spaced along the length

        # Option 1) for vertices: uniform mesh
        self.vertices = np.linspace(0., self.length, self.n_elements + 1)

        # When the BC uncertainty influence fall-off quickly, try refined mesh at the domain boundary
        # Option 2) mesh refined at the boundary (right side)
        refined_length_fraction = 0.1  # Fraction of the length to refine
        refined_elements_fraction = 0.25  # Fraction of elements to refine

        refined_elements_count = int(self.n_elements * refined_elements_fraction)
        # self.vertices = np.concatenate((
        #     np.linspace(0., self.length * (1. - refined_length_fraction), self.n_elements - refined_elements_count + 1), # inner (larger) part of the mesh
        #     np.linspace(self.length * (1. - refined_length_fraction), self.length, refined_elements_count + 1)[1:] # refined (smaller outer) part of the mesh
        # ))
        #TODO mind round-off errors in the mesh size

        #print(f"Using vertices: {self.vertices}")  ### Debugging output

        # Create a Mesh object from the vertices

        # Option 1) for mesh: use FESTIM's MeshFromVertices
        self.model.mesh = F.MeshFromVertices(
            type=self.coordinate_system_type,  # Specify (spherical) mesh type; available coordinate systems: 'cartesian', 'cylindrical', 'spherical'; default is Cartesian
            vertices=self.vertices,  # Use the vertices defined above
            )
        #TODO: add a fallback for unsupported coordinate systems
        
        # Option 2) use FESTIM's Mesh - and FeniCS (Dolfin ?) objects - specific for spherical geometry
        # self.model.mesh = F.Mesh(
        #     type="spherical",  # Specify spherical mesh type
        # )

        return self.model.mesh

    def _specify_boundary_conditions(self, config):
        """
        Specify boundary conditions for the FESTIM model.
        This method can be extended to include specific boundary condition logic.
        """
        print("Specifying boundary conditions...")

        # Sperical case, apply DirichletBC at boundary (in relative terms, r=1.0), NeumannBC (FluxBC) at the center (r=0.0)
        self.model.boundary_conditions = [
            F.FluxBC(
                surfaces=[1],  # Assuming a single surface at the end of the mesh
                value=float(config['boundary_conditions']['left_bc_concentration_value']),  # boundary value
                field=0,
            ),
            F.DirichletBC(
                surfaces=[2],  # Assuming a single surface at the start of the mesh
                value=float(config['boundary_conditions']['right_bc_concentration_value']),  # boundary value
                field=0,
            ),
        ]

        #print(f"Using boundary value at outer surfaces: {self.model.boundary_conditions[1].__dict__}") ###DEBUG
        #print(f"Using constant volumetric source term with values: {self.model.sources[0].__dict__}") ###DEBUG

        return self.model.boundary_conditions

    def _specify_materials(self, config):
        """
        Specify materials for the FESTIM model.
        This method can be extended to include specific material logic.
        """
        print("Specifying materials...")

        # Define material properties
        self.model.materials = F.Material(
            id=1,
            D_0=float(config['materials']['D_0']),  # diffusion coefficient
            E_D=float(config['materials']['E_D']),  # activation energy
            thermal_cond=float(config['materials']['thermal_conductivity']),  # thermal conductivity
            rho=float(config['materials']['rho']),  # density
            heat_capacity=float(config['materials']['heat_capacity']),  # specific heat capacity
            #solubility=float(config['materials']['solubility']),  # solubility
            )
        # TODO: fetch data from HTM DataBase - LiO2 as an example (absent in HTM DB)

        #print(f"Using material properties: D_0={self.model.materials[0].D_0}, E_D={self.model.materials[0].E_D}, T={self.model.T.__dict__}") ###DEBUG

        return self.model.materials

    def _add_source_terms(self, config):
        """
        Add source terms to the FESTIM model.
        This method can be extended to include specific source term logic.
        """
        print("Adding source terms...")

        #print(f"Passing the source term value: {config['source_terms']['source_value']}") ###DEBUG
        
        # Set a constant volumetric source term
        self.model.sources = [
            F.Source(
                value=float(config['source_terms']['source_concentration_value']),  # source term value [m^-3 s^-1]
                volume=1,
                field=0,
            ),
        ]

        return self.model.sources

    def _add_heat_conduction(self, config):
        """
        Add heat conduction to the FESTIM model.
        This method can be extended to include specific heat conduction logic.
        """
        print("Adding heat conduction model...")

        # Add a new quantity of interest for temperature to analyse after the simulation
        self.quantities_of_interest["temperature"] = None

        # Example: Set a constant heat conduction coefficient

        # Add model for T, temperature quantity, and redefine the model's attribute
        if self.model.settings.transient:
            self.model.T = F.HeatTransferProblem(
                transient=True,
                initial_condition=F.InitialCondition(
                    value=float(config['model_parameters']['T_0']),  # Initial temperature [K]
                    field="T",
                ),
            )
            print(f" >> Using transient heat problem with the initial temperature: {self.model.T.initial_condition.value} [K]")  # Debugging output
        else:
            self.model.T = F.HeatTransferProblem(
                transient=False,
            )
            print(f" >> Using steady-state heat problem")  # Debugging output

        # Define heat transfer coefficient and external temperature

        h_coeff = float(config['boundary_conditions']['right_bc_hcoeff_value'])  # Convective heat transfer coefficient [W/(m²*K)]

        T_ext = float(config['boundary_conditions']['right_bc_T_value'])  # External temperature [K]
        # For now, use a constant external temperature

        if 'source_heat_type' in config['source_terms']:
            if config['source_terms']['source_heat_type'] == 'constant':
                Q_source = float(config['source_terms']['source_heat_value'])  # Heat source term [W/m³]
        else:
            Q_source = 0.0

        # Apply appropriate boundary conditions for heat transfer
        # At the centre (r=0), apply Dirichlet BC
        self.model.T.boundary_conditions.append(
            F.DirichletBC(
                value=config['boundary_conditions']['left_bc_T_value'],  # left boundary temperature [K]
                surfaces=[1],
                field="T",
            )
        )
        # At the outer surface (r=1), apply Convective Flux BC
        self.model.T.boundary_conditions.append(
            F.ConvectiveFlux(
                h_coeff=h_coeff,
                T_ext=T_ext,  # External temperature [K]
                surfaces=[2], 
                #field="T",
            )
        )
        print(f" >> Using boundary conditions for temperature: T={self.model.T.boundary_conditions[0].value} [K] at surface 1, h_coeff={h_coeff}, T_ext={T_ext} [K] at surface 2")  # Debugging output

        # Apply appropriate source terms for heat transfer
        self.model.T.sources.append(
            F.Source(
                value=Q_source,  # Heat source term [W/m³]
                volume=1,  # Assuming a single volume for the entire mesh
                field="T",
            )
        )   
        print(f" >> Using source term for heat transfer: Q_source={Q_source} [W/m³]")  # Debugging output
        
        return self.model.T

    def _specify_time_integration_settings(self, config):
        """
        Specify time integration settings for the FESTIM model.
        This method can be extended to include specific time integration logic.
        """
        print("Specifying time integration settings...")

        # Define time integration settings for the simulation
        self.model.dt = F.Stepsize(
            #float(config['simulation']['time_step']), # op1) fixed time step size

            initial_value=float(config['simulation']['time_step']),  # op2) initial time step size for adaptive dt
            stepsize_change_ratio=float(config['simulation']['stepsize_change_ratio']),  # ratio for adaptive time stepping
            #min_value=float(config['simulation']['min_time_step']),  # minimum time step size
            max_stepsize=float(config['simulation']['max_stepsize']),  # maximum time step size
            dt_min=1e-05,  # minimum time step size for adaptive time stepping

            milestones=self.milestone_times, # check points for results export
        ) 

        return self.model.dt

    def _specify_outputs(self, config):
        """
        Specify outputs for the FESTIM model.
        This method can be extended to include specific output logic.
        """
        print("Specifying outputs...")

        # Define outputs for the simulation - folder for saving results
        self.result_folder = config['simulation']['output_directory']

        # Specify milestone times for results export
        self.milestone_times = config['simulation']['milestone_times']  # List of times to export results

        # Define output formats
        if self.model.settings.transient:
            # For transient simulations, export results at specified milestone times
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
                    filename=f"{self.result_folder}/results_tritium_concentration.txt",
                    times=self.milestone_times,
                )
            ]
        else:
            # For steady-state simulations, export results at the end of the simulation
            self.model.exports = [
                F.TXTExport(
                    field="solute",
                    filename=f"{self.result_folder}/results_tritium_concentration.txt",
                )
            ]

        # Add temperature export if heat transfer model is used
        if 'heat_model' in config['model_parameters'] and config['model_parameters']['heat_model'] == 'heat_transfer':
            if self.model.settings.transient:
                self.model.exports.append(
                    F.TXTExport(
                        field="T",
                        filename=f"{self.result_folder}/results_temperature.txt",
                        times=self.milestone_times,
                    )
                )
            else:
                self.model.exports.append(
                    F.TXTExport(
                        field="T",
                        filename=f"{self.result_folder}/results_temperature.txt",
                    )
                )

        return self.model.exports

    def _add_derived_quantities(self, list_of_derived_quantities_names=['tritium_inventory']):
        """
        Add derived quantities to the FESTIM model.
        This method can be extended to include specific derived quantity logic.
        """
        print("Adding derived quantities...")

        # Compute total tritium inventory
        if 'tritium_inventory' in list_of_derived_quantities_names:
            # This will compute the total tritium inventory and add it to the model's exports
            derived_quantities = Diagnostics.compute_total_tritium_inventory(self.result_folder)
        else:
            derived_quantities = None

        # Add derived quantities to the model's exports
        self.model.exports.append(derived_quantities)

        # Add derived quantities to the model's quantities of interest
        for q_name in list_of_derived_quantities_names:
            if q_name not in self.quantities_of_interest:
                self.quantities_of_interest[q_name] = None

        return derived_quantities

    def inspect_model_structure(self):
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

    def run(self):
        """
        Run the FESTIM simulation.
        This method initializes the model, runs the simulation, and stores the results.
        """
        # Initialize the model
        self.model.initialise()

        # Input and initalisation verification
        # self.inspect_model_structure()  ### Print the structure of the model for debugging

        # Run the simulation
        self.results = self.model.run()

        print("FESTIM simulation completed successfully!")

        # Output verification
        #print(f"Results: {self.results}") ### DEBUG
        self.result_flag = True

        # Export results
        # self.model.export_results()

        #TODO: Think of better BCs
        #TODO: Read Lithium (and LiTO) data from HTM DataBase - absent in HTM DB
        #TODO: Use proprietary visualisation and diagnostics tools
        #TODO: Explore cartesian/cylindrical/spherical geometries/coordinates/curvatures (+, diifrence to be recorded)
        #TODO: Add important physical effects
        #TODO: Couple with heat conductivity and temperature (+, in testing)

        # n_elem_print = 3
        # print(f">>> Model.run: Printing last {n_elem_print} elements of the results for last time of {self.milestone_times[-1]}: {self.results[-n_elem_print:, -1]}")  # Print last n elements of the results for the last time step ###DEBUG

        return self.results
