#!/home/yehor/miniconda3/envs/festim-env/bin/python3
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import numpy as np

# Import FESTIM library
import festim as F

# Local imports
from .diagnostics import Diagnostics

# Import DOLFINX
import dolfinx

##### Class definitions #####

class ProfileExport(F.VolumeQuantity):

    def compute(self):
        profile = self.field.solution.x.array[:].copy()

        self.data.append(profile)


class BaseModel:
    """
    Base class for all models.
    """
    def __init__(self, ):

        pass

    def _specify_geometry(self, config):
        """
        Specify the geometry of the model.
        """
        raise NotImplementedError("Geometry specification method not implemented in the base class.")

    def _specify_materials(self, config):
        """
        Specify the materials used in the model.
        """
        raise NotImplementedError("Materials specification method not implemented in the base class.")
    
    def _specify_boundary_conditions(self, config):
        """
        Specify the boundary conditions for the model.
        """
        raise NotImplementedError("Boundary conditions specification method not implemented in the base class.")
    
    def _add_source_terms(self, config):
        """
        Add source terms to the model.
        """
        raise NotImplementedError("Source terms method not implemented in the base class.")

    def _add_heat_conduction(self, config):
        """
        Add heat conduction to the model.
        """
        raise NotImplementedError("Heat conduction method not implemented in the base class.")

    def _specify_time_integration_settings(self, config):
        """
        Specify the time integration settings for the model.
        """
        raise NotImplementedError("Time integration settings method not implemented in the base class.")

    def _add_derived_quantities(self, config):
        """
        Add derived quantities to the model.
        """
        raise NotImplementedError("Derived quantities method not implemented in the base class.")
    
    def run(self):
        """
        Run the model simulation.
        This method should be overridden in derived classes.
        """
        raise NotImplementedError("The run method must be implemented in derived classes.")


class Model_legacy(BaseModel):
    """
    Model class for the FESTIM simulation.
    Old version: using version 1.4 of FESTIM API
    """

    def __init__(self, config=None):

        #super().__init__(**kwargs)

        self.name = "FESTIM Model"
        self.description = "Model for FESTIM simulation"

        self.config = config if config else {}

        # Create a FESTIM model instance
        self.model = F.Simulation()

        self.results = None  # Placeholder for results
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
            self.model.transient = True

            self.model.settings = F.Settings(
                transient=True,  # Enable transient simulation
                final_time=float(config['model_parameters']['total_time']),  # final time of the simulation [s]
                absolute_tolerance=float(config['simulation']['absolute_tolerance']),  #  absolute tolerance
                relative_tolerance=float(config['simulation']['relative_tolerance']),  #  relative tolerance
            )

                # initial_condition=F.InitialCondition(
                #     value=float(config['initial_conditions']['concentration']['value']),  # Initial temperature [K]
                #     field=1,
                # ),
                    
            print("Model is set to transient simulation.")
        else:

            self.model.transient = False

            self.model.settings = F.Settings(
                transient=False,  # Enable transient simulation
                absolute_tolerance=float(config['simulation']['absolute_tolerance']),  #  absolute tolerance
                relative_tolerance=float(config['simulation']['relative_tolerance']),  #  relative tolerance
                maximum_iterations=60,  # maximum number of iterations for steady-state simulation
            )

            print("Model is set to steady-state simulation.")

        # Define model parameters: temperature (and heat transfer model)
        if 'heat_model' in config['model_parameters'] and config['model_parameters']['heat_model'] == 'heat_transfer':
            # Use heat transfer model
            self._add_heat_conduction(config)
            #TODO test thoroughly!
        else:
            self.model.T = config['model_parameters']['T_0'] # set fixed background temperature

        # print (f" >> Using heat transfer model: {self.model.T.__dict__}") ###DEBUG

        # Define Boundary conditions
        self._specify_boundary_conditions(config)

        # Define source terms
        self._add_source_terms(config)

        # Define derived quantities - TODO implement
        # self.derived_quantities = self.define_derived_quantities()

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

        print(f" > config['model_parameters'] = \n {config['model_parameters']}")  ###DEBUG
        print(f" > config['simulation'] = \n {config['simulation']}")  ###DEBUG

        print(f" > Initialisation finished! Model initialized with {self.n_elements} elements")  ###DEBUG

    def _specify_geometry(self, config, refine_flag=False):
        """
        Specify the geometry of the FESTIM model.
        This method can be extended to include specific geometry logic.
        """
        print("Specifying geometry...")

        print(f" > config['geometry'] = \n {config['geometry']}")  ###DEBUG
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
        if refine_flag: 
            refined_length_fraction = 0.1  # Fraction of the length to refine
            refined_elements_fraction = 0.25  # Fraction of elements to refine
            refined_elements_count = int(self.n_elements * refined_elements_fraction)

            self.vertices = np.concatenate((
                np.linspace(0., self.length * (1. - refined_length_fraction), self.n_elements - refined_elements_count + 1), # inner (larger) part of the mesh
                np.linspace(self.length * (1. - refined_length_fraction), self.length, refined_elements_count + 1)[1:] # refined (smaller outer) part of the mesh
            ))
            #TODO mind round-off errors in the mesh size

        #print(f"Using vertices: {self.vertices}")  ###DEBUG output

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

        #print(f" >> Using mesh object: {self.model.mesh.__dict__}")  ###DEBUG
        return self.model.mesh

    def _specify_boundary_conditions(self, config, quantity='concentration'):
        """
        Specify boundary conditions for the FESTIM model.
        This method can be extended to include specific boundary condition logic.
        """
        print("Specifying boundary conditions...")

        print(f" > config['boundary_conditions'] = \n {config['boundary_conditions']}")  ###DEBUG

        if self.model.boundary_conditions is None:
            self.model.boundary_conditions = []

        # Spherical case, by default: apply DirichletBC at boundary (in relative terms, r=1.0), NeumannBC (FluxBC) at the center (r=0.0)
        
        # Iterate over BCs, quantites are on the top level
        for bc_quantity, bc_specs in config['boundary_conditions'].items():

            # Map boundary condition quantities to FESTIM fields
            quantity_map = {'concentration': 0, 'temperature': "T"}

            # Determine the field based on the boundary condition quantity
            field = quantity_map.get(bc_quantity, None)
            if field is None:
                raise ValueError(f"Unknown or unsupported boundary condition quantity: {bc_quantity}")
   
            # Map locations to surfaces in spherical coordinates: left (r=0.0) and right (r=1.0)
            surface_map = {'left': 1, 'right': 2}

            # Iterate over BCs specifications, locations are on the second level
            # ATTENTION: apply a simple filter e.g. only add BCs if the quantity matches the one specified
            if bc_quantity == quantity:

                for bc_location, bc_vals in bc_specs.items():
                    # ATTENTION: so far, only 1D is supported

                    # Map locations to surfaces in spherical coordinates: left (r=0.0) and right (r=1.0)
                    surface = surface_map.get(bc_location, None)
                    if surface is None:
                        raise ValueError(f"Unknown or unsupported boundary condition location: {bc_location}")

                    if bc_vals['type'] == 'dirichlet':
                        # Dirichlet boundary condition
                        self.model.boundary_conditions.append(
                            F.DirichletBC(
                                value=float(bc_vals['value']),  # Boundary condition value
                                surfaces=[surface],  # Apply to the specified surface
                                field=field,  # Field for the boundary condition
                            )
                        )
                        print(f" >> Using Dirichlet BC at surface {surface} with value {bc_vals['value']} for field {field}")
                    elif bc_vals['type'] == 'neumann':
                        # Neumann boundary condition (Flux)
                        self.model.boundary_conditions.append(
                            F.FluxBC(
                                value=float(bc_vals['value']),  # Boundary condition value
                                surfaces=[surface],  # Apply to the specified surface
                                field=field,  # Field for the boundary condition
                            )
                        )
                        print(f" >> Using Neumann BC at surface {surface} with value {bc_vals['value']} for field {field}")
                    elif bc_vals['type'] == 'convective_flux':
                        # Convective flux boundary condition
                        self.model.boundary_conditions.append(
                            F.ConvectiveFlux(
                                h_coeff=float(bc_vals['hcoeff_value']),  # Convective heat transfer coefficient
                                T_ext=float(bc_vals['Text_value']),  # External temperature
                                surfaces=[surface],  # Apply to the specified surface
                                field=field,  # Field for the boundary condition
                            )
                        )
                        print(f" >> Using Convective Flux BC at surface {surface} with h_coeff {bc_vals['hcoeff_value']} and T_ext {bc_vals['Text_value']} for field {field}")
                    else:
                        raise ValueError(f"Unknown or unsupported boundary condition type: {bc_vals['type']}")

        #print(f"Using boundary value at outer surfaces: {self.model.boundary_conditions[1].__dict__}") ###DEBUG
        #print(f"Using constant volumetric source term with values: {self.model.sources[0].__dict__}") ###DEBUG

        print(f" >> Using boundary conditions: {self.model.boundary_conditions}")  ###DEBUG
        return self.model.boundary_conditions

    def _specify_materials(self, config):
        """
        Specify materials for the FESTIM model.
        This method can be extended to include specific material logic.
        """
        print("Specifying materials...")

        print(f" > config['materials'] = \n {config['materials']}")  ###DEBUG

        # Define material properties
        material = F.Material(
            id=1,
            
            D_0=float(config['materials']['D_0']),  # diffusion coefficient
            E_D=float(config['materials']['E_D']),  # activation energy
            thermal_cond=float(config['materials']['thermal_conductivity']),  # thermal conductivity
            rho=float(config['materials']['rho']),  # density
            heat_capacity=float(config['materials']['heat_capacity']),  # specific heat capacity
            
            #solubility=float(config['materials']['solubility']),  # solubility
            )
        # TODO: fetch data from HTM DataBase - LiO2 as an example (absent in HTM DB)

        self.model.materials = material

        #print(f"Using material properties: D_0={self.model.materials[0].D_0}, E_D={self.model.materials[0].E_D}, T={self.model.T.__dict__}") ###DEBUG

        print(f" >> Using material properties: {self.model.materials.__dict__}")  ###DEBUG
        return self.model.materials

    def _add_source_terms(self, config, quantity='concentration'):
        """
        Add source terms to the FESTIM model.
        This method can be extended to include specific source term logic.
        """
        print("Adding source terms...")

        if self.model.sources is None:
            self.model.sources = []

        print(f" > config['source_terms'] = \n {config['source_terms']}")  ###DEBUG

        # Iterate over source terms in the configuration
        for source_type, source_specs in config['source_terms'].items():
            if source_type == 'concentration':
                field = 0
            elif source_type == 'heat':
                field = "T"
            else:
                raise ValueError(f"Unknown or unsupported source term type: {source_type}") 
            
            # Check if the source term matches the specified quantity
            if source_type == quantity:
                if source_specs['type'] == 'constant':
                    # Constant source term
                    self.model.sources.append(
                        F.Source(
                            value=float(source_specs['value']),  # Source term value
                            volume=1,  # Assuming a single volume for the entire mesh
                            field=field,  # Field for the source term
                        )
                    )
                    print(f" >> Using constant source term with value {source_specs['value']} for field {field}") ###DEBUG
                else:
                    raise ValueError(f"Unknown or unsupported source term type: {source_specs['type']}")

        print(f" >> Using source terms: {self.model.sources}")  ###DEBUG
        return self.model.sources

    def _add_heat_conduction(self, config):
        """
        Add heat conduction to the FESTIM model.
        This method can be extended to include specific heat conduction logic.
        """
        print("Adding heat conduction model...")

        #print(f" > config[''] = \n {config['']}")  ###DEBUG

        # Add a new quantity of interest for temperature to analyse after the simulation
        self.quantities_of_interest["temperature"] = None

        # Example: Set a constant heat conduction coefficient

        # Add model for T, temperature quantity, and redefine the model's attribute
        if self.model.settings.transient:

            self.model.T = F.HeatTransferProblem(
                transient=True,
                initial_condition=F.InitialCondition(
                    value=float(config['initial_conditions']['temperature']['value']),  # Initial temperature [K]
                    field="T",
                ),
                # absolute_tolerance=float(config['simulation']['absolute_tolerance']),  #  absolute tolerance
                # relative_tolerance=float(config['simulation']['relative_tolerance']),  #  relative tolerance     
            )

            #self.model.T = F.HeatTransferProblem(transient=True, initial_condition=F.InitialCondition(field="T", value=300)) ###DEBUG

            print(f" >> Using transient heat problem with the initial temperature: {self.model.T.initial_condition.value} [K]")  ###DEBUG output
        else:
            self.model.T = F.HeatTransferProblem(
                transient=False,
                maximum_iterations=60,  # maximum number of iterations for steady-state simulation
            )
            print(f" >> Using steady-state heat problem")  # Debugging output

        # If exists, apply a heat source term
        self.model.sources = []  # Initialize source terms for heat transfer

        if 'heat' in config['source_terms']:
            if config['source_terms']['heat']['type'] == 'constant':
                Q_source = float(config['source_terms']['heat']['value'])  # Heat source term [W/m³]

                self.model.sources.append(
                    F.Source(
                        value=Q_source,  # Source term value
                        volume=1,  # Assuming a single volume for the entire mesh
                        field="T", # applying to temperature field
                    )
                )

                #self.model.sources = [F.Source(value=100, field="T", volume=1)] ###DEBUG

        else:
            print(f"Warning: No heat source term specified, using Q_source = 0.0 W/m³")
            Q_source = 0.0

        print(f" >> Using source term for heat transfer: Q_source={Q_source} [W/m³]")  ###DEBUG output

        # Apply appropriate boundary conditions for heat transfer
        surfaces_nums = [1, 2]
        surface_names = ['left', 'right']
        surface_map = {'left':1, 'right':2}

        # Check if temperature boundary conditions are specified - apply fallback if not
        if 'temperature' not in config['boundary_conditions']:
            print("Warning: No temperature boundary conditions specified, using default values.")
            config['boundary_conditions']['temperature'] = {
                'left': {'type': 'dirichlet', 'value': 300.0},  # Default left boundary temperature [K]
                'right': {'type': 'neumann', 'value': 0.0}  # Default right boundary temperature gradient [K]
            }

        # Iterate over surfaces and apply boundary conditions
        for surface_name, surface_num in surface_map.items():
            print(f" >>> Applying HEAT boundary conditions for surface {surface_name} (surface number {surface_num})")  ###DEBUG

            # Check if the surface has a temperature boundary condition specified
            if surface_name in config['boundary_conditions']['temperature']:
                print(f" >>> Using boundary condition for surface {surface_name}: {config['boundary_conditions']['temperature'][surface_name]}")  ###DEBUG

                # Get the boundary condition for the surface
                bc = config['boundary_conditions']['temperature'][surface_name]

                # Apply the boundary condition based on its type
                # - 1) Dirichlet BC: fixed temperature
                if bc['type'] == 'dirichlet':
                    # Dirichlet boundary condition
                    self.model.boundary_conditions.append(
                        F.DirichletBC(
                            value=float(bc['value']),  # Boundary condition value
                            surfaces=[surface_num],  # Apply to the specified surface
                            field="T",  # Field for the boundary condition
                        )
                    )
                    print(f" >>> Using Dirichlet BC at surface {surface_num} with value {bc['value']} [K]")  # Debugging output

                # - 2) Neumann BC: fixed flux
                elif bc['type'] == 'neumann':
                    # Neumann boundary condition (Flux)
                    self.model.boundary_conditions.append(
                        F.FluxBC(
                            value=float(bc['value']),  # Boundary condition value
                            surfaces=[surface_num],  # Apply to the specified surface
                            field="T",  # Field for the boundary condition
                        )
                    )
                    print(f" >>> Using Neumann BC at surface {surface_num} with value {bc['value']} [K m^-1]")  ###DEBUG output

                # - 3) Convective flux BC: convective heat transfer
                elif bc['type'] == 'convective_flux':
                    # Define heat transfer coefficient (if applicable)
                    h_coeff = float(config['boundary_conditions']['temperature'][surface_name]['hcoeff_value'])  # Convective heat transfer coefficient [W/(m²*K)]
                    if h_coeff is None:
                        raise ValueError("Convective heat transfer coefficient (h_coeff) not specified.")
                    # Convective flux boundary condition
                    self.model.boundary_conditions.append(
                        F.ConvectiveFlux(
                            h_coeff=h_coeff,  # Convective heat transfer coefficient [W/(m²*K)]
                            T_ext=float(bc['value']),  # External temperature [K]
                            surfaces=[surface_num],  # Apply to the specified surface
                            #field="T",  # Field for the boundary condition
                        )
                    )
                    print(f" >>> Using Convective Flux BC at surface {surface_num} with h_coeff {h_coeff} [W/(m²*K)] and T_ext {bc['value']} [K]")  ###DEBUG output
                else:
                    raise ValueError(f"Unknown or unsupported boundary condition type: {bc['type']}")
            else:
                print(f"Warning: No temperature boundary condition specified for surface {surface_name}, using default values.")

        #self.model.boundary_conditions = [F.DirichletBC(surfaces=[1, 2], value=400, field="T")] ###DEBUG

        #print(f" >> Using boundary conditions for temperature: T={self.model.T.boundary_conditions[0].value} [K] at surface 1, h_coeff={h_coeff}, T_ext={config['boundary_conditions']['temperature']['right']['value']} [K] at surface 2")  # Debugging output
        
        print(f" >> Using heat transfer model: {self.model.T.__dict__}")  ###DEBUG
        return self.model.T

    def _specify_time_integration_settings(self, config):
        """
        Specify time integration settings for the FESTIM model.
        This method can be extended to include specific time integration logic.
        """
        print("Specifying time integration settings...")

        # Define time integration settings for the simulation
        dt = float(config['simulation']['time_step'])

        self.model.dt = F.Stepsize(
            # dt, # op1) fixed time step size

            initial_value=dt,  # op2) initial time step size for adaptive dt
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


class Model(BaseModel):
    """
    Model class for the FESTIM simulation.
    Uses 2.0 version of FESTIM library.
    """
    def __init__(self, config=None):

        """
        Initialize the model with the given parameters.

        Parameters:
            - config: a nested dictionary generated from the YAML config file
        """
        self.name = "FESTIM Model"
        self.lib_version = "2.0"
        print(f" ! Initialising {self.name} !..")

        #super().__init__()

        #TODO: add derived quantities, outputs, postprocessing

        # Assign the config from initialisation parameter
        self.config = config if config else {}

        # Save simulation results here later
        self.results = {}

        # Put dictionary of quantities of interest to be computed and stored after the simulation
        if "quantities_of_interest" in config.get("simulation", {}):
            self.quantities_of_interest = {k: None for k in config["simulation"]["quantities_of_interest"]}
        else:
            self.quantities_of_interest = {'tritium_concentration': None}

        # Map quantity name to their id-s or FESTIM shorthands
        self.quantity_map = {'concentration': 0, 'temperature': "T"}  

        # Get common model parameters
        model_parameters_problems_config = config.get("model_parameters", {}).get("problems", {})

        # Specify dictionary of problems: keys are names, values are FESTIM 2.0 problems instances
        if "problems" in config.get("model_parameters", {}):
            self.problems = {k: {} for k,_ in model_parameters_problems_config.items()}
        else:
            self.problems = {'tritium_transport': {},}
        print(f" > Model problems: {self.problems}")  ###DEBUG

        # Specify materials used
        self._specify_materials(config)

        # Initialise geometry common for all problems
        # TODO that should be split into common and problem specific one
        self._specify_geometry(config)

        # Type of the models
        self.transient = bool(config.get("model_parameters", {}).get("transient", False))

        # Specify whether the model has non-stationary term or not
        if self.transient:
            self.total_time = float(config.get("model_parameters", {}).get("total_time", 1.0))

        # Read tolerances config input
        tolerance_config = config.get("simulation", {}).get("tolerances", {})

        # Initialise each problem separately, add all necessary components
        for problem_name, problem_instance in self.problems.items():
            print(f"Initialising problem for: {problem_name}")

            # Read the config dictionary for the particular problem: tritium transport
            config_transport_problem = model_parameters_problems_config.get(problem_name, {})

            # Add problem-specific initialisation here
            if problem_name == "tritium_transport":

                # Set up name(s) of the relevant quantities of interest
                problem_instance['qoi_name'] = 'tritium_concentration'
                # TODO this could be a list of QoI names

                qoi_name_local = 'concentration'
                qoi_name_condition_local = 'concentration'

                # Create a FESTIM problem instance
                problem_instance['festim_problem'] = F.HydrogenTransportProblem()

                # Set species in question
                species_names_config = config_transport_problem.get("species", [])

                self.species_descriptor = {
                    "Tritium": {"festim_name": "T"}
                }

                if not hasattr(self, 'species') or self.species is None:
                    self.species = {k:None for k in species_names_config} # TODO can be merged with the following code block ...
                
                for species_k, species_v in self.species.items():
                    self.species[species_k] = F.Species(self.species_descriptor[species_k]["festim_name"])
                # TODO make automatic parsing of names; create species naming dictionary or make them the same

                # Specify species list for FESTIM Model
                problem_instance['festim_problem'].species = [v for _,v in self.species.items()]

            elif problem_name == "heat_transport":

                # Set up name(s) of the relevant quantities of interest
                problem_instance['qoi_name'] = 'temperature'

                qoi_name_local = 'heat'
                qoi_name_condition_local = 'temperature'

                # Create a FESTIM problem instance
                problem_instance['festim_problem'] = F.HeatTransferProblem()

                # Specify Initial Conditions
                self._specify_initial_conditions(config, quantity_filter=qoi_name_condition_local)

            # Read tolerances from the config input
            absolute_tolerance = float(tolerance_config.get("absolute_tolerance", 1.e0).get(problem_name, 1.e0))
            relative_tolerance = float(tolerance_config.get("relative_tolerance", 1.e-10))

            # Specify settings, including transient/steady
            if self.transient:
                # Set settings for transient problem, including time stepping

                problem_instance['festim_problem'].settings = F.Settings(
                    transient=self.transient,
                    final_time=self.total_time, # difference here (and time step size) for transient problems #TODO could be done by in-line comprehension, e.g. ternary operator
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )
            else:
                problem_instance['festim_problem'].settings = F.Settings(
                    transient=self.transient,
                    atol=absolute_tolerance,
                    rtol=relative_tolerance,
                )

            # Specify geometry and mesh # must be called after _specify_geometry()
            problem_instance['festim_problem'].subdomains = self.subdomains
            problem_instance['festim_problem'].mesh = self.meshes[int(config_transport_problem.get("domains", [{}])[0].get("id", 1))]
            
            # Specify Boundary Conditions
            problem_instance['festim_problem'].boundary_conditions = self._specify_boundary_conditions(config, quantity_filter=qoi_name_condition_local)
            print(f" >> Specified boundary conditions for {qoi_name_condition_local}: {problem_instance['festim_problem'].boundary_conditions}") ###DEBUG

            # Add Source terms
            problem_instance['festim_problem'].sources = self._add_source_terms(config, quantity_filter=qoi_name_local)

            # Set back the specified problem instance  - should work in situ in Python3
            #self.problems[problem_name] = problem_instance

            print(f" > Finished problem initialisation for {problem_name}")
            print(f" >> State of the self.problems after {problem_name} initialisation: {self.problems}") ###DEBUG
            # TODO: the two problems can be specified without if-statement with a map of name and FESTIM object class

        # ###DEBUG BLOCK
        # self.problems['heat_transport']['festim_problem'].boundary_conditions = [
        #     F.FixedTemperatureBC(subdomain=self.domain_surfaces[1], value=600.0),
        #     F.FixedTemperatureBC(subdomain=self.domain_surfaces[2], value=550.0),
        # ] ###DEBUG
        # print(f" >>>! Manually re-specified boundary conditions for {qoi_name_condition_local}: {problem_instance['festim_problem'].boundary_conditions}") ###DEBUG
        # ###

        # Check if a pair of problems is present
        if 'tritium_transport' in model_parameters_problems_config and 'heat_transport' in model_parameters_problems_config:
            # Add coupling model for tritium transport and heat transport
            print(f" > Coupling tritium transport and heat transport problems...")

            #self.problems['tritium_heat_coupling'] = {}
            #self.problems['tritium_heat_coupling']['qoi_name'] = None

            if self.transient:
                # Create a FESTIM coupled model
                self.model = F.CoupledTransientHeatTransferHydrogenTransport(
                    heat_problem=self.problems['heat_transport']['festim_problem'],
                    hydrogen_problem=self.problems['tritium_transport']['festim_problem'],
                )

                # Define absolute tolerance: minimal of abs. tol. across problems
                absolute_tolerance = min([tol for _, tol in tolerance_config.get("absolute_tolerance", {}).items()])
                relative_tolerance = tolerance_config.get("relative_tolerance", 1.e-10)

                # Set settings and time stepping for the transient model
                self.model.settings = F.Settings(
                    transient=True,
                    final_time=self.total_time,
                    atol=absolute_tolerance,  #ATTENTION: FESTIM2.0 does not use this parameter for coupled problems
                    rtol=relative_tolerance,
                )

                # Set the time stepping for all transient models
                self._specify_time_integration_settings(config)
            else:
                # Set the steady-state problem
                #TODO: look up FESTIM2.0 class for coupled steady problem

                self.model.settings = F.Settings(
                    transient=False,
                    #atol=float(config.get("simulation", {}).get("absolute_tolerance", 1.e10).get(problem_name, 1.e0)),
                    rtol=float(config.get("simulation", {}).get("relative_tolerance", 1.e-10)),
                )

                raise NotImplementedError("Steady-state coupled model is not implemented yet in FESTIM 2.0")
        
        else:
            print(f"No models to couple found!")

            model_to_solve = 'tritium_transport' #ATTENTION: workaround for DEBUG

            print(f" Setting single (first, {model_to_solve}) problem as the model to solve")

            # Setting time stepping before tritium transport problem is set to be main model to be solved
            if self.transient:
                self._specify_time_integration_settings(config)

            # Set constant background temperature
            if 'tritium_transport' in self.problems:

                # Read the IC for the problem for tempearature
                initial_conditions_config = config.get("initial_conditions", {}).get("temperature", {})

                # Set the IC as background temperature: here - constant T
                self.problems['tritium_transport']['festim_problem'].temperature = self._get_config_entry(initial_conditions_config, "value", float)

            self.model = self.problems[model_to_solve]['festim_problem']
            # TODO make Python refer it by reference 

        print(f" >> Initialised problems: \n {self.problems}")  ###DEBUG

        # Specify outputs
        self._specify_outputs(config)

        # Specify logging level

        if 'log_level' in config.get("simulation", {}):
            log_level = config.get("simulation", {}).get("log_level", "info")
            if log_level == "debug":
                dolfinx.log.set_log_level(dolfinx.log.LogLevel.DEBUG)
            elif log_level == "info":
                dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
            elif log_level == "warning":
                dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
            elif log_level == "error":
                dolfinx.log.set_log_level(dolfinx.log.LogLevel.ERROR)
            else:
                print(f" >>>! Warning: Unknown log level '{log_level}', skipping.")

        print(f"Model {self.name} initialized with {len(self.problems)} problems and {len(self.materials)} materials. Number of mesh elements: {self.n_elements}.")

        print(f" >>> Material parameters used: \n{self.problems['tritium_transport']['festim_problem'].volume_subdomains[0].material.__dict__}") ###DEBUG

        print(f" > Initialisation finished !")

    def _get_config_entry(self, config_node, entry_name, entry_type: (float| int| str| list) = float):
        """
        Get a configuration entry from the config node (sub-dictionary generated from a YAML config file)
        """
        entry = config_node.get(entry_name, {})

        # Check if the entry is empty
        if entry_name not in config_node:
            print(f" >>>! Warning: Configuration entry '{entry_name}' of type {entry_type} not found in {config_node} !")
            return {}

        # If the entry is of the expected type, return it
        if isinstance(entry, entry_type):
            return entry

        # If the entry is a dictionary, return the mean value
        if isinstance(entry, dict):
            entry = entry.get("mean", None)  # Return the mean value if it's a dictionary

            # Check if the mean value is None
            if entry is None:
                print(f" >>>! Warning: Configuration entry '{entry_name}' is None !")
                return {}

            # Convert the entry to the expected type
            entry = entry_type(entry)  # Convert to the expected type

            return entry

        print(f" >>> Returning empty dictionary for {config_node}.{entry_name} = {entry}")
        return {}

    def _specify_geometry(self, config):
        """
        Specify geometry for the model.
        That includes: type of coordinate system, dimensions, and mesh details.
        Has to be called after _specify_materials
        Parameters:
            - config: a nested dictionary generated from the YAML config file
        TODO: this should be a method that can be called for a list of various subdomains - split into common geometry specification and per-species one
        """
        print("Specifying geometry...")
        
        # super()._specify_geometry(config)

        print(f" config[geometry]: {config.get('geometry', {})}") ###DEBUG

        # Specifying number of physical dimensions of the model
        self.n_dimensions = int(config.get("geometry", {}).get("dimensionality", 1))

        # Specifying type of coordinate system: cartesian, cylindrical (polar), spherical
        self.coordinate_system_type = str(config.get("geometry", {}).get("coordinate_system", "cartesian"))

        # Specifying type of mesh
        self.mesh_type = str(config.get("simulation", {}).get("meshes", [{}])[0].get("mesh_type", "regular"))

        # Specifying size of the mesh
        self.n_elements = int(config.get("simulation", {}).get("meshes", [{}])[0].get("n_elements", 128))  # Default to 128 elements if not specified

        # Create datastructures for some of the main geometry elements
        self.subdomains = []
        self.meshes = {}

        if self.n_dimensions == 1:
            # Create a 1D mesh

            self.domain_sizes = {}
            self.domain_volumes = {}
            self.domain_surfaces = {}

            # In 1D, the mesh is simply a line, volume is an interval, and surfaces are its ends
            self.max_surface_per_domain = 2

            config_domains = config.get("geometry", {}).get("domains", [{}])

            config_meshes = config.get("simulation", {}).get("meshes", [{}])

            # Iterate over domain configurations, each being an interval
            for config_domain in config_domains:

                id = int(config_domain.get("id", 1))  # Default to 1 if not specified

                # Get the corresponding mesh configuration for the domain
                config_mesh = next((m for m in config_meshes if int(m.get("domain_id", 1)) == id), {})

                print(f" > Specifying domain {id}") ###DEBUG
                print(f" >> config domain: \n{config_domain}")

                # Set the physical size (length in 1D) of the domain
                self.domain_sizes[id] = float(config_domain.get("length", 1.0))

                # Create a 1D volume subdomain (coordinate interval)
                self.domain_volumes[id] = F.VolumeSubdomain1D(
                        id=id,
                        borders=[0.0, self.domain_sizes[id]],
                        material=self.materials[int(config_domain.get("material", 1))]
                    )

                # Create 1D surface subdomain to the left (centre, boundary points)
                self.domain_surfaces[(id-1)*self.max_surface_per_domain+1] = F.SurfaceSubdomain1D(id=(id-1)*self.max_surface_per_domain+1, x=0.0)

                # Create 1D surface subdomain to the right (outer surface, boundary points)
                # Numbering of surfaces assumes max_surface_per_domain surfaces per domain, then offset by domain id
                self.domain_surfaces[(id-1)*self.max_surface_per_domain+2] = F.SurfaceSubdomain1D(id=(id-1)*self.max_surface_per_domain+2, x=self.domain_sizes[id])

                # Create a 1D mesh
                if self.mesh_type == "regular":
                    # Regular mesh with uniformly spaced elements
                    print(f" >> Creating regular mesh with {self.n_elements} elements for domain {id} of size {self.domain_sizes[id]}")  ###DEBUG
                    vertices = np.linspace(0., self.domain_sizes[id], self.n_elements + 1)
                elif self.mesh_type == "refined":
                    # Refined mesh with more elements near the surfaces (ends)
                    print(f" >> Creating refined mesh with {self.n_elements} elements for domain {id} of size {self.domain_sizes[id]}")  ###DEBUG
                    if config_mesh.get("refinement", "").get("rule", "") == "quadratic":
                        print(f" >>> Using quadratic refinement")  ###DEBUG
                        if config_mesh.get("refinement", "").get("location", "") == "right":
                            print(f" >>> Refining mesh towards the right end")  ###DEBUG
                            x = np.linspace(0., 1., self.n_elements + 1)
                            vertices = self.domain_sizes[id] * (1 - (1 - x)**2 / (1 - x[0])**2)  # Quadratic refinement
                        elif config_mesh.get("refinement", "").get("location", "") == "left":
                            print(f" >>> Refining mesh towards the left end")  ###DEBUG
                            x = np.linspace(0., 1., self.n_elements + 1)
                            vertices = self.domain_sizes[id] * (x**2) / (x[-1]**2)  # Quadratic refinement
                        else:
                            raise ValueError(f"Unknown refinement location: {config_mesh.get('locations', '')}")
                    else:
                        raise ValueError(f"Unknown refinement rule: {config_mesh.get('refinement', '').get('rule', '')}")
                else:
                    raise ValueError(f"Unknown mesh type: {self.mesh_type}")

                self.vertices = vertices
                self.meshes[id] = F.Mesh1D(vertices) #TODO: double check if this is accroding to FESTIM2.0
                # TODO: add spherical coordinates
                # TODO: add refined meshes

                # For 1D problem, map names of the surfaces to their id-s
                self.surface_map = {'left': {'festim_id': 1, 'loc_id': 1}, 'right': {'festim_id': 2, 'loc_id': 2}}

                print(f" >> Created 1D mesh with {self.n_elements} elements for domain {id} of size {self.domain_sizes[id]}")  ###DEBUG
           
            for surface_id in self.domain_surfaces:
                print(f" >> Domain surface {surface_id}: {self.domain_surfaces[surface_id].__dict__}")  ###DEBUG
            for volume_id in self.domain_volumes:
                print(f" >> Domain volume {volume_id}: {self.domain_volumes[volume_id].__dict__}")  ###DEBUG

        elif self.n_dimensions == 2:
            raise NotImplementedError("2D geometry is not implemented yet.")
        elif self.n_dimensions == 3:
            raise NotImplementedError("3D geometry is not implemented yet.")
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.n_dimensions}. Only 1D, 2D, and 3D are supported.")

        # Subdomains are all volumes and surfaces used in the problem
        self.subdomains = [*self.domain_volumes.values(), *self.domain_surfaces.values()]
        print(f" >> Specified subdomains :\n{self.subdomains}") ###DEBUG

        return self.subdomains, self.meshes

    def _specify_materials(self, config):
        """
        Specify materials for the model.
        Params:
            - config: a nested dictionary generated from the YAML config file
        TODO: it should be a list of materials, each could be assigned to a different subdomain
        """
        print("Specifying materials...")

        #super()._specify_materials(config)

        print(f" config[materials]: {config.get('materials', {})}") ###DEBUG

        # Set empty domain of materials
        if not hasattr(self, 'materials') or self.materials is None:
            self.materials = {}

        # Iterate over materials specified in the config
        for material_config in config.get("materials", []):
            # Create a FESTIM material object
            material = F.Material(
                #id=material_config.get("material_id", 1),
                name=str(material_config.get("material_name", "Unknown")),
                D_0=float(material_config.get("D_0", {}).get("mean", 0.0)), # Diffusion coefficient [m^2/s]
                E_D=float(material_config.get("E_D", {}).get("mean", 0.0)), # Activation energy [eV]
                thermal_conductivity=float(material_config.get("thermal_conductivity", {}).get("mean", 0.0)), # Thermal conductivity [W/(m*K)]
                density=float(material_config.get("rho", {}).get("mean", 0.0)), # Density [kg/m^3]
                heat_capacity=float(material_config.get("heat_capacity", {}).get("mean", 0.0)) # Heat capacity [J/(kg*K)]
            )
            self.materials[int(material_config.get("material_id", 1))] = material

        return self.materials

    def _specify_boundary_conditions(self, config, quantity_filter=None):
        """
        Specify boundary conditions for the model.
        Prepares a list of BCs that has to be assigned to a concrete problem later.
        Parameters:
            - config: a nested dictionary generated from the YAML config file
            - quantity_filter: the quantity (or type of problem) to which the boundary conditions apply (e.g., temperature, displacement). If None, apply to all those found in config
        TODO: this should be list of BCs per problem (x domain, x surface)
        """
        print("Specifying boundary conditions...")

        #super()._specify_boundary_conditions(config)

        print(f" config[boundary_conditions]: {config.get('boundary_conditions', {})}") ###DEBUG

        # Create an empty list for all the BCs
        if not hasattr(self, 'boundary_conditions') or self.boundary_conditions is None:
            self.boundary_conditions = []

        boundary_conditions = []  

        # Iterate over boundary conditions in the configuration
        for bc_quantity, bc_config in config.get("boundary_conditions", []).items():

            field = self.quantity_map.get(bc_quantity, "concentration")  # Default to concentration if not specified
            #TODO should it have a ValueError as a fallback? Also, does not seem to be needed in FESTIM 2.0

            # Check if the boundary condition applies to the specified quantity
            if bc_quantity == quantity_filter or quantity_filter is None:
                # Iterate over boundary condition locations (surfaces)
                for bc_location, bc_values in bc_config.items():

                    # Get the surface location ID for FESTIM model based on YAML specification
                    surface_loc_id = int(self.surface_map.get(bc_location, 1).get('loc_id', None))
                    #TODO should it have ValueError as a fallback? Alternatively, use bc_values['id']

                    value = self._get_config_entry(bc_values, 'value', float)  # Get the value of the boundary condition

                    # Specify BC for solute concentration
                    if bc_quantity == "concentration":

                        species = self.species.get(bc_values.get('species', 'Tritium'), None)

                        # print(f" >>> Species is a FESTIM class: {isinstance(species, F.Species)}") ###DEBUG

                        if bc_values['type'] == 'dirichlet':
                            # Dirichlet boundary condition

                            bc = F.FixedConcentrationBC(
                                species=species,
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=value,
                            )

                            print(f" >> Using {bc_values['type']} BC for {bc_quantity} at surface {surface_loc_id} with value {bc_values['value']} for field {field} and species {species}") ###DEBUG

                        elif bc_values['type'] == 'neumann':
                            
                            bc = F.ParticleFluxBC(
                                species=species,
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=value,
                            )
                        
                        elif bc_values['type'] == 'surface_reaction':

                            k_r0 = self._get_config_entry(bc_values, "k_r0", float)
                            E_kr = self._get_config_entry(bc_values, "E_kr", float)
                            k_d0 = self._get_config_entry(bc_values, "k_d0", float)
                            E_kd = self._get_config_entry(bc_values, "E_kd", float)
                            P_g = self._get_config_entry(bc_values, "P_g", float)

                            bc = F.SurfaceReactionBC(
                                reactant=[species],
                                subdomain=self.domain_surfaces[surface_loc_id],
                                gas_pressure=P_g,
                                k_r0=k_r0,
                                E_kr=E_kr,
                                k_d0=k_d0,
                                E_kd=E_kd,  
                            )

                        else:
                            raise ValueError(f"Unknown or unsupported boundary condition type: {bc_values['type']}")

                    # Specify BC for temperature
                    elif bc_quantity == "temperature":

                        if bc_values['type'] == 'dirichlet':
                            # Dirichlet BC for temperature
                            # print(f" >>> Adding {bc_values['type']} BC for {bc_quantity}") ###DEBUG

                            bc = F.FixedTemperatureBC(
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=value,
                            )

                            print(f" >> Using {bc_values['type']} BC for {bc_quantity} at surface {surface_loc_id} with value {bc_values['value']} for field {field}") ###DEBUG

                        elif bc_values['type'] == 'neumann':

                            bc = F.HeatFluxBC(
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=value,
                            )

                            print(f"Using heat flux boundary conditions at surface {surface_loc_id}") ###DEBUG

                        elif bc_values['type'] == 'convective_flux':

                            T_ext = self._get_config_entry(bc_values, "T_ext", float) # External temperature
                            h_conv = self._get_config_entry(bc_values, "h_conv", float)  # Convective heat transfer coefficient

                            flux_function = lambda T: h_conv * (T - T_ext)

                            bc = F.HeatFluxBC(
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=flux_function, 
                            )

                        elif bc_values['type'] == 'radiative_flux':

                            T_amb = self._get_config_entry(bc_values, "T_amb", float)  # External ambient temperature
                            epsilon = self._get_config_entry(bc_values, "emissivity", float) # Emissivity of ceramics

                            sigma = 5.671e-8 # Stefan-Boltzmann constant [W m^-2 K^-4]

                            flux_function = lambda T: epsilon * sigma * (T**4 - T_amb**4)

                            bc = F.HeatFluxBC(
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=flux_function, 
                            )

                        elif bc_values['type'] == 'combined_flux':

                            T_ext = self._get_config_entry(bc_values, "T_ext", float) # External temperature (He)
                            h_conv = self._get_config_entry(bc_values, "h_conv", float)  # Convective heat transfer coefficient
                            T_amb = self._get_config_entry(bc_values, "T_amb", float)  # External ambient temperature
                            epsilon = self._get_config_entry(bc_values, "emissivity", float) # Emissivity of ceramics

                            sigma = 5.671e-8 # Stefan-Boltzmann constant [W m^-2 K^-4]

                            flux_function = lambda T: h_conv * (T - T_ext) + epsilon * sigma * (T**4 - T_amb**4)
                            # TODO double check the direction convention, assuming here that '+' is outwards

                            bc = F.HeatFluxBC(
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=flux_function, 
                            )

                        else:
                            raise ValueError(f"Unknown or unsupported boundary condition type: {bc_values['type']}")

                    boundary_conditions.append(bc)

        print(f" >> Specified boundary conditions: {boundary_conditions}")  ###DEBUG
        
        self.boundary_conditions.extend(boundary_conditions)

        return boundary_conditions

    def _specify_initial_conditions(self, config, quantity_filter=None):
        """
        Specify initial conditions for the model.
        Has to be called after geometry and problems are specified

        Parameters:
            config:
            quantity_filter:
        """
        print("Specifying initial conditions...")

        # Create an initial condition
        if not hasattr(self, 'initial_conditions') or self.initial_conditions is None:
            self.initial_conditions = []

        for ic_name, ic_config in config.get("initial_conditions", {}).items():

            print(f" >> initial condition config for {ic_name}: \n{ic_config}") ###DEBUG

            if ic_name == quantity_filter or quantity_filter is None:
                # Add IC to the FESTIM 2.0 Model
                if ic_name == "temperature":
                    # Add IC for temperature
                    if "heat_transport" in self.problems:
                        if self.problems['heat_transport']['festim_problem'] is not None:
                            # For heat transport problem, set initial temperature
                            initial_temp_value = self._get_config_entry(ic_config, "value", float)
                            initial_temp_domain = self.domain_volumes.get(self._get_config_entry(ic_config, "domain_id", int), 1)

                            initial_condition = F.InitialTemperature(
                                value=initial_temp_value,
                                volume=initial_temp_domain,
                            )

                            print(f" >> Set IC for {ic_name} with value {initial_temp_value} at domain {initial_temp_domain}")  ###DEBUG

                            self.problems['heat_transport']['festim_problem'].initial_conditions = [initial_condition]

                            self.initial_conditions.append(initial_condition)

                        else:
                            raise ValueError("Heat transport problem does not have FESTIM model initialised.")
                    else:
                        raise ValueError("Heat transport problem is not defined.")
                else:
                    raise ValueError(f"Unknown or unsupported initial condition type: {ic_name}")

        return self.initial_conditions

    def _add_source_terms(self, config, quantity_filter=None):
        """
        Add source terms to the model.
        Prepare a list of source terms that has to be assigned to a specific problem later.
        Parameters:
            - config: a nested dictionary generated from the YAML config file
            - quantity_filter: the quantity (or type of problem) to which the source terms apply (e.g., concentration, heat). If None, apply to all those found in config
        TODO: this should be list of source terms per problem (x domain)
        """
        print(f"Adding source terms...")

        # super()._add_source_terms(config)

        print(f" config[source_terms]: {config.get('source_terms', {})}") ###DEBUG

        # Create an empty list for all the source terms
        if not hasattr(self, 'source_terms') or self.source_terms is None:
            self.source_terms = []

        # Create a local list for the current source terms (specific to a problem and quantity, to be assigned later)
        source_terms = []

        # Map volume domain names to their FESTIM and local id-s
        volume_map = {k: k for k,v in self.domain_volumes.items()} # No an identity mapping. This could be modified for a more complex mapping if needed
        print(" >>> volume_map:", volume_map) ###DEBUG
        print(" >>> self.domain_volumes:", self.domain_volumes) ###DEBUG

        # Iterate over source terms in the configuration
        for source_term_name, source_term_config in config.get("source_terms", {}).items():
            # Check if the source term applies to the specified quantity
            if source_term_name == quantity_filter or quantity_filter is None:
                print(f" >> Adding source term for: {source_term_name}") ###DEBUG

                source_term_value = self._get_config_entry(source_term_config, 'value', float)

                source_term_domain = self.domain_volumes.get(int(volume_map.get(source_term_config.get('domain_id', 1))), 1)

                print(f" >>> Source term value = {source_term_value} at domain {source_term_domain}") ###DEBUG

                # Apply a source term for solute concentration
                if source_term_name == "concentration":

                    # Create a FESTIM particle source
                    source_term = F.ParticleSource(
                        value=source_term_value,
                        volume=source_term_domain,
                        species=self.species.get(source_term_config.get('species', "Tritium")),
                    )

                    source_terms.append(source_term)

                # Apply a source term for heat
                elif source_term_name == "heat":

                    # Create a FESTIM heat source
                    source_term = F.HeatSource(
                        value=source_term_value,
                        volume=source_term_domain,
                    )

                    source_terms.append(source_term)

                else:
                    raise ValueError(f"Unknown or unsupported source term type: {source_term_name}")
                
        self.source_terms.extend(source_terms)

        print(f" >>> Local source terms are: \n{source_terms}") ###DEBUG

        return source_terms

    def _add_heat_conduction(self, config):
        """
        Add heat conduction to the model.
        ATTENTION: FESTIM 2.0 does not need this function.
        Parameters:
            - config: a nested dictionary generated from the YAML config file
        """

        super()._add_heat_conduction(config)
        
        print(f"FESTIM 2.0 treats heat conduction differently than FESTIM 1.0; now heat conduction is considered by default as one of the problems in the list!")

    def _specify_time_integration_settings(self, config):
        """
        Specify the time integration settings for the model.
        Parameters:
            - config: a nested dictionary generated from the YAML config file
        Has to be applies for transient problems
        Has to be applies after settings are specified
        TODO: has to be applied to separate problems
        """
        print(f"Specifying time integration settings...")

        #super()._specify_time_integration_settings(config)

        # Read the time step settings from the input config
        time_step_config = config.get("simulation", {}).get("time_step", {})

        print(f" config[time_integration]: {time_step_config}") ###DEBUG

        # Read the default time step size [s]
        dt = float(time_step_config.get("default_value", 1.e-5))  # Default to 1e-5 if not specified

        # Apply SAME timestepping for all the problems in the list, which should include self.model if it is coupling
        if hasattr(self, 'model') and self.model is not None:
            print(f" >>> Adding time stepping to the main (coupling) model: {self.model }") ###DEBUG
            problem_list = [self.model]

            # print(f" >>> self.model is found among the problems: {self.model in [problem['festim_problem'] for _,problem in self.problems.items() if 'festim_problem' in problem]}") ###DEBUG

        else:
            problem_list = []

        print (f" >> State of self.problems before adding timestepping settings: {self.problems}")

        problem_list += [problem['festim_problem'] for _,problem in self.problems.items() if 'festim_problem' in problem]

        #TODO assure that (1) all objects in the list are FESTIM2.0 problems, (2) problems do not repeat, e.g. self.model is not the reference to one of the model as in the case of no coupling

        for problem in problem_list:
            if problem.settings is not None:

                if time_step_config.get("time_stepping_type") == "fixed":
                    print(f" >>> Setting fixed timestep (dt={dt}) for {problem}") ###DEBUG
                    problem.settings.stepsize = F.Stepsize(dt)
                elif time_step_config.get("time_stepping_type") == "adaptive":
                    print(f" >>> Setting adaptive timestep for {problem}") ###DEBUG
                    stepsize_change_ratio=float(time_step_config.get("stepsize_change_ratio", 1.5))
                    max_stepsize=float(time_step_config.get("max_stepsize", 1e-3))
                    dt_min=float(time_step_config.get("min_time_step", 1e-5))
                    print(f" >>> initial stepsize = {dt} \n >>> max stepsize = {max_stepsize} \n >>> stepsize change ratio = {stepsize_change_ratio}") ###DEBUG

                    problem.settings.stepsize = F.Stepsize(
                        initial_value=dt,  # Initial time step size
                        stepsize_change_ratio=stepsize_change_ratio,  # Ratio for adaptive time stepping
                        max_stepsize=max_stepsize,  # Maximum time step size
                        dt_min=dt_min,  # Minimum time step size
                )
            else:
                raise ValueError(f"Unknown or unsupported time stepping type: {config.get('simulation', {}).get('time_stepping_type')}")
            
        return dt

    def _specify_outputs(self, config):
        """
        Specify the outputs for the model.

        Parameters:
            - config: a nested dictionary generated from the YAML config file
        """

        print(f"Specifying outputs...")

        #super()._specify_outputs(config)

        # Define outputs for the simulation - folder for saving results
        self.result_folder = str(config.get('simulation', {}).get('output_directory', ''))

        # Specify milestone times for results export
        self.milestone_times = config.get('simulation', {}).get('milestone_times', [])  # List of times to export results

        # Apply mutliple milestone output for transient/steady problems
        #if self.transient:

        print(f" >>> State of self.problems before specifying outputs: {self.problems}") ###DEBUG

        # Iterate over problems to get different QoIs
        for problem_name, problem in self.problems.items():
            print(f" >> Specifying outputs for problem: {problem}") ###DEBUG
            
            if 'festim_problem' in problem:
                # Get tritium transport related data
                if problem['qoi_name'] == 'tritium_concentration':

                    if not hasattr(problem['festim_problem'], 'exports') or problem['festim_problem'].exports is None:
                        problem['festim_problem'].exports = []

                    problem['festim_problem'].exports.append(
                        F.VTXSpeciesExport(
                            field=problem['festim_problem'].species,
                            filename=f"{self.result_folder}/tritium_concentration.vtx",
                            checkpoint=self.transient,
                        )
                    )

                    # Specify bespoke export for 1D profiles, for tritium concentration
                    problem['festim_problem'].exports.append(
                        ProfileExport(
                            field=problem['festim_problem'].species[0],
                            volume=self.domain_volumes[1],  # Assuming domain ID 1 for the profile export
                            # filename=f"{self.result_folder}/tritium_concentration_profile.txt",
                        )
                    )

                # Get heat transport related data
                if problem['qoi_name'] == 'temperature':

                    if not hasattr(problem['festim_problem'], 'exports') or problem['festim_problem'].exports is None:
                        problem['festim_problem'].exports = []

                    problem['festim_problem'].exports.append(
                        F.VTXTemperatureExport(
                            filename=f"{self.result_folder}/temperature.vtx",
                            #times=self.milestone_times,
                            #checkpoint=self.transient,
                        )
                    )

                    # # Specify bespoke export for 1D profiles, for temperature
                    # problem['festim_problem'].exports.append(
                    #     ProfileExport(
                    #         field=problem['festim_problem'].u,
                    #         volume=self.domain_volumes[1],  # Assuming domain ID 1 for the profile export
                    #         # filename=f"{self.result_folder}/temperature_profile.txt",
                    #     )
                    # )
    
            # else:    
            #     raise NotImplementedError("Exporting results is only implemented for transient problems.")

    def _add_derived_quantities(self, config):
        """
        Add derived quantities to the model.

        Parameters:
            - config: a nested dictionary generated from the YAML config file
        """

        super()._add_derived_quantities(config)

    def _export_results(self, config=None):
        """
        Export the results of the simulation.
        """
        print("Exporting results...")
        for problem_name, problem in self.problems.items():
            if 'festim_problem' in problem:
                print(f" >> Exporting results for problem: {problem_name}") 
                #self.results[problem['qoi_name']] = {}

                # Export the final state fo the fileds into results
                self.results[problem['qoi_name']] = problem['festim_problem'].u.x.array[:].copy()

                # print(f" >> Results for {problem['qoi_name']}: \n{problem['festim_problem'].u.x.array}") ###DEBUG

    def inspect_model_structure(self):
        """Print detailed structure of the FESTIM 2.0 model object."""
        print("FESTIM 2.0 Model Structure:")
        print("=" * 50)
        
        # Model object structure
        print(f"Problems: {len(self.problems)}")

        for i, problem in enumerate(self.problems.values()):

            print(f"  Problem {i}: {type(problem)} - {[attr for attr in dir(problem) if not attr.startswith('_')]}")

            print(f"{problem['festim_problem'].__dict__}")

        # Recursively print all the attributes of self.model
        print(f"Content of self.model (coupled problem)")
        for attr in dir(self.model):
            if not attr.startswith('_'):
                print(f"  {attr}: {getattr(self.model, attr)}")

    def run(self):
        """
        Run the model - performs the simulation.
        """

        # Initialize the model
        self.model.initialise()

        # Inspect the model
        # self.inspect_model_structure()

        try:
            self.model.run()
            print("FESTIM simulation completed successfully!")

            self._export_results()
            print(f"Results exported to {self.result_folder}")

        except Exception as e:
            print(f"! Error occurred while running the model: {e}")

        print(f" >>> tritium concentration profile data collected: {self.problems['tritium_transport']['festim_problem'].exports[1].data}")

        print(f" ... Finishing the simulation... \n")
        return self.results

