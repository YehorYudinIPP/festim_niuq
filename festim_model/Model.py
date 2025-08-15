#!/home/yehor/miniconda3/envs/festim-env/bin/python3
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import numpy as np

# Import FESTIM library
import festim as F

# Local imports
from .diagnostics import Diagnostics

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
                if source_specs['source_type'] == 'constant':
                    # Constant source term
                    self.model.sources.append(
                        F.Source(
                            value=float(source_specs['source_value']),  # Source term value
                            volume=1,  # Assuming a single volume for the entire mesh
                            field=field,  # Field for the source term
                        )
                    )
                    print(f" >> Using constant source term with value {source_specs['source_value']} for field {field}") ###DEBUG
                else:
                    raise ValueError(f"Unknown or unsupported source term type: {source_specs['source_type']}")

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
            if config['source_terms']['heat']['source_type'] == 'constant':
                Q_source = float(config['source_terms']['heat']['source_value'])  # Heat source term [W/m³]

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

        # Specify dictionary of problems: keys are names, values are FESTIM 2.0 problems instances
        if "problems" in config.get("model_parameters", {}):
            self.problems = {k: {} for k,_ in config["model_parameters"]["problems"].items()}
        else:
            self.problems = {'tritium_transport': {},}

        # Get common model parameters
        model_parameters_problems_config = config.get("model_parameters", {}).get("problems", {})

        # Specify materials used
        self._specify_materials(config)

        # Initialise geometry common for all problems
        # TODO that should be split into common and problem specific one
        self._specify_geometry(config)

        # Type of the models
        self.transient = config.get("model_parameters", {}).get("transient", False)

        if self.transient:
            self.total_time = config.get("model_parameters", {}).get("total_time", 1.0)

        # Initialise each problem separately, add all necessary components
        for problem_name, problem_instance in self.problems.items():
            print(f"Initialising problem for: {problem_name}")
            # Add problem-specific initialisation here
            if problem_name == "tritium_transport":
                # Read the config dictionary for the particular problem
                config_tritium_transport = model_parameters_problems_config.get("tritium_transport", {})

                problem_instance['qoi_name'] = 'tritium_concentration'

                problem_instance['festim_problem'] = F.HydrogenTransportProblem()

                # Specify settings, including transient/steady
                if self.transient:
                    problem_instance['festim_problem'].settings = F.Settings(
                        transient=True,
                        final_time=self.total_time,
                        atol=1e+6,
                        rtol=1e-6
                )
                else:
                    problem_instance['festim_problem'].settings = F.Settings(
                        transient=False,
                        atol=1e+6,
                        rtol=1e-6
                    )

                # Species in question
                species_names = config_tritium_transport.get("species", [])
                if not hasattr(self, 'species') or self.species is None:
                    self.species = {k:None for k in species_names}
                self.species['Tritium'] = F.Species("T")

                # Specify species
                problem_instance['festim_problem'].species = [v for _,v in self.species.items()]

                # Specify geometry and mesh
                problem_instance['festim_problem'].subdomains = self.subdomains
                problem_instance['festim_problem'].mesh = self.meshes[config_tritium_transport.get("domains", [{}])[0].get("id", 1)]

                # Specify Boundary Conditions
                problem_instance['festim_problem'].boundary_conditions = self._specify_boundary_conditions(config, quantity_filter='concentration')

                # Add Source terms
                problem_instance['festim_problem'].sources = self._add_source_terms(config, quantity_filter='concentration')

                self.problems[problem_name] = problem_instance

            elif problem_name == "heat_transport":
                # Read the config dictionary for the particular problem
                config_heat_transport = model_parameters_problems_config.get("heat_transport", {})

                problem_instance['qoi_name'] = 'temperature'

                problem_instance['festim_problem'] = F.HeatTransferProblem()

                # Specify settings, including transient/steady
                if self.transient:
                    problem_instance['festim_problem'].settings = F.Settings(
                        transient=True,
                        final_time=self.total_time,
                        atol=1e+6,
                        rtol=1e-6
                )
                else:
                    problem_instance['festim_problem'].settings = F.Settings(
                        transient=False,
                        atol=1e+6,
                        rtol=1e-6
                    )

                # Specify geometry and mesh
                problem_instance['festim_problem'].subdomains = self.subdomains
                problem_instance['festim_problem'].mesh = self.meshes[config_heat_transport.get("domains", [{}])[0].get("id", 1)]

                # Specify Boundary Conditions
                problem_instance['festim_problem'].boundary_conditions = self._specify_boundary_conditions(config, quantity_filter='heat')

                # Add Source terms
                problem_instance['festim_problem'].sources = self._add_source_terms(config, quantity_filter='concentration')

                self.problems[problem_name] = problem_instance

        if 'tritium_transport' in model_parameters_problems_config and 'heat_transport' in model_parameters_problems_config:
            # Add coupling model for tritium transport and heat transport
            #self.problems['tritium_heat_coupling'] = {}

            #self.problems['tritium_heat_coupling']['qoi_name'] = None

            self.model = F.CoupledTransientHeatTransferHydrogenTransport(
                heat_problem=self.problems['heat_transport']['festim_problem'],
                hydrogen_problem=self.problems['tritium_transport']['festim_problem'],
            )

            if self.transient:
                self.model.settings = F.Settings(
                    transient=True,
                    final_time=self.total_time,
                    atol=1e+6,
                    rtol=1e-6
            )
                self._specify_time_integration_settings(config)
            else:
                self.model.settings = F.Settings(
                    transient=False,
                    atol=1e+6,
                    rtol=1e-6
                )

        print(f" >> Initialised problems: \n {self.problems}")  ###DEBUG

        # Specify outputs
        self._specify_outputs(config)

        print(f"Model {self.name} initialized with {len(self.problems)} problems and {len(self.materials)} materials. Number of mesh elements: {self.n_elements}.")
        print(f" > Initialisation finished !")

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
        self.n_dimensions = config.get("geometry", {}).get("dimensionality", 1)

        # Specifying type of coordinate system: cartesian, cylindrical (polar), spherical
        self.coordinate_system_type = config.get("geometry", {}).get("coordinate_system", "cartesian")

        # Specifying type of mesh
        self.mesh_type = config.get("simulation", {}).get("mesh_type", "regular")

        # Specifying size of the mesh
        self.n_elements = int(config.get("simulation", {}).get("n_elements", 128))  # Default to 128 elements if not specified

        if self.n_dimensions == 1:
            # Create a 1D mesh

            self.domain_sizes = {}
            self.domain_volumes = {}
            self.domain_surfaces = {}
            self.subdomains = []
            self.meshes = {}

            self.max_surface_per_domain = 2

            config_domains = config.get("geometry", {}).get("domains", [{}])

            for config_domain in config_domains:

                print(f" >> Specifying domain {config_domain.get('id', 1)}") ###DEBUG

                id = int(config_domain.get("id", 1))  # Default to 1 if not specified

                self.domain_sizes[id] = float(config_domain.get("length", 1.0))

                # Create a 1D volume subdomain (coordinate interval)
                self.domain_volumes[id] = F.VolumeSubdomain1D(
                        id=id,
                        borders=[0.0, self.domain_sizes[id]],
                        material=self.materials[config_domain.get("material", 1)]
                    )

                # Create 1D surface subdomain to the left (centre, boundary points)
                self.domain_surfaces[(id-1)*self.max_surface_per_domain+1] = F.SurfaceSubdomain1D(id=(id-1)*self.max_surface_per_domain+1, x=0.0)

                # Create 1D surface subdomain to the right (outer surface, boundary points)
                self.domain_surfaces[(id-1)*self.max_surface_per_domain+2] = F.SurfaceSubdomain1D(id=(id-1)*self.max_surface_per_domain+2, x=self.domain_sizes[id])

                vertices = np.linspace(0., self.domain_sizes[id], self.n_elements + 1)
                self.vertices = vertices
                self.meshes[id] = F.Mesh1D(vertices) #TODO: double check if this is accroding to FESTIM2.0

                print(f" >> Created 1D mesh with {self.n_elements} elements for domain {id} of size {self.domain_sizes[id]}")  ###DEBUG
                print(f" >> Domain {id} volume: {self.domain_volumes[id].__dict__}")  ###DEBUG

        elif self.n_dimensions == 2:
            raise NotImplementedError("2D geometry is not implemented yet.")
        elif self.n_dimensions == 3:
            raise NotImplementedError("3D geometry is not implemented yet.")
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.n_dimensions}. Only 1D, 2D, and 3D are supported.")

        # Domains are all volumes and surfaces used in the problem
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

        if not hasattr(self, 'materials') or self.materials is None:
            self.materials = {}

        for material_config in config.get("materials", []):
            material = F.Material(
                #id=material_config.get("material_id", 1),
                name=material_config.get("material_name", "Unknown"),
                D_0=material_config.get("D_0", {}).get("mean", 0.0),
                E_D=material_config.get("E_D", {}).get("mean", 0.0),
                thermal_conductivity=material_config.get("thermal_conductivity", {}).get("mean", 0.0),
                density=material_config.get("rho", 0.0),
                heat_capacity=material_config.get("heat_capacity", 0.0)
            )
            self.materials[material_config.get("material_id", 1)] = material

        return self.materials

    def _specify_boundary_conditions(self, config, quantity_filter=None):
        """
        Specify boundary conditions for the model.
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

        # Map quantity name to their id-s otr FESTIM shorthands
        quantity_map = {'concentration': 0, 'temperature': "T"}    

        # For 1D problem, map names of the surfaces to their id-s
        surface_map = {'left': {'festim_id': 1, 'loc_id': 1}, 'right': {'festim_id': 2, 'loc_id': 2}}

        # Iterate over boundary conditions in the configuration
        for bc_quantity, bc_config in config.get("boundary_conditions", []).items():

            field = quantity_map.get(bc_quantity, "concentration")  # Default to concentration if not specified
            #TODO should it have a ValueError as a fallback?

            if bc_quantity == quantity_filter or quantity_filter is None:
                for bc_location, bc_values in bc_config.items():

                    surface_loc_id = surface_map.get(bc_location, 1).get('loc_id', None)
                    #TODO should it have ValueError as a fallback?

                    if bc_quantity == "concentration":
                        if bc_values['type'] == 'dirichlet':
                            # Dirichlet boundary condition
                            bc = F.FixedConcentrationBC(
                                species=self.species.get(bc_values.get('species', 'Tritium'), None),
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=float(bc_values.get('value', 0.0)),
                            )
                            print(f" >> Using Dirichlet BC at surface {surface_loc_id} with value {bc_values['value']} for field {field}") ###DEBUG
                        elif bc_values['type'] == 'neumann':
                            bc = F.ParticleFluxBC(
                                species=self.species.get(bc_values.get('species', 'Tritium'), None),
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=float(bc_values.get('value', 0.0)),
                            )
                        elif bc_values['type'] == 'surface_reaction':
                            raise NotImplementedError("Surface reaction boundary conditions are not implemented yet.")
                        else:
                            raise ValueError(f"Unknown or unsupported boundary condition type: {bc_values['type']}")
                    elif bc_quantity == "heat":
                        if bc_values['type'] == 'dirichlet':
                            # Dirichlet BC for temperature
                            bc = F.FixedTemperatureBC(
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=float(bc_values.get('value', 0.0)),
                            )
                        elif bc_values['type'] == 'neumann':
                            bc = F.HeatFluxBC(
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=float(bc_values.get('value', 0.0)),
                            )
                        elif bc_values['type'] == 'convective_flux':
                            T_ext = float(bc_values.get('T_ext_value', 0.0))  # External temperature
                            h_coeff = float(bc_values.get('h_coeff_value', 0.0))  # Convective heat transfer coefficient
                            bc = F.HeatFluxBC(
                                subdomain=self.domain_surfaces[surface_loc_id],
                                value=lambda x : h_coeff * (T_ext - x), 
                            )
                        else:
                            raise ValueError(f"Unknown or unsupported boundary condition type: {bc_values['type']}")

                    boundary_conditions.append(bc)

        self.boundary_conditions.extend(boundary_conditions)

        return boundary_conditions

    def _add_source_terms(self, config, quantity_filter=None):
        """
        Add source terms to the model.
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

        source_terms = []

        # Map volume domain names to their FESTIM and local id-s
        volume_map = {k: k for k in self.domain_volumes.items()} # This could be modified for a more complex mapping if needed

        # # Map source term names to their ids or FESTIM shorthands
        # source_term_map = {'concentration': 0, 'heat': 'T'}

        # Iterate over source terms in the configuration
        for source_term_name, source_term_config in config.get("source_terms", {}).items():

            if source_term_name is quantity_filter or quantity_filter is None:
                if source_term_name == "concentration":

                    source_term = F.ParticleSource(
                        value=source_term_config.get('source_value', 0.0),  # Source term value
                        volume=self.domain_volumes[volume_map.get(source_term_config.get('domain_id', 1))],
                        species=self.species.get(source_term_config.get('species', "Tritium")),
                    )

                    source_terms.append(source_term)
                elif source_term_name == "heat":

                    source_term = F.HeatSource(
                        value=source_term_config.get('source_value', 0.0),  # Source term value
                        volume=self.domain_volumes[volume_map.get(source_term_config.get('domain_id', 1))],
                    )

                    source_terms.append(source_term)
                else:
                    raise ValueError(f"Unknown or unsupported source term type: {source_term_name}")
                
        self.source_terms.extend(source_terms)

        return source_terms

    def _add_heat_conduction(self, config):
        """
        Add heat conduction to the model.
        Parameters:
            - config: a nested dictionary generated from the YAML config file
        TODO IMPLEMENT!
        TODO: Add new coupling model!
        """

        super()._add_heat_conduction(config)

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

        print(f" config[time_integration]: {config.get('time_integration', {})}") ###DEBUG

        dt = float(config.get("simulation", {}).get("time_step", 1e-5))  # Default to 1e-5 if not specified

        problem_list = [self.model] + [problem['festim_problem'] for _,problem in self.problems.items() if 'festim_problem' in problem]

        for problem in problem_list:
            if problem.settings is not None:

                if config.get("simulation", {}).get("time_stepping_type") == "fixed":
                    problem.settings.stepsize = F.Stepsize(dt)
                elif config.get("simulation", {}).get("time_stepping_type") == "adaptive":
                    problem.settings.stepsize = F.Stepsize(
                        initial_value=dt,  # Initial time step size
                        stepsize_change_ratio=float(config.get("simulation", {}).get("stepsize_change_ratio", 1.5)),  # Ratio for adaptive time stepping
                        max_stepsize=float(config.get("simulation", {}).get("max_stepsize", 1e-3)),  # Maximum time step size
                        dt_min=float(config.get("simulation", {}).get("min_time_step", 1e-5)),  # Minimum time step size
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
        self.result_folder = config.get('simulation', {}).get('output_directory', '')

        # Specify milestone times for results export
        self.milestone_times = config.get('simulation', {}).get('milestone_times', [])  # List of times to export results

        if self.transient:

            for problem_name, problem in self.problems.items():
                print(f" >> Specifying outputs for problem: {problem}") ###DEBUG
                if 'festim_problem' in problem:
                    if problem['qoi_name'] == 'tritium_concentration':
                        if not hasattr(problem['festim_problem'], 'exports') or problem['festim_problem'].exports is None:
                            problem['festim_problem'].exports = []
                        problem['festim_problem'].exports.append(
                            F.VTXSpeciesExport(
                                field=problem['festim_problem'].species,
                                filename=f"{self.result_folder}/tritium_concentration.bp",
                                checkpoint=True,
                            )
                        )
                    if problem['qoi_name'] == 'temperature':
                        if not hasattr(problem['festim_problem'], 'exports') or problem['festim_problem'].exports is None:
                            problem['festim_problem'].exports = []
                        problem['festim_problem'].exports.append(
                            F.VTXTemperatureExport(
                                filename=f"{self.result_folder}/temperature.bp",
                                #times=self.milestone_times,
                            )
                        )
        else:    
            raise NotImplementedError("Exporting results is only implemented for transient problems.")

    def _add_derived_quantities(self, config):
        """
        Add derived quantities to the model.

        Parameters:
            - config: a nested dictionary generated from the YAML config file
        """

        super()._add_derived_quantities(config)

    def _export_results(self, config):
        """
        Export the results of the simulation.
        """
        for problem_name, problem in self.problems.items():
            if 'festim_problem' in problem:
                self.results[problem['qoi_name']] = {}
                self.results[problem['qoi_name']]['final'] = problem['festim_problem'].u.x.array

    def run(self):
        """
        Run the model - performs the simulation.
        """

        # Initialize the model
        self.model.initialise()

        try:
            self.model.run()
            print("FESTIM simulation completed successfully!")


        except Exception as e:
            print(f"! Error occurred while running the model: {e}")

        print(f" ... Finishing the simulation... \n")

