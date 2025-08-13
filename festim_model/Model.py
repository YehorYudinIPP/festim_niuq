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
        # self = F.Simulation  # simulation does not run in festim 2
        # self.model = F.HydrogenTransportProblem() <---- we may need to use this in the future for the .model object 
        
        # Create the main simulation components
        self.mesh = None
        self.mesh = None
        self.materials = []
        self.subdomains = []
        self.boundary_conditions = []
        self.sources = []
        self.results = None

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

            self.settings = F.Settings(
                transient=True,  # Enable transient simulation
                final_time=float(config['model_parameters']['total_time']),  # final time of the simulation [s]
                atol=float(config['simulation']['absolute_tolerance']),  #  absolute tolerance
                rtol=float(config['simulation']['relative_tolerance']),  #  relative tolerance
            )
                    
            print("Model is set to transient simulation.")
        else:

            self.settings = F.Settings(
                transient=False,  # Enable transient simulation
                atol=float(config['simulation']['absolute_tolerance']),  #  absolute tolerance
                rtol=float(config['simulation']['relative_tolerance']),  #  relative tolerance
            )

            print("Model is set to steady-state simulation.")

        # Define model parameters: temperature (and heat transfer model)
        if 'heat_model' in config['model_parameters'] and config['model_parameters']['heat_model'] == 'heat_transfer':
            # Use heat transfer model
            self._add_heat_conduction(config)
            #TODO test thoroughly!
        else:
            self.T = config['model_parameters']['T_0'] # set fixed background temperature

        print (f" >> Using heat transfer model: {self.T.__dict__}") ###DEBUG

        # Define Boundary conditions
        self._specify_boundary_conditions(config)

        # Define source terms
        self._add_source_terms(config)

        # Define derived quantities - TODO implement
        #self.derived_quantities = self.define_derived_quantities()

        # Define exports: result format
        self._specify_outputs(config)

        # Define time stepping (for transient simulation)
        if self.settings.transient:
            self._specify_time_integration_settings(config)
        else:
            self.dt = None
        #TODO: since model convergence so quickly make time step adaptive, based on the model parameters and mesh size - exam the influence of the time step on the results

        # Added derived quantities to the model exports
        self._add_derived_quantities(['tritium_inventory'])

        print(f" > config['model_parameters'] = \n {config['model_parameters']}")  ###DEBUG
        print(f" > config['simulation'] = \n {config['simulation']}")  ###DEBUG

        print(f" > Initialisation finished! Model initialized with {self.n_elements} elements")  ###DEBUG

        #def _specify_geometry(self, config):
        """
        Specify the geometry of the FESTIM model.
        This method can be extended to include specific geometry logic.
        Can be 1D, 2D, or 3D geometry.
        """
        print("Specifying geometry...")


        # Check if the required parameters are present in the config
        if 'geometry' not in config or 'boundary_file' not in config['geometry'] or 'volume_file' not in config['geometry']:
            raise KeyError("Missing required geometry parameters in the configuration.") ###DEBUG
        print(f" > config['geometry'] = \n {config['geometry']}")   ###DEBUG
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
        # self.mesh = F.MeshFromVertices(
        #     type=self.coordinate_system_type,  # Specify (spherical) mesh type; available coordinate systems: 'cartesian', 'cylindrical', 'spherical'; default is Cartesian
        #     vertices=self.vertices,  # Use the vertices defined above
        #     )
        self.mesh = F.Mesh1D(
            vertices=self.vertices,  # Specify coordinate system type
        )
        #TODO: add a fallback for unsupported coordinate systems
        
        # Option 2) use FESTIM's Mesh - and FeniCS (Dolfin ?) objects - specific for spherical geometry
        # self.mesh = F.Mesh(
        #     type="spherical",  # Specify spherical mesh type
        # )
        #2D Case 
        
        # Check if the boundary and volume files are valid
        if not isinstance(config['geometry']['boundary_file'], str) or not config['geometry']['boundary_file']:
            raise ValueError("Invalid boundary file specified in the configuration.")
        if not isinstance(config['geometry']['volume_file'], str) or not config['geometry']['volume_file']:
            raise ValueError("Invalid volume file specified in the configuration.")
        
        self.mesh = F.MeshFromXDMF(                                        
            boundary_file=config['geometry']['boundary_file'], 
            volume_file=config['geometry']['volume_file']
        )

        return self.mesh
    
    def _specify_geometry(self, config):
        """
        Specify the geometry of the FESTIM model.
        This method can be extended to include specific geometry logic.
        Can be 1D, 2D, or 3D geometry.
        """
        print("Specifying geometry...")

        # Check if the required parameters are present in the config
        if 'geometry' not in config:
            raise KeyError("Missing 'geometry' section in the configuration.")
        
        geometry_type = config['geometry'].get('type', '1D')  # Default to 1D if not specified
        
        if geometry_type == '1D':
            print("Using 1D geometry.")
            self._create_1d_mesh(config)
            
        elif geometry_type == '2D':
            print("Using 2D geometry.")
            self._create_2d_mesh(config)
            
        elif geometry_type == '3D':
            print("Using 3D geometry.")
            self._create_3d_mesh(config)
            
        else:
            raise ValueError(f"Unsupported geometry type: {geometry_type}")
        
        print(f" > config['geometry'] = \n {config['geometry']}")

    def _create_1d_mesh(self, config):
        """Create 1D mesh from config parameters"""
        # Specifying numerical parameters of mesh: physical size and number of elements
        self.n_elements = int(config['simulation']['n_elements'])
        self.length = float(config['geometry']['length'])  # Length of the geometry [m]
        
        self.coordinate_system_type = config['geometry'].get('coordinate_system', 'cartesian')
        
        # Create vertices for the mesh - uniform mesh
        self.vertices = np.linspace(0., self.length, self.n_elements + 1)
        
        # Create 1D mesh
        self.mesh = F.Mesh1D(
            vertices=self.vertices,
        )
        
        return self.mesh

    def _create_2d_mesh(self, config):
        """Create 2D mesh from XDMF files"""
        # Check if the boundary and volume files are valid
        if 'boundary_file' not in config['geometry'] or 'volume_file' not in config['geometry']:
            raise KeyError("Missing boundary_file or volume_file for 2D geometry.")
        
        if not isinstance(config['geometry']['boundary_file'], str) or not config['geometry']['boundary_file']:
            raise ValueError("Invalid boundary file specified in the configuration.")
        if not isinstance(config['geometry']['volume_file'], str) or not config['geometry']['volume_file']:
            raise ValueError("Invalid volume file specified in the configuration.")
        
        self.mesh = F.MeshFromXDMF(                                        
            boundary_file=config['geometry']['boundary_file'], 
            volume_file=config['geometry']['volume_file']
        )
        
        return self.mesh

    def _create_3d_mesh(self, config):
        """Create 3D mesh - to be implemented"""
        # Implement 3D mesh creation logic here
        raise NotImplementedError("3D mesh creation not yet implemented")

    def _specify_boundary_conditions(self, config, quantity='concentration'):
        """
        Specify boundary conditions for the FESTIM model.
        This method can be extended to include specific boundary condition logic.
        """
        print("Specifying boundary conditions...")

        print(f" > config['boundary_conditions'] = \n {config['boundary_conditions']}")  ###DEBUG

        self.boundary_conditions = []

        # Spherical case, by default: apply DirichletBC at boundary (in relative terms, r=1.0), NeumannBC (FluxBC) at the center (r=0.0)
        
        # Iterate over BCs, quantites are on the top level
        for bc_quantity, bc_specs in config['boundary_conditions'].items():

            if bc_quantity == 'concentration':
                field = 0
            elif bc_quantity == 'temperature':
                field = "T"
            else:
                raise ValueError(f"Unknown or unsupported boundary condition quantity: {bc_quantity}")

            # Iterate over BCs specifications, locations are on the second level
            # ATTENTION: apply a simple filter e.g. only add BCs if the quantity matches the one specified
            if bc_quantity == quantity:
                for bc_location, bc_vals in bc_specs.items():

                    # ATTENTION: so far, only 1D is supported
                    if bc_location == 'left':
                        surface = 1  # Left surface (r=0.0) in spherical coordinates
                    elif bc_location == 'right':
                        surface = 2  # Right surface (r=1.0) in spherical coordinates
                    else:
                        raise ValueError(f"Unknown or unsupported boundary condition location: {bc_location}")
                    
                    if bc_vals['type'] == 'dirichlet':
                        # Dirichlet boundary condition
                        self.boundary_conditions.append(
                            F.DirichletBC(
                                value=float(bc_vals['value']),  # Boundary condition value
                                surfaces=[surface],  # Apply to the specified surface
                                field=field,  # Field for the boundary condition
                            )
                        )
                        print(f" >> Using Dirichlet BC at surface {surface} with value {bc_vals['value']} for field {field}")
                    elif bc_vals['type'] == 'neumann':
                        # Neumann boundary condition (Flux)
                        self.boundary_conditions.append(
                            F.FluxBCBase(
                                value=float(bc_vals['value']),  # Boundary condition value
                                surfaces=[surface],  # Apply to the specified surface
                                field=field,  # Field for the boundary condition
                            )
                        )
                        print(f" >> Using Neumann BC at surface {surface} with value {bc_vals['value']} for field {field}")
                    elif bc_vals['type'] == 'convective_flux':
                        # Convective flux boundary condition
                        self.boundary_conditions.append(
                            F.HeatFluxBC(
                                h_coeff=float(bc_vals['hcoeff_value']),  # Convective heat transfer coefficient
                                T_ext=float(bc_vals['Text_value']),  # External temperature
                                surfaces=[surface],  # Apply to the specified surface
                                field=field,  # Field for the boundary condition
                            )
                        )
                        print(f" >> Using Convective Flux BC at surface {surface} with h_coeff {bc_vals['hcoeff_value']} and T_ext {bc_vals['Text_value']} for field {field}")
                    else:
                        raise ValueError(f"Unknown or unsupported boundary condition type: {bc_vals['type']}")

        #print(f"Using boundary value at outer surfaces: {self.boundary_conditions[1].__dict__}") ###DEBUG
        #print(f"Using constant volumetric source term with values: {self.sources[0].__dict__}") ###DEBUG

        return self.boundary_conditions

    def _specify_materials(self, config):
        """
        Specify materials for the FESTIM model.
        This method can be extended to include specific material logic.
        """
        print("Specifying materials...")

        print(f" > config['materials'] = \n {config['materials']}")  ###DEBUG

        # Define material properties
        self.materials = [F.Material(
            name=config['materials']['name'],  # Material name
            D_0=float(config['materials']['D_0']),  # diffusion coefficient
            E_D=float(config['materials']['E_D']),  # activation energy
            thermal_conductivity=float(config['materials']['thermal_conductivity']),  # thermal conductivity
            density=float(config['materials']['rho']),  # density
            heat_capacity=float(config['materials']['heat_capacity']),  # specific heat capacity
            #solubility=float(config['materials']['solubility']),  # solubility
            )]
        # TODO: fetch data from HTM DataBase - LiO2 as an example (absent in HTM DB)

        #print(f"Using material properties: D_0={self.materials[0].D_0}, E_D={self.materials[0].E_D}, T={self.T.__dict__}") ###DEBUG

        return self.materials

    def _add_source_terms(self, config, quantity='concentration'):
        """
        Add source terms to the FESTIM model.
        This method can be extended to include specific source term logic.
        """
        print("Adding source terms...")

        self.sources = []

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
                    self.sources.append(
                        F.Source(
                            value=float(source_specs['source_value']),  # Source term value
                            volume=1,  # Assuming a single volume for the entire mesh
                            field=field,  # Field for the source term
                        )
                    )
                    print(f" >> Using constant source term with value {source_specs['source_value']} for field {field}")
                else:
                    raise ValueError(f"Unknown or unsupported source term type: {source_specs['source_type']}")

        return self.sources

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
        if self.settings.transient:
            self.T = F.HeatTransferProblem(
                #transient=True,
                initial_condition=F.InitialCondition(
                    value=float(config['model_parameters']['T_0']),  # Initial temperature [K]
                    species="T",
                ),
            )
            print(f" >> Using transient heat problem with the initial temperature: {self.T.initial_condition.value} [K]")  # Debugging output
       # else:s
            ##self.T = F.HeatTransferProblem(
               # transient=False,
            #)
           # print(f" >> Using steady-state heat problem")  # Debugging output

        # Define heat transfer coefficient and external temperature

        h_coeff = float(config['boundary_conditions']['temperature']['right']['hcoeff_value'])  # Convective heat transfer coefficient [W/(m²*K)]

        T_ext = float(config['boundary_conditions']['temperature']['right']['Text_value'])  # External temperature [K]
        # For now, use a constant external temperature

        if 'heat' in config['source_terms']:
            if config['source_terms']['heat']['source_type'] == 'constant':
                Q_source = float(config['source_terms']['heat']['source_value'])  # Heat source term [W/m³]
        else:
            print(f"Warning: No heat source term specified, using Q_source = 0.0 W/m³")
            Q_source = 0.0

        # Apply appropriate boundary conditions for heat transfer
        # At the centre (r=0), apply Dirichlet BC
        self.T.boundary_conditions.append(
            F.DirichletBC(
                subdomain=1,  # Assuming subdomain 1 corresponds to the left surface (r=0)
                value=config['boundary_conditions']['temperature']['left']['value'],  # left boundary temperature [K]
                #surfaces=[1],
                species="T",
            )
        )
        # At the outer surface (r=1), apply Convective Flux BC
        self.T.boundary_conditions.append(
            F.HeatFluxBC(
                subdomain=2,  # Assuming subdomain 2 corresponds to the right surface (r=1)
                value = config['boundary_conditions']['temperature']['right']['Text_value']
                #h_coeff=h_coeff,
                #T_ext=T_ext,  # External temperature [K]
                #surfaces=[2],
                #species="T",
            )
        )
        print(f" >> Using boundary conditions for temperature: T={self.T.boundary_conditions[0].value} [K] at surface 1, h_coeff={h_coeff}, T_ext={T_ext} [K] at surface 2")  # Debugging output

        volume = F.VolumeSubdomain(id=1, material=self.materials)  # Assuming a single volume for the entire mesh

        # Apply appropriate source terms for heat transfer
        self.T.sources.append(
            F.SourceBase(
                value=Q_source,  # Heat source term [W/m³]
                volume = volume # Assuming a single volume for the entire mesh
                #field="T",
            )
        )   
        print(f" >> Using source term for heat transfer: Q_source={Q_source} [W/m³]")  # Debugging output
        
        return self.T

    def _specify_time_integration_settings(self, config):
        """
        Specify time integration settings for the FESTIM model.
        This method can be extended to include specific time integration logic.
        """
        print("Specifying time integration settings...")

        # Define time integration settings for the simulation
        self.dt = F.Stepsize(
            #float(config['simulation']['time_step']), # op1) fixed time step size

            initial_value=float(config['simulation']['time_step']),  # op2) initial time step size for adaptive dt
            stepsize_change_ratio=float(config['simulation']['stepsize_change_ratio']),  # ratio for adaptive time stepping
            #min_value=float(config['simulation']['min_time_step']),  # minimum time step size
            max_stepsize=float(config['simulation']['max_stepsize']),  # maximum time step size
            dt_min=1e-05,  # minimum time step size for adaptive time stepping

            milestones=self.milestone_times, # check points for results export
        ) 

        return self.dt

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
        if self.settings.transient:
            # For transient simulations, export results at specified milestone times
            self.exports = [
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
            self.exports = [
                F.TXTExport(
                    field="solute",
                    filename=f"{self.result_folder}/results_tritium_concentration.txt",
                )
            ]

        # Add temperature export if heat transfer model is used
        if 'heat_model' in config['model_parameters'] and config['model_parameters']['heat_model'] == 'heat_transfer':
            if self.settings.transient:
                self.exports.append(
                    F.TXTExport(
                        field="T",
                        filename=f"{self.result_folder}/results_temperature.txt",
                        times=self.milestone_times,
                    )
                )
            else:
                self.exports.append(
                    F.TXTExport(
                        field="T",
                        filename=f"{self.result_folder}/results_temperature.txt",
                    )
                )

        return self.exports

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
        self.exports.append(derived_quantities)

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
        print(f"Model type: {type(self)}")
        print(f"Model attributes: {[attr for attr in dir(self) if not attr.startswith('_')]}")
        
        # Materials
        if hasattr(self, 'materials'):
            print(f"Materials: {len(self.materials)} materials")
            for i, mat in enumerate(self.materials):
                print(f"  Material {i}: {type(mat)} - {[attr for attr in dir(mat) if not attr.startswith('_')]}")
        
        # Boundary conditions
        if hasattr(self, 'boundary_conditions'):
            print(f"Boundary conditions: {len(self.boundary_conditions)} conditions")
            for i, bc in enumerate(self.boundary_conditions):
                print(f"  BC {i}: {type(bc)}")
        
        # Sources
        if hasattr(self, 'sources'):
            print(f"Sources: {len(self.sources)} sources")
            for i, source in enumerate(self.sources):
                print(f"  Source {i}: {type(source)}")

    def run(self):
        """
        Run the FESTIM simulation.
        This method initializes the model, runs the simulation, and stores the results.
        """
        # Initialize the model
        self.initialise()

        # Input and initalisation verification
        # self.inspect_model_structure()  ### Print the structure of the model for debugging

        # Run the simulation
        self.results = self.run()

        print("FESTIM simulation completed successfully!")

        # Output verification
        #print(f"Results: {self.results}") ### DEBUG
        self.result_flag = True

        # Export results
        # self.export_results()

        #TODO: Think of better BCs
        #TODO: Read Lithium (and LiTO) data from HTM DataBase - absent in HTM DB
        #TODO: Use proprietary visualisation and diagnostics tools
        #TODO: Explore cartesian/cylindrical/spherical geometries/coordinates/curvatures (+, diifrence to be recorded)
        #TODO: Add important physical effects
        #TODO: Couple with heat conductivity and temperature (+, in testing)

        # n_elem_print = 3
        # print(f">>> Model.run: Printing last {n_elem_print} elements of the results for last time of {self.milestone_times[-1]}: {self.results[-n_elem_print:, -1]}")  # Print last n elements of the results for the last time step ###DEBUG

        return self.results
