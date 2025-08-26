import festim as F
import numpy as np
import matplotlib.pyplot as plt
import os

"""
This script sets up a 1D a coupled tritium transport and heat transfer simulation in a pebble using FESTIM.
It does not include surface reactions.
The pebble is 0.1 cm in diameter (0.001 m) and uses a 1D mesh.
For the hydrogen transport problem, it includes a constant isotropic tritium source and vacuum boundary conditions at both ends of the pebble.
For the heat transfer problem, it includes a heat source and fixed temperature boundary conditions.
"""


# Create 1D mesh for 0.1 cm pebble (0.001 m) with cartesian coordinates
pebble_length = 0.001  # 0.1 cm in meters
n_elements = 300
mesh = F.Mesh1D(np.linspace(0, pebble_length, num=n_elements))

# Define material properties INCLUDING thermal properties
pebble_material = F.Material(
    name="Li2O",
    D_0=3.41795E-08,           # Diffusion pre-exponential factor [m²/s]
    E_D=0.518613,              # Activation energy [eV]
    thermal_conductivity=11,  # Thermal conductivity [W/m/K]
    density=2013,              # Density [kg/m³]
    heat_capacity=2049         # Heat capacity [J/kg/K]
)

# Define subdomains
vol = F.VolumeSubdomain1D(id=1, borders=[0, pebble_length], material=pebble_material)
surf_left = F.SurfaceSubdomain1D(id=1, x=0)
surf_right = F.SurfaceSubdomain1D(id=2, x=pebble_length)

subdomains = [vol, surf_left, surf_right]

# 1. CREATE HYDROGEN TRANSPORT PROBLEM
hydrogen_problem = F.HydrogenTransportProblem()
hydrogen_problem.mesh = mesh
hydrogen_problem.subdomains = subdomains

# Define tritium species for hydrogen problem
mobile_T = F.Species("T")
hydrogen_problem.species = [mobile_T]

# Tritium source
tritium_source = F.ParticleSource(
    value=1e15,  # Source rate [atoms/m³/s]
    volume=vol,
    species=mobile_T
)
hydrogen_problem.sources = [tritium_source]

# Hydrogen transport boundary conditions (vacuum)
left_bc_H = F.DirichletBC(
    subdomain=surf_left, 
    value=0, 
    species=mobile_T
    )

right_bc_H = F.DirichletBC(
    subdomain=surf_right, 
    value=0, 
    species=mobile_T
)

hydrogen_problem.boundary_conditions = [left_bc_H, right_bc_H]

# 2. CREATE HEAT TRANSFER PROBLEM
heat_problem = F.HeatTransferProblem()
heat_problem.mesh = mesh
heat_problem.subdomains = subdomains

# Heat source
heat_source = F.HeatSource(
    value=5e5,   # Heat generation rate [W/m³] from neutron deposits
    volume=vol
)
heat_problem.sources = [heat_source]


# For transient heat transfer, we need BOTH initial conditions AND proper settings
heat_problem.initial_conditions = [
    F.InitialTemperature(
        value=550   # Initial temperature [K] - between boundary values
    )
]

# Make sure the boundary condition values make sense
left_bc_T = F.FixedTemperatureBC(
    subdomain=surf_left,
    value=600  # Fixed temperature [K] at boundary
)

right_bc_T = F.FixedTemperatureBC(
    subdomain=surf_right,
    value=550  # Fixed temperature [K] at boundary  
)

heat_problem.boundary_conditions = [left_bc_T, right_bc_T]

# CRITICAL: Set more reasonable tolerances for transient problems
heat_problem.settings = F.Settings(
    atol=1e8,      # Use same tolerances as working example
    rtol=1e-10,      # Use same tolerances as working example  
    final_time=200,
    transient=True  # Important: specify transient
)
heat_problem.settings.stepsize = F.Stepsize(1)

# Also fix hydrogen problem settings to match
hydrogen_problem.settings = F.Settings(
    atol=1e10,      # Use same tolerances as working example
    rtol=1e-10,      # Use same tolerances as working example
    final_time=200,
    transient=True  # Important: specify transient
)
hydrogen_problem.settings.stepsize = F.Stepsize(1)

print("Creating coupled problem")

# 3. CREATE COUPLED PROBLEM
model = F.CoupledTransientHeatTransferHydrogenTransport(
    heat_problem=heat_problem,
    hydrogen_problem=hydrogen_problem
)

model.settings = F.Settings(
    atol=1e8,      # Use same tolerances as working example 
    rtol=1e-10,      # Use same tolerances as working example
    final_time=200,  # Total simulation time [s]
    transient=True   # Important: specify transient
)
model.settings.stepsize = F.Stepsize(1)  # Time step [s

print("Setting up exports on individual problems...")

# Export hydrogen/tritium data
hydrogen_problem.exports = [
    F.VTXSpeciesExport(
        filename="results/1D/heat_trans+h_transport/pebble_tritium.bp",
        field=[mobile_T],
        checkpoint=True,
    ),
    F.XDMFExport(
        filename="results/1D/heat_trans+h_transport/pebble_tritium.xdmf",
        field=[mobile_T]
    )
]

# Export temperature data
heat_problem.exports = [
    F.VTXTemperatureExport(
        filename="results/1D/heat_trans+h_transport/pebble_temperature.bp",
    )
]
print("Individual problem exports configured")

print("Initializing coupled simulation...")
try:
    model.initialise()
    print("Initialization successful!")
except Exception as init_error:
    print(f"Initialization failed: {init_error}")
    exit(1)

print("Running coupled simulation...")
try:
    model.run()
    print("Simulation completed successfully!")

    # Use individual problem settings for reporting
    print(f"Heat problem final time: {heat_problem.settings.final_time}")
    print(f"Hydrogen problem final time: {hydrogen_problem.settings.final_time}")

except Exception as run_error:
    print(f"Simulation failed: {run_error}")
    print("Simulation did not complete successfully")
    exit(1)

# Fix 3: Update the file checking to look in the results directory
files_to_check = [
    "results/1D/heat_trans+h_transport/pebble_tritium.bp",
    "results/1D/heat_trans+h_transport/pebble_tritium.xdmf",
    "results/1D/heat_trans+h_transport/pebble_temperature.bp"
]

for filename in files_to_check:
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"{filename}: {file_size} bytes")
        if file_size == 0:
            print(f"  WARNING: {filename} is empty!")
    else:
        print(f"{filename}: File not found!")

# Access data directly from the model (more reliable than file reading)
print("Extracting data directly from model...")

try:
    # Get concentration data from hydrogen problem
    final_conc_values = model.hydrogen_problem.u.x.array
    
    # Get temperature data from heat problem  
    final_temp_values = model.heat_problem.u.x.array
    
    print(f"Successfully extracted data:")
    print(f"  Concentration range: {np.min(final_conc_values):.2e} to {np.max(final_conc_values):.2e} atoms/m³")
    print(f"  Temperature range: {np.min(final_temp_values):.1f} to {np.max(final_temp_values):.1f} K")


    # Create simplified visualization
    x_coords = np.linspace(0, pebble_length, len(final_conc_values))
    

    #reduce width of figure 
    plt.figure(figsize=(10, 8))
    # plt.figure(figsize=(15, 8))
    
    # Plot 1: Tritium concentration profile
    plt.subplot(2, 2, 1)
    plt.plot(x_coords*1000, final_conc_values, 'b-', linewidth=2)
    plt.xlabel('Position (mm)')
    plt.ylabel('Tritium Concentration (atoms/m³)')
    plt.title('Final Tritium Profile')
    plt.grid(True)
    
    # Plot 2: Temperature profile
    plt.subplot(2, 2, 2)
    plt.plot(x_coords*1000, final_temp_values, 'r-', linewidth=2)
    plt.xlabel('Position (mm)')
    plt.ylabel('Temperature (K)')
    plt.title('Final Temperature Profile')
    plt.grid(True)
    
    # Plot 3: Coupling effect - concentration vs temperature
    plt.subplot(2, 2, 3)
    scatter = plt.scatter(final_temp_values, final_conc_values, c=x_coords*1000, cmap='viridis', s=20)
    plt.xlabel('Local Temperature (K)')
    plt.ylabel('Local Concentration (atoms/m³)')
    plt.title('Concentration vs Temperature')
    plt.colorbar(scatter, label='Position (mm)')
    
    # Plot 4: Physical insights summary
    plt.subplot(2, 2, 4)
    max_temp = np.max(final_temp_values)
    min_temp = np.min(final_temp_values)
    center_temp = final_temp_values[len(final_temp_values)//2]
    
    plt.text(0.05, 0.9, f"Pebble Length: {pebble_length*1000:.1f} mm", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.8, f"Heat Source: {5e5:.1e} W/m³", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.7, f"Tritium Source: {1e15:.1e} atoms/m³/s", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.6, f"Final Time: {model.settings.final_time} s", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.5, f"Max Temperature: {max_temp:.1f} K", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.4, f"Temperature Rise: {max_temp - min_temp:.1f} K", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.3, f"Max Concentration: {np.max(final_conc_values):.2e}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.2, f"Heat BC: Fixed Temp {left_bc_T.value} K", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.1, f"H Transport BC: Vacuum (Dirichlet 0)", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.0, f"Temp-Conc Correlation: {np.corrcoef(final_temp_values, final_conc_values)[0,1]:.3f}", fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Coupled Simulation Summary')
    
    plt.tight_layout()
    plt.savefig('results/1D/heat_trans+h_transport/fixedT_vacuum_bcs/coupled_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print quantitative analysis
    print("\n=== COUPLED SIMULATION ANALYSIS ===")
    print(f"Temperature rise from heat generation: {max_temp - min_temp:.1f} K")
    print(f"Center temperature: {center_temp:.1f} K")
    print(f"Boundary temperature: {final_temp_values[0]:.1f} K")
    print(f"Maximum tritium concentration: {np.max(final_conc_values):.2e} atoms/m³")
    print(f"Temperature-concentration correlation: {np.corrcoef(final_temp_values, final_conc_values)[0,1]:.3f}")
    
    print("Enhanced coupled simulation analysis complete!")

except Exception as e:
    print(f"Error in data extraction or visualization: {e}")
    print("Check if the simulation actually completed properly")

print("Coupled analysis complete!")