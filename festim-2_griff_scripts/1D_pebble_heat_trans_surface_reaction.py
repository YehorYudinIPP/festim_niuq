import festim as F
import numpy as np
import matplotlib.pyplot as plt
import dolfinx as f
import os
if not os.path.exists('results'):
    os.makedirs('results')

"""
This script sets up a 1D a coupled tritium transport and heat transfer simulation in a pebble using FESTIM.
It includes surface reactions.
The pebble is 0.1 cm in diameter (0.001 m) and uses a 1D mesh.
For the hydrogen transport problem, it includes a constant isotropic tritium source. The hydrogen transport boundary conditions implemented are surface reactions with kinetics.
Heat transfer BCs -- 1: The convective heat flux of neutrons to the left side and a fixed temperature condition on the right. 2: Fixed temperature condition on the left and a radiative cooling condition on the right. 
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
tritium_source_rate = 1e15  # Source rate [atoms/m³/s]

# Define tritium species for hydrogen problem
mobile_T = F.Species("T")
hydrogen_problem.species = [mobile_T]
# trapped_T = F.Species("T_trapped") 

# Add trapping sites
# 

# Tritium source
tritium_source = F.ParticleSource(
    value=tritium_source_rate,
    volume=vol,
    species=mobile_T
)
hydrogen_problem.sources = [tritium_source]
# hydrogen_problem.traps = [trap_sites]

# Define your kinetic parameters and purge gas tritium pressure function
k_r0 = 1e-1               # Forward rate pre-factor [adjust units as needed]
E_kr = 0.8                 # Forward activation [eV]
k_d0 = 1e-3                # Backward rate pre-factor [adjust units as needed]
E_kd = 0.5                 # Backward activation [eV]

def purge_gas_T_pressure(t=None):
    # Option 1: Helium purge with low tritium contamination
    # return 1e-3   # 1 mPa (still very clean)

    # # Option 2: More realistic purge gas tritium content
    return 1e-2   # 10 mPa

    # # Option 3: Higher tritium partial pressure (less efficient purge)
    # return 0.1    # 100 mPa

    # # Option 4: Poor purge efficiency
    # return 1.0    # 1 Pa
    

# Surface reaction BC for left and right boundaries
left_bc_H = F.SurfaceReactionBC(
    reactant=[mobile_T],
    gas_pressure=purge_gas_T_pressure,
    k_r0=k_r0,
    E_kr=E_kr,
    k_d0=k_d0,
    E_kd=E_kd,
    subdomain=surf_left,
)

right_bc_H = F.SurfaceReactionBC(
    reactant=[mobile_T],
    gas_pressure=purge_gas_T_pressure,
    k_r0=k_r0,
    E_kr=E_kr,
    k_d0=k_d0,
    E_kd=E_kd,
    subdomain=surf_right,
)

hydrogen_problem.boundary_conditions = [left_bc_H, right_bc_H]

# 2. CREATE HEAT TRANSFER PROBLEM
heat_problem = F.HeatTransferProblem()
heat_problem.mesh = mesh
heat_problem.subdomains = subdomains
heat_value = 5e5 # Heat generation rate [W/m³] from neutron deposits

# Heat source
heat_source = F.HeatSource(
    value=heat_value,
    volume=vol
)
heat_problem.sources = [heat_source]

# For transient heat transfer, we need BOTH initial conditions AND proper settings
heat_problem.initial_conditions = [
    F.InitialTemperature(
        value=1000   # Initial temperature [K] - between boundary values
    )
]

# NOTE: we cannot use flux bcs for both surfaces because FESTIM requires a temperature scale. 
# Instead we use either one of two options: 
# OPTION 1: we use a fixed temperature on the left side and a convective flux on the right side.
# OPTION 2: we use a fixed temperature on the right side, and a convective flux on the left side.

# OPTION 1: Fixed temperature on the left side and convective flux on the right side
# Make sure the boundary condition values make sense
left_bc_T = F.FixedTemperatureBC(
    subdomain=surf_left,
    value=900  # Fixed temperature [K] at boundary i.e., hot from neutrons
    )

def combined_cooling_flux(T, t=None):
    """
    Combined convective + radiative cooling
    """
    # Convective cooling to helium purge gas
    h_conv = 1000  # Heat transfer coefficient [W/m²/K]
    T_gas = 650    # Helium temperature [K]
    q_convective = h_conv * (T - T_gas)
    
    # Stefan-Boltzmann radiation cooling
    epsilon = 0.8      # Emissivity of Li2O surface (typical ceramic)
    sigma = 5.67e-8    # Stefan-Boltzmann constant [W/m²/K⁴]
    T_amb = 1000        # Ambient/chamber temperature [K]
    q_radiation = epsilon * sigma * (T**4 - T_amb**4)
    
    # Total cooling flux (both effects)
    return q_convective + q_radiation

# Define the right boundary condition using the combined cooling flux function
# convective_flux = lambda T: 1000 * (T - 400)

right_bc_T = F.HeatFluxBC(
    subdomain=surf_right,
    value=combined_cooling_flux  # Temperature-dependent function or convective flux
)
# OPTION 2: Fixed temperature on the right side and convective flux on the left side

# to simulate the flux of neutrons on the surface of the pebble
# left_bc_T=F.HeatFluxBC(
#     subdomain=surf_left,
#     value=heat_value  
# )

# Fixed temperature boundary condition for the right side
# This simulates a constant temperature on the right side of the pebble
# right_bc_T = F.FixedTemperatureBC(
#     subdomain=surf_right,
#     value=750  # Fixed temperature [K] at boundary
# )

heat_problem.boundary_conditions = [left_bc_T, right_bc_T]

# simulation settings
# Run time:
run_time = 900

heat_problem.settings = F.Settings(
    atol=1e8,      # Use same tolerances as working example
    rtol=1e-10,      # Use same tolerances as working example
    final_time=run_time,
    transient=True  # Important: specify transient
)
heat_problem.settings.stepsize = F.Stepsize(1)

hydrogen_problem.settings = F.Settings(
    atol=1e10,      # Use same tolerances as working example
    rtol=1e-10,      # Use same tolerances as working example
    final_time=run_time,
    transient=True  # Important: specify transient
)
hydrogen_problem.settings.stepsize = F.Stepsize(1)

print("Creating coupled problem")

# 3. CREATE COUPLED PROBLEM
model = F.CoupledTransientHeatTransferHydrogenTransport(
    heat_problem=heat_problem,
    hydrogen_problem=hydrogen_problem
)

print("Setting up exports on individual problems...")

# Export hydrogen/tritium data
hydrogen_problem.exports = [
    F.VTXSpeciesExport(
        filename="results/hydrogen_transport/pebble_tritium.bp",
        field=[mobile_T],
        checkpoint=True,
    ),
    F.XDMFExport(
        filename="results/hydrogen_transport/pebble_tritium.xdmf",
        field=[mobile_T]
    )
]

# Export temperature data
heat_problem.exports = [
    F.VTXTemperatureExport(
        filename="results/heat_transfer/pebble_temperature.bp",
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
    "results/1D/heat_trans+h_transport/surf_reactions/pebble_tritium.bp",
    "results/1D/heat_trans+h_transport/surf_reactions/pebble_tritium.xdmf",
    "results/1D/heat_trans+h_transport/surf_reactions/pebble_temperature.bp"
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

    # Plot 4: Physical insights summary
    plt.subplot(2, 2, 4)
    #plt.text(0.05, 0.9, f"Pebble Length: {pebble_length*1000:.1f} mm", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.8, f"Heat Source: {heat_value} W/m³", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.7, f"Tritium Source: {tritium_source_rate:.1e} atoms/m³/s", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.6, f"Final Time: {heat_problem.settings.final_time} s", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.5, f"Max Temperature: {max_temp:.1f} K", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.4, f"Temperature Rise: {max_temp - min_temp:.1f} K", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.3, f"Max Concentration: {np.max(final_conc_values):.2e}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.22, f"Heat BCs: Left Surf - Fixed Temp: {left_bc_T.value} K", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.18, f"Right Surf - Convective Cooling", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.1, f"H Transport BC: Surface reaction", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.05, 0.0, f"Temp-Conc Correlation: {np.corrcoef(final_temp_values, final_conc_values)[0,1]:.3f}", fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Coupled Simulation Summary')

    plt.tight_layout()
    plt.savefig('results/1D/heat_trans+h_transport/surf_reactions/coupled_analysis_srBCs_2.png', dpi=300, bbox_inches='tight')
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


print("\n=== FUSION BREEDING BLANKET INSIGHTS ===")

# 1. Tritium Inventory Analysis
total_tritium = np.trapz(final_conc_values, x_coords) * 1e6  # Convert to atoms/m² (per unit area)
tritium_generation_rate = tritium_source_rate * pebble_length * 1e6  # atoms/m²/s
steady_state_ratio = total_tritium / (tritium_generation_rate * run_time)

print(f"\n--- Tritium Inventory Management ---")
print(f"Total tritium inventory: {total_tritium:.2e} atoms/m² (per unit pebble area)")
print(f"Tritium generation rate: {tritium_generation_rate:.2e} atoms/m²/s")
print(f"Inventory buildup ratio: {steady_state_ratio:.3f} (fraction of generated tritium retained)")

if steady_state_ratio > 0.1:
    print("HIGH INVENTORY: Significant tritium retention - purge efficiency may need improvement")
elif steady_state_ratio > 0.05:
    print("MODERATE INVENTORY: Acceptable tritium retention for breeding blanket operation")
else:
    print("LOW INVENTORY: Excellent tritium recovery - efficient purge system")

# 2. Temperature-Diffusion Coupling Analysis
diffusion_center = pebble_material.D_0 * np.exp(-pebble_material.E_D * 11604.5 / center_temp)  # eV to K conversion
diffusion_boundary = pebble_material.D_0 * np.exp(-pebble_material.E_D * 11604.5 / final_temp_values[0])
diffusion_enhancement = diffusion_center / diffusion_boundary

print(f"\n--- Temperature-Transport Coupling ---")
print(f"Diffusion coefficient at center: {diffusion_center:.2e} m²/s")
print(f"Diffusion coefficient at boundary: {diffusion_boundary:.2e} m²/s")
print(f"Temperature-enhanced diffusion factor: {diffusion_enhancement:.2f}x")

if diffusion_enhancement > 2.0:
    print("STRONG COUPLING: Temperature significantly enhances tritium transport")
elif diffusion_enhancement > 1.5:
    print("MODERATE COUPLING: Noticeable temperature effect on diffusion")
else:
    print("WEAK COUPLING: Temperature has minimal effect on transport")

# 3. Heat Transfer Performance
heat_flux_left = pebble_material.thermal_conductivity * (final_temp_values[1] - final_temp_values[0]) / (x_coords[1] - x_coords[0])
heat_flux_right = pebble_material.thermal_conductivity * (final_temp_values[-1] - final_temp_values[-2]) / (x_coords[-1] - x_coords[-2])
avg_heat_flux = (abs(heat_flux_left) + abs(heat_flux_right)) / 2

print(f"\n--- Heat Removal Analysis ---")
print(f"Heat flux at left boundary: {heat_flux_left:.1e} W/m²")
print(f"Heat flux at right boundary: {heat_flux_right:.1e} W/m²")
print(f"Average heat removal rate: {avg_heat_flux:.1e} W/m²")
print(f"Heat generation per unit area: {heat_value * pebble_length:.1e} W/m²")

# 4. Surface Reaction Efficiency
conc_surface_left = final_conc_values[0]
conc_surface_right = final_conc_values[-1]
conc_center = final_conc_values[len(final_conc_values)//2]
surface_depletion = 1 - (conc_surface_left + conc_surface_right) / (2 * conc_center)

print(f"\n--- Surface Desorption Performance ---")
print(f"Surface concentration (left): {conc_surface_left:.2e} atoms/m³")
print(f"Surface concentration (right): {conc_surface_right:.2e} atoms/m³")
print(f"Center concentration: {conc_center:.2e} atoms/m³")
print(f"Surface depletion factor: {surface_depletion:.3f}")

if surface_depletion > 0.5:
    print("EFFICIENT SURFACE DESORPTION: Strong concentration gradient toward surfaces")
elif surface_depletion > 0.2:
    print("MODERATE SURFACE DESORPTION: Noticeable surface effect")
else:
    print("POOR SURFACE DESORPTION: Surface kinetics may be limiting tritium release")

# # 6. Scaling Predictions
# print(f"\n--- Scaling to Full Breeding Blanket ---")
# pebble_volume = 4/3 * np.pi * (0.5e-3)**3  # Assuming 1mm diameter sphere
# pebbles_per_m3 = 0.64 / pebble_volume  # 64% packing fraction
# tritium_per_m3 = np.mean(final_conc_values) * pebbles_per_m3

# print(f"Estimated pebbles per m³: {pebbles_per_m3:.2e}")
# print(f"Tritium concentration in packed bed: {tritium_per_m3:.2e} atoms/m³")
# print(f"Equivalent tritium mass density: {tritium_per_m3 * 3 * 1.66e-27 / 0.64:.2e} kg/m³")  # Account for packing