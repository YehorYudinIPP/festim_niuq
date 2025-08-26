import festim as F
import numpy as np
import matplotlib.pyplot as plt

"""
NOTE:
This script sets up a 1D tritium transport simulation in a pebble using FESTIM.
It models tritium transport only without heat transfer (neutron heating) or surface reactions.
The pebble is 0.1 cm in diameter (0.001 m) and uses a 1D mesh.
The simulation includes a constant isotropic tritium source and vacuum boundary
conditions at both ends.
"""

# Create hydrogen transport problem
model = F.HydrogenTransportProblem()

# Create 1D mesh for 0.1 cm pebble (0.001 m) with cartesian coordinates
pebble_length = 0.001  # 0.1 cm in meters
n_elements = 100
model.mesh = F.Mesh1D(np.linspace(0, pebble_length, num=n_elements))

# Define material properties for the pebble (you'll need to adjust these values)
pebble_material = F.Material(
    name="Li2O",  # or whatever your pebble material is
    D_0=3.41795E-08,     # Diffusion pre-exponential factor [m²/s]
    E_D=0.518613       # Activation energy [eV] - adjust based on your material
)


# Define subdomains
vol = F.VolumeSubdomain1D(id=1, borders=[0, pebble_length], material=pebble_material)
surf_left = F.SurfaceSubdomain1D(id=1, x=0)          # Left boundary
surf_right = F.SurfaceSubdomain1D(id=2, x=pebble_length)  # Right boundary

model.subdomains = [vol, surf_left, surf_right]

# Define tritium species
mobile_T = F.Species("T")
model.species = [mobile_T]

# Add constant isotropic tritium source throughout the pebble
tritium_source = F.ParticleSource(
    value=1e15,  # Source rate [atoms/m³/s] - adjust as needed
    volume=vol,    # Applied to volume subdomain 1
    species=mobile_T
)
model.sources = [tritium_source]

# Vacuum boundary conditions (zero concentration at boundaries)
left_bc = F.DirichletBC(subdomain=surf_left, value=0, species=mobile_T)
right_bc = F.DirichletBC(subdomain=surf_right, value=0, species=mobile_T)
model.boundary_conditions = [left_bc, right_bc]

# Simulation settings
model.settings = F.Settings(
    atol=1e10,      # Absolute tolerance
    rtol=1e-10,     # Relative tolerance  
    final_time=200  # Final time [s] - adjust as needed
)
model.settings.stepsize = F.Stepsize(1)  # Time step [s]

# Set temperature (constant)
model.temperature = 700  # Temperature [K] - adjust for your conditions

# Export mobile tritium concentration vs time
model.exports = [
    F.VTXSpeciesExport(
        filename="results/1D/tritium_transport/vacuum_bcs/pebble_tritium.bp",
        field=[mobile_T],
        checkpoint=True,
    ),
    # Add export for monitoring concentration at specific points
    F.XDMFExport(
        filename="results/1D/tritium_transport/vacuum_bcspebble_results.xdmf",
        field=[mobile_T]
    )
]

# Run simulation
print("Initializing simulation...")
model.initialise()
print("Running simulation...")
model.run()
print("Simulation completed!")

# Post-process: Read final concentration field
final_concentration = F.read_function_from_file(
    filename="results/1D/tritium_transport/vacuum_bcs/pebble_tritium.bp",
    name="T",
    timestamp=model.settings.final_time,
)

print(f"Simulation completed for {pebble_length*100} cm pebble")
print(f"Final time: {model.settings.final_time} seconds")


# Replace lines 98-148 with this simplified plotting:

print("Creating visualizations...")

# Extract spatial coordinates and final concentration values
x_coords = np.linspace(0, pebble_length, n_elements)
final_conc_values = final_concentration.x.array
max_conc = np.max(final_conc_values)

# Create simple 2-panel plot
plt.figure(figsize=(12, 6))

# Plot 1: Final concentration profile
plt.subplot(1, 2, 1)
plt.plot(x_coords*1000, final_conc_values, 'b-', linewidth=2)
plt.xlabel('Position (mm)')
plt.ylabel('Tritium Concentration (atoms/m³)')
plt.title(f'Final Tritium Profile (t = {model.settings.final_time} s)')
plt.grid(True)

# Plot 2: Summary info
plt.subplot(1, 2, 2)
plt.text(0.1, 0.8, f"Pebble Length: {pebble_length*1000:.1f} mm", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.7, f"Temperature: {model.temperature} K", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.6, f"Source Rate: {1e15:.1e} atoms/m³/s", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.5, f"Final Time: {model.settings.final_time} s", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.4, f"Max Concentration: {max_conc:.2e} atoms/m³", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.3, f"Center Concentration: {final_conc_values[len(final_conc_values)//2]:.2e} atoms/m³", fontsize=12, transform=plt.gca().transAxes)
#include boundary conditions
plt.text(0.1, 0.2, "Boundary Conditions: Vacuum (Dirichlet 0)", fontsize=12, transform=plt.gca().transAxes)
plt.text(0.1, 0.1, "Mesh Type: 1D", fontsize=12, transform=plt.gca().transAxes)
plt.axis('off')
plt.title('Simulation Parameters')

plt.tight_layout()
plt.savefig('results/1D/tritium_transport/tritium_transport.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Analysis complete! Check 'tritium_transport.png' for results.")