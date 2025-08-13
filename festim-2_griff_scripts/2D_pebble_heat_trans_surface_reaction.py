# Replace lines 1-9 with:
import festim as F
import numpy as np
import matplotlib.pyplot as plt
import dolfinx as f
import dolfinx.io
from dolfinx import mesh
from mpi4py import MPI  # Import MPI from mpi4py, not dolfinx
import os
import gmsh 

if not os.path.exists('results'):
    os.makedirs('results')

# GMSH 2D Pebble Mesh Creation
def create_2d_pebble_mesh():
    """
    Create a 2D rectangular mesh representing a pebble cross-section using gmsh
    """
    gmsh.initialize()
    gmsh.model.add("pebble_2d")
    
    # Pebble dimensions
    pebble_width = 1.0e-3   # 1 mm width
    pebble_height = 1.0e-3  # 1 mm height
    
    # Create rectangular geometry
    # Points (bottom-left, bottom-right, top-right, top-left)
    p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
    p2 = gmsh.model.geo.addPoint(pebble_width, 0.0, 0.0)
    p3 = gmsh.model.geo.addPoint(pebble_width, pebble_height, 0.0)
    p4 = gmsh.model.geo.addPoint(0.0, pebble_height, 0.0)
    
    # Lines
    l1 = gmsh.model.geo.addLine(p1, p2)  # Bottom edge
    l2 = gmsh.model.geo.addLine(p2, p3)  # Right edge  
    l3 = gmsh.model.geo.addLine(p3, p4)  # Top edge
    l4 = gmsh.model.geo.addLine(p4, p1)  # Left edge
    
    # Create curve loop and surface
    curve_loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
    surface = gmsh.model.geo.addPlaneSurface([curve_loop])
    
    # Synchronize geometry
    gmsh.model.geo.synchronize()
    
    # Physical groups for boundaries and volume
    gmsh.model.addPhysicalGroup(1, [l1], 1, "bottom")    # Bottom surface
    gmsh.model.addPhysicalGroup(1, [l2], 2, "right")     # Right surface (cooling)
    gmsh.model.addPhysicalGroup(1, [l3], 3, "top")       # Top surface
    gmsh.model.addPhysicalGroup(1, [l4], 4, "left")      # Left surface (neutron flux)
    gmsh.model.addPhysicalGroup(2, [surface], 5, "volume")  # Volume
    
    # Set mesh size
    characteristic_length = pebble_width / 50  # 50 elements across width
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), characteristic_length)
    
    # Generate 2D mesh
    gmsh.model.mesh.generate(2)
    
    # Convert to dolfinx mesh - FIX THE MPI REFERENCE
    mesh_2d, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
        gmsh.model, 
        comm=MPI.COMM_WORLD,  # Use MPI.COMM_WORLD instead of f.MPI.COMM_WORLD
        rank=0, 
        gdim=2
    )
    
    gmsh.finalize()
    
    return mesh_2d, cell_tags, facet_tags, pebble_width, pebble_height

# Create 2D mesh
mesh_2d, cell_tags, facet_tags, pebble_width, pebble_height = create_2d_pebble_mesh()

# Create FESTIM mesh
festim_mesh = F.Mesh(mesh_2d)

# Define material properties INCLUDING thermal properties
pebble_material = F.Material(
    name="Li2O",
    D_0=3.41795E-08,           # Diffusion pre-exponential factor [m²/s]
    E_D=0.518613,              # Activation energy [eV]
    thermal_conductivity=11,   # Thermal conductivity [W/m/K]
    density=2013,              # Density [kg/m³]
    heat_capacity=2049         # Heat capacity [J/kg/K]
)

# Define 2D subdomains using physical group tags
vol_2d = F.VolumeSubdomain(id=5, material=pebble_material)  # Volume (physical group 5)
surf_bottom = F.SurfaceSubdomain(id=1)  # Bottom surface (physical group 1)
surf_right = F.SurfaceSubdomain(id=2)   # Right surface (physical group 2)
surf_top = F.SurfaceSubdomain(id=3)     # Top surface (physical group 3)
surf_left = F.SurfaceSubdomain(id=4)    # Left surface (physical group 4)

subdomains_2d = [vol_2d, surf_bottom, surf_right, surf_top, surf_left]

print(f"2D Pebble mesh created: {pebble_width*1000:.1f} × {pebble_height*1000:.1f} mm")
print(f"Mesh elements: {mesh_2d.topology.index_map(2).size_local}")

# 1. CREATE HYDROGEN TRANSPORT PROBLEM (2D)
hydrogen_problem = F.HydrogenTransportProblem()
hydrogen_problem.mesh = festim_mesh
hydrogen_problem.subdomains = subdomains_2d
tritium_source_rate = 1e15  # Source rate [atoms/m³/s]

# Define tritium species
mobile_T = F.Species("T")
hydrogen_problem.species = [mobile_T]

# Tritium source (2D)
tritium_source = F.ParticleSource(
    value=tritium_source_rate,
    volume=vol_2d,
    species=mobile_T
)
hydrogen_problem.sources = [tritium_source]

# Define kinetic parameters
k_r0 = 1e-1    # Forward rate pre-factor
E_kr = 0.8     # Forward activation [eV]
k_d0 = 1e-3    # Backward rate pre-factor
E_kd = 0.5     # Backward activation [eV]

def purge_gas_T_pressure(t=None):
    return 1e-3   # 1 mPa purge gas tritium pressure

# 2D Surface reaction boundary conditions
# Left surface: High neutron flux side - efficient purge
left_bc_H = F.SurfaceReactionBC(
    reactant=[mobile_T],
    gas_pressure=purge_gas_T_pressure,
    k_r0=k_r0,
    E_kr=E_kr,
    k_d0=k_d0,
    E_kd=E_kd,
    subdomain=surf_left,
)

# Right surface: Cooled side - efficient purge
right_bc_H = F.SurfaceReactionBC(
    reactant=[mobile_T],
    gas_pressure=purge_gas_T_pressure,
    k_r0=k_r0,
    E_kr=E_kr,
    k_d0=k_d0,
    E_kd=E_kd,
    subdomain=surf_right,
)

# Top/bottom surfaces: Contact with adjacent pebbles - reduced purge efficiency
def reduced_purge_pressure(t=None):
    return 1e-2   # 10 mPa (higher pressure = less efficient purge)

top_bc_H = F.SurfaceReactionBC(
    reactant=[mobile_T],
    gas_pressure=reduced_purge_pressure,
    k_r0=k_r0 * 0.5,  # Reduced desorption rate (contact surface)
    E_kr=E_kr,
    k_d0=k_d0,
    E_kd=E_kd,
    subdomain=surf_top,
)

bottom_bc_H = F.SurfaceReactionBC(
    reactant=[mobile_T],
    gas_pressure=reduced_purge_pressure,
    k_r0=k_r0 * 0.5,  # Reduced desorption rate (contact surface)
    E_kr=E_kr,
    k_d0=k_d0,
    E_kd=E_kd,
    subdomain=surf_bottom,
)

hydrogen_problem.boundary_conditions = [left_bc_H, right_bc_H, top_bc_H, bottom_bc_H]

# 2. CREATE HEAT TRANSFER PROBLEM (2D)
heat_problem = F.HeatTransferProblem()
heat_problem.mesh = festim_mesh
heat_problem.subdomains = subdomains_2d
heat_value = 5e5  # Heat generation rate [W/m³]

# Heat source (2D)
heat_source = F.HeatSource(
    value=heat_value,
    volume=vol_2d
)
heat_problem.sources = [heat_source]

# Initial conditions
heat_problem.initial_conditions = [
    F.InitialTemperature(value=700)  # Initial temperature [K]
]

# 2D Heat transfer boundary conditions
# Left surface: Fixed hot temperature (neutron heating)
left_bc_T = F.FixedTemperatureBC(
    subdomain=surf_left,
    value=900  # Hot neutron-facing surface [K]
)

# Right surface: Convective + radiative cooling
def combined_cooling_flux(T, t=None):
    """
    Combined convective + radiative cooling for 2D
    """
    # Convective cooling to helium purge gas
    h_conv = 1000  # Heat transfer coefficient [W/m²/K]
    T_gas = 400    # Helium temperature [K]
    q_convective = h_conv * (T - T_gas)
    
    # Stefan-Boltzmann radiation cooling
    epsilon = 0.8      # Emissivity of Li2O surface
    sigma = 5.67e-8    # Stefan-Boltzmann constant [W/m²/K⁴]
    T_amb = 300        # Ambient temperature [K]
    q_radiation = epsilon * sigma * (T**4 - T_amb**4)
    
    return q_convective + q_radiation

right_bc_T = F.HeatFluxBC(
    subdomain=surf_right,
    value=combined_cooling_flux
)

# Top/bottom surfaces: Reduced cooling (contact with adjacent pebbles)
def contact_cooling(T, t=None):
    """
    Reduced heat transfer to adjacent pebbles
    """
    h_contact = 200    # Lower heat transfer coefficient [W/m²/K]
    T_adjacent = 750   # Adjacent pebble temperature [K]
    return h_contact * (T - T_adjacent)

top_bc_T = F.HeatFluxBC(subdomain=surf_top, value=contact_cooling)
bottom_bc_T = F.HeatFluxBC(subdomain=surf_bottom, value=contact_cooling)

heat_problem.boundary_conditions = [left_bc_T, right_bc_T, top_bc_T, bottom_bc_T]

# Simulation settings
run_time = 100

heat_problem.settings = F.Settings(
    atol=1e8,
    rtol=1e-10,
    final_time=run_time,
    transient=True
)
heat_problem.settings.stepsize = F.Stepsize(1)

hydrogen_problem.settings = F.Settings(
    atol=1e10,
    rtol=1e-10,
    final_time=run_time,
    transient=True
)
hydrogen_problem.settings.stepsize = F.Stepsize(1)

print("Creating 2D coupled problem...")

# 3. CREATE COUPLED PROBLEM
model = F.CoupledTransientHeatTransferHydrogenTransport(
    heat_problem=heat_problem,
    hydrogen_problem=hydrogen_problem
)

# Export settings
hydrogen_problem.exports = [
    F.VTXSpeciesExport(
        filename="results/hydrogen_transport/pebble_tritium_2d.bp",
        field=[mobile_T],
        checkpoint=True,
    ),
    F.XDMFExport(
        filename="results/hydrogen_transport/pebble_tritium_2d.xdmf",
        field=[mobile_T]
    )
]

# heat_problem.exports = [
#     F.VTXTemperatureExport(
#         filename="results/heat_transfer/pebble_temperature_2d.bp",
#     )
# ]

print("Initializing 2D coupled simulation...")
try:
    model.initialise()
    print("2D Initialization successful!")
except Exception as init_error:
    print(f"2D Initialization failed: {init_error}")
    exit(1)

print("Running 2D coupled simulation...")
try:
    model.run()
    print("2D Simulation completed successfully!")
except Exception as run_error:
    print(f"2D Simulation failed: {run_error}")
    exit(1)

print("Extracting 2D results...")
try:
    # Get 2D data
    final_conc_values_2d = model.hydrogen_problem.u.x.array
    final_temp_values_2d = model.heat_problem.u.x.array
    
    print(f"2D Results extracted:")
    print(f"  Concentration range: {np.min(final_conc_values_2d):.2e} to {np.max(final_conc_values_2d):.2e} atoms/m³")
    print(f"  Temperature range: {np.min(final_temp_values_2d):.1f} to {np.max(final_temp_values_2d):.1f} K")
    print(f"  2D mesh nodes: {len(final_temp_values_2d)}")
    
    # 2D-specific visualization
    plt.figure(figsize=(20, 12))
    
    # Get mesh coordinates for plotting
    mesh_coords = mesh_2d.geometry.x
    x_coords_2d = mesh_coords[:, 0] * 1000  # Convert to mm
    y_coords_2d = mesh_coords[:, 1] * 1000  # Convert to mm
    
    # Plot 1: 2D Temperature contour
    plt.subplot(2, 4, 1)
    scatter_temp = plt.scatter(x_coords_2d, y_coords_2d, c=final_temp_values_2d, cmap='Reds', s=10)
    plt.colorbar(scatter_temp, label='Temperature (K)')
    plt.xlabel('Width (mm)')
    plt.ylabel('Height (mm)')
    plt.title('2D Temperature Distribution')
    plt.axis('equal')
    
    # Plot 2: 2D Concentration contour
    plt.subplot(2, 4, 2)
    scatter_conc = plt.scatter(x_coords_2d, y_coords_2d, c=final_conc_values_2d, cmap='Blues', s=10)
    plt.colorbar(scatter_conc, label='Concentration (atoms/m³)')
    plt.xlabel('Width (mm)')
    plt.ylabel('Height (mm)')
    plt.title('2D Tritium Distribution')
    plt.axis('equal')
    
    # Plot 3: Temperature vs Concentration coupling
    plt.subplot(2, 4, 3)
    plt.scatter(final_temp_values_2d, final_conc_values_2d, c=x_coords_2d, cmap='viridis', s=10, alpha=0.6)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Concentration (atoms/m³)')
    plt.title('2D Coupling Analysis')
    cbar = plt.colorbar(label='X Position (mm)')
    
    # Plot 4: Cross-sectional profiles
    plt.subplot(2, 4, 4)
    # Find center line (y ≈ height/2)
    center_indices = np.where(np.abs(y_coords_2d - 0.5) < 0.1)[0]
    center_x = x_coords_2d[center_indices]
    center_temp = final_temp_values_2d[center_indices]
    center_conc = final_conc_values_2d[center_indices]
    
    # Sort by x coordinate
    sort_idx = np.argsort(center_x)
    plt.plot(center_x[sort_idx], center_temp[sort_idx], 'r-', linewidth=2, label='Temperature')
    plt.xlabel('Width (mm)')
    plt.ylabel('Temperature (K)', color='r')
    plt.tick_params(axis='y', labelcolor='r')
    
    ax2 = plt.gca().twinx()
    ax2.plot(center_x[sort_idx], center_conc[sort_idx], 'b-', linewidth=2, label='Concentration')
    ax2.set_ylabel('Concentration (atoms/m³)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    plt.title('2D Center Line Profiles')
    
    # Plot 5: Edge temperature analysis
    plt.subplot(2, 4, 5)
    # Left edge (x ≈ 0)
    left_indices = np.where(x_coords_2d < 0.1)[0]
    left_y = y_coords_2d[left_indices]
    left_temp = final_temp_values_2d[left_indices]
    
    # Right edge (x ≈ max)
    right_indices = np.where(x_coords_2d > np.max(x_coords_2d) - 0.1)[0]
    right_y = y_coords_2d[right_indices]
    right_temp = final_temp_values_2d[right_indices]
    
    plt.scatter(left_y, left_temp, c='red', label='Left Edge (Hot)', s=20)
    plt.scatter(right_y, right_temp, c='blue', label='Right Edge (Cool)', s=20)
    plt.xlabel('Height (mm)')
    plt.ylabel('Temperature (K)')
    plt.title('Edge Temperature Profiles')
    plt.legend()
    
    # Plot 6: 2D Statistics
    plt.subplot(2, 4, 6)
    stats_2d = {
        'Max Temp': np.max(final_temp_values_2d),
        'Min Temp': np.min(final_temp_values_2d),
        'Avg Temp': np.mean(final_temp_values_2d),
        'Max Conc': np.max(final_conc_values_2d),
        'Min Conc': np.min(final_conc_values_2d),
        'Avg Conc': np.mean(final_conc_values_2d)
    }
    
    plt.text(0.1, 0.9, f"2D Pebble Analysis", fontsize=12, weight='bold', transform=plt.gca().transAxes)
    plt.text(0.1, 0.8, f"Max Temperature: {stats_2d['Max Temp']:.1f} K", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f"Min Temperature: {stats_2d['Min Temp']:.1f} K", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f"Temperature Range: {stats_2d['Max Temp'] - stats_2d['Min Temp']:.1f} K", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f"Max Concentration: {stats_2d['Max Conc']:.2e}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.4, f"Mesh Nodes: {len(final_temp_values_2d)}", fontsize=10, transform=plt.gca().transAxes)
    plt.text(0.1, 0.3, f"Pebble Size: {pebble_width*1000:.1f}×{pebble_height*1000:.1f} mm", fontsize=10, transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('2D Simulation Summary')
    
    # Plot 7: Corner effects
    plt.subplot(2, 4, 7)
    # Find corner points
    corners = {
        'Bottom-Left': np.argmin(x_coords_2d + y_coords_2d),
        'Bottom-Right': np.argmin((np.max(x_coords_2d) - x_coords_2d) + y_coords_2d),
        'Top-Left': np.argmin(x_coords_2d + (np.max(y_coords_2d) - y_coords_2d)),
        'Top-Right': np.argmin((np.max(x_coords_2d) - x_coords_2d) + (np.max(y_coords_2d) - y_coords_2d)),
        'Center': np.argmin((x_coords_2d - np.max(x_coords_2d)/2)**2 + (y_coords_2d - np.max(y_coords_2d)/2)**2)
    }
    
    corner_temps = {name: final_temp_values_2d[idx] for name, idx in corners.items()}
    corner_concs = {name: final_conc_values_2d[idx] for name, idx in corners.items()}
    
    positions = list(corner_temps.keys())
    temps = list(corner_temps.values())
    concs = [corner_concs[pos] for pos in positions]
    
    x_pos = np.arange(len(positions))
    plt.bar(x_pos, temps, alpha=0.7, color='red', label='Temperature')
    plt.xlabel('Position')
    plt.ylabel('Temperature (K)', color='red')
    plt.xticks(x_pos, positions, rotation=45)
    plt.tick_params(axis='y', labelcolor='red')
    plt.title('2D Corner Analysis')
    
    # Plot 8: 2D Gradients
    plt.subplot(2, 4, 8)
    # Calculate approximate gradients
    max_temp_idx = np.argmax(final_temp_values_2d)
    min_temp_idx = np.argmin(final_temp_values_2d)
    
    temp_gradient = (final_temp_values_2d[max_temp_idx] - final_temp_values_2d[min_temp_idx]) / \
                   np.sqrt((x_coords_2d[max_temp_idx] - x_coords_2d[min_temp_idx])**2 + 
                          (y_coords_2d[max_temp_idx] - y_coords_2d[min_temp_idx])**2)
    
    max_conc_idx = np.argmax(final_conc_values_2d)
    min_conc_idx = np.argmin(final_conc_values_2d)
    
    conc_gradient = (final_conc_values_2d[max_conc_idx] - final_conc_values_2d[min_conc_idx]) / \
                   np.sqrt((x_coords_2d[max_conc_idx] - x_coords_2d[min_conc_idx])**2 + 
                          (y_coords_2d[max_conc_idx] - y_coords_2d[min_conc_idx])**2)
    
    gradients = ['Temperature', 'Concentration']
    gradient_values = [temp_gradient, conc_gradient/1e18]  # Scale concentration for visibility
    
    plt.bar(gradients, gradient_values, color=['red', 'blue'], alpha=0.7)
    plt.ylabel('Gradient (K/mm, 1e18 atoms/m³/mm)')
    plt.title('2D Spatial Gradients')
    
    plt.tight_layout()
    plt.savefig('results/2D_pebble_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print 2D analysis
    print("\n=== 2D FUSION BREEDING BLANKET ANALYSIS ===")
    print(f"2D Temperature range: {np.min(final_temp_values_2d):.1f} to {np.max(final_temp_values_2d):.1f} K")
    print(f"2D Concentration range: {np.min(final_conc_values_2d):.2e} to {np.max(final_conc_values_2d):.2e} atoms/m³")
    print(f"Corner temperature variation: {np.max(list(corner_temps.values())) - np.min(list(corner_temps.values())):.1f} K")
    print(f"2D vs 1D geometry ratio: {(pebble_width * pebble_height) / (pebble_width * 1):.2f}")
    
    print("2D Enhanced coupled simulation analysis complete!")

except Exception as e:
    print(f"Error in 2D analysis: {e}")

print("2D Coupled analysis complete!")