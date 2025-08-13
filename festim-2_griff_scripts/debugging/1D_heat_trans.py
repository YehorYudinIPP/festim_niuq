import festim as F
import numpy as np
import matplotlib.pyplot as plt

# Create 1D mesh
pebble_length = 0.001
n_elements = 100
mesh = F.Mesh1D(np.linspace(0, pebble_length, num=n_elements))

# Define material WITH thermal properties (this is required!)
pebble_material = F.Material(
    name="Li2O",
    thermal_conductivity=2.5,    # W/(m·K) - REQUIRED for heat transfer
    density=2013,                # kg/m³ - REQUIRED for heat transfer
    heat_capacity=1300           # J/(kg·K) - REQUIRED for heat transfer
)

# Define subdomains
vol = F.VolumeSubdomain1D(id=1, borders=[0, pebble_length], material=pebble_material)
surf_left = F.SurfaceSubdomain1D(id=1, x=0)
surf_right = F.SurfaceSubdomain1D(id=2, x=pebble_length)

# TEST SIMPLIFIED HEAT TRANSFER
heat_model = F.HeatTransferProblem()
heat_model.mesh = mesh
heat_model.subdomains = [vol, surf_left, surf_right]

# Try different heat source syntax
heat_source = F.HeatSource(
    value=1e6,      # W/m³
    volume=vol
)
heat_model.sources = [heat_source]

# Boundary conditions with different temperatures to force gradient
left_bc = F.FixedTemperatureBC(subdomain=surf_left, value=600)   # Lower temp
right_bc = F.FixedTemperatureBC(subdomain=surf_right, value=700) # Higher temp
heat_model.boundary_conditions = [left_bc, right_bc]

# Try without initial conditions first
# heat_model.initial_conditions = []

# Settings
heat_model.settings = F.Settings(
    atol=1e-6,      # Less strict tolerance
    rtol=1e-6,      # Less strict tolerance
    final_time=10,  # Shorter time
    transient=True
)
heat_model.settings.stepsize = F.Stepsize(0.1)  # Smaller timestep

# Add thermal properties directly to the settings if needed
try:
    heat_model.thermal_conductivity = 2.5
    heat_model.density = 2013
    heat_model.heat_capacity = 1300
except:
    print("Cannot set thermal properties directly on model")

print("Testing simplified heat transfer...")
heat_model.initialise()
heat_model.run()

# Check results
if hasattr(heat_model, 'u') and heat_model.u is not None:
    temp_values = heat_model.u.x.array
    print(f"Heat-only temperature range: {np.min(temp_values):.1f} to {np.max(temp_values):.1f} K")
    
    if np.max(temp_values) > 0:
        # Plot results
        x_coords = np.linspace(0, pebble_length, len(temp_values))
        plt.figure(figsize=(10, 5))
        plt.plot(x_coords*1000, temp_values, 'r-', linewidth=2)
        plt.xlabel('Position (mm)')
        plt.ylabel('Temperature (K)')
        plt.title('Heat Transfer Only - Temperature Profile')
        plt.grid(True)
        plt.savefig('heat_only_test_fixed.png', dpi=300)
        plt.show()
    else:
        print("Temperature values are still zero - heat transfer not working")
else:
    print("No solution found - model.u is None or doesn't exist")
    print(f"Model attributes: {dir(heat_model)}")