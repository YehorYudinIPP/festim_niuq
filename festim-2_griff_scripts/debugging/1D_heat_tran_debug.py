import festim as F
import numpy as np
import matplotlib.pyplot as plt

print(f"FESTIM version: {F.__version__}")

# Try the most minimal heat transfer example possible
mesh = F.Mesh1D(np.linspace(0, 0.001, 10))  # Very simple mesh

# Try without any material properties first
simple_material = F.Material(
    name="test",
    thermal_conductivity=2.5,    # W/(m·K) - REQUIRED for heat transfer
    density=2013,                # kg/m³ - REQUIRED for heat transfer
    heat_capacity=1300    
    )

vol = F.VolumeSubdomain1D(id=1, borders=[0, 0.001], material=simple_material)
surf_left = F.SurfaceSubdomain1D(id=1, x=0)
surf_right = F.SurfaceSubdomain1D(id=2, x=0.001)

# Minimal heat transfer problem
heat_test = F.HeatTransferProblem()
heat_test.mesh = mesh
heat_test.subdomains = [vol, surf_left, surf_right]

# Just boundary conditions - no sources or initial conditions
bc_left = F.FixedTemperatureBC(subdomain=surf_left, value=300)
bc_right = F.FixedTemperatureBC(subdomain=surf_right, value=400)
heat_test.boundary_conditions = [bc_left, bc_right]

# Steady state first
heat_test.settings = F.Settings(
    atol=1e-6,      # Less strict tolerance
    rtol=1e-6,  
    transient=False
    )

print("Testing minimal steady-state heat transfer...")
try:
    heat_test.initialise()
    print("Minimal heat transfer initialized successfully")
    
    # Check if there are any attributes we're missing
    print(f"Heat test attributes: {[attr for attr in dir(heat_test) if not attr.startswith('_')]}")
    
    heat_test.run()
    
    if hasattr(heat_test, 'u') and heat_test.u is not None:
        temps = heat_test.u.x.array
        print(f"Minimal test temperatures: {np.min(temps):.1f} to {np.max(temps):.1f} K")
        
        if np.max(temps) > 0:
            print("SUCCESS: Minimal heat transfer is working!")
            plt.figure()
            plt.plot(temps)
            plt.title('Minimal Heat Transfer Test')
            plt.savefig('minimal_heat_test.png')
            plt.show()
        else:
            print("FAIL: Still getting 0.0 K temperatures")
    else:
        print("FAIL: No solution generated")
        
except Exception as e:
    print(f"Minimal test failed: {e}")
    import traceback
    traceback.print_exc()