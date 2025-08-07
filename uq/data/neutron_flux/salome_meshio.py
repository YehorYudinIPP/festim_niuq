import numpy as np
import meshio
import matplotlib.pyplot as plt

mesh = meshio.read("/mnt/c/Users/tmg25bcx/OneDrive - Bangor University/Research/Salome/Meshes/Mesh_1.med")
mesh.write("/mnt/c/Users/tmg25bcx/OneDrive - Bangor University/Research/Salome/Meshes/Mesh_1.xdmf")

# Generate synthetic tritium source (e.g. Gaussian centered in the domain)
points = mesh.points
x = points[:, 0]
y = points[:, 1]
#tritium_source = 1e20 * np.exp(-((x - 0.0005)**2 + (y - 0.0005)**2) / 1e-8)

# tritium_source = (
#     1e20 * np.exp(-((x - 0.0003)**2 + (y - 0.0003)**2) / 5e-9) +
#     5e19 * np.exp(-((x - 0.0007)**2 + (y - 0.0006)**2) / 2e-8) +
#     2e19 * np.exp(-((x - 0.0002)**2 + (y - 0.0008)**2) / 1e-8)
# )

np.random.seed(42)  # for reproducibility
noise = np.random.uniform(0.8, 1.2, size=x.shape)

tritium_source = (
    5e19 * np.exp(-((x - 0.0005)**2 + (y - 0.0005)**2) / 2e-8) +
    2e19
) * noise


# Add this source as point data
mesh.point_data["tritium_source"] = tritium_source

# Write new .xdmf file with source field
meshio.write("/mnt/c/Users/tmg25bcx/OneDrive - Bangor University/Research/Salome/Meshes/salome_mesh_with_source_vii.xdmf", mesh)

plt.tricontourf(x, y, tritium_source, levels=14, cmap="RdYlBu")
plt.colorbar(label="Tritium Source (H/m³/s)")
plt.title("Tritium Source Distribution")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.savefig("/mnt/c/Users/tmg25bcx/OneDrive - Bangor University/Research/Salome/Meshes/tritium_source_distribution.png")
plt.show()
