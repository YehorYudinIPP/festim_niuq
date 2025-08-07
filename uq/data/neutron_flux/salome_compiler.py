import salome
salome.salome_init()

import SMESH
from salome.smesh import smeshBuilder
import numpy as np

# Access existing mesh (change 'Mesh_1' if yours is named differently)
mesh_obj = salome.myStudy.FindObjectByName("Mesh_1", SMESH.SMESH_Mesh)
if mesh_obj is None:
    raise Exception("Mesh 'Mesh_1' not found. Check your mesh name.")
mesh = mesh_obj.GetObject()

# Get all node IDs and coordinates
node_ids = mesh.GetNodesId()
coords = [mesh.GetNodeXYZ(nid) for nid in node_ids]

# Define spatial source (Gaussian centered at 0.5 mm)
source_values = []
for x, y, z in coords:
    val = 1e20 * np.exp(-((x - 0.0005)**2 + (y - 0.0005)**2) / (2 * (0.0002)**2))
    source_values.append(val)

# Assign a scalar field to nodes
field = mesh.CreateScalarMapOnNodes("TritiumSource")
for nid, val in zip(node_ids, source_values):
    field.SetScalar(nid, val)

print("✅ Tritium source field assigned to mesh nodes.")
