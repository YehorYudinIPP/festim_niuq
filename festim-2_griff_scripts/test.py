import festim as F
import numpy as np

model = F.HydrogenTransportProblem()
model.mesh = F.Mesh1D(np.linspace(0, 1, num=200))

tungst = F.Material(name="W", D_0=1, E_D=0)
vol = F.VolumeSubdomain1D(id=1, borders=[0, 1], material=tungst)
surf_left = F.SurfaceSubdomain1D(id=1, x=0)
surf_right = F.SurfaceSubdomain1D(id=2, x=1)

model.subdomains = [vol, surf_left, surf_right]

mobile_H = F.Species("H")
#trapH = F.Species("trapH", mobile=False)

model.species = [mobile_H]

model.settings = F.Settings(atol=1e8, rtol=1e-12, final_time=100)
model.settings.stepsize = F.Stepsize(10)
model.temperature = 600


model.exports = [
    F.VTXSpeciesExport(
        filename="checkpoint.bp",
        field=[mobile_H],
        checkpoint=True,
    )
]

model.initialise()
model.run()

# try and read the file

c_H_in = F.read_function_from_file(
    filename="checkpoint.bp",
    name="H",
    timestamp=100,
)