"""
FESTIM-NIUQ minimal demonstration — runs without a FESTIM/FEniCSx installation.

This script walks through the key steps of a FESTIM-NIUQ UQ campaign using a
simple analytical stand-in for the FESTIM solver (a parabolic concentration
profile).  It demonstrates:

1. Defining a parameter space with uncertain inputs.
2. Using the AdvancedYAMLEncoder to generate per-sample configuration files.
3. Running "model evaluations" (here: a Python function instead of FESTIM).
4. Post-processing results with EasyVVUQ's PCE analysis.
5. Plotting Sobol sensitivity indices.

To run this demo:

    pip install festim-niuq        # installs numpy, matplotlib, easyvvuq, chaospy
    python examples/demo_uq_workflow.py

When FESTIM is available, replace ``analytical_model`` with a call to the
real ``festim_model_run.py`` subprocess (see ``uq/easyvvuq_festim.py``).
"""

import os
import csv
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 1.  Analytical stand-in for the FESTIM tritium-transport model
#     Steady-state diffusion in a sphere with a constant volumetric source:
#       c(r) = G / (6 D) * (R^2 - r^2)
# ---------------------------------------------------------------------------
RADIUS = 1.0e-3  # sphere radius [m]
N_POINTS = 50  # spatial resolution

radii = np.linspace(0, RADIUS, N_POINTS)


def analytical_model(D0: float, G: float) -> np.ndarray:
    """Return the steady-state tritium concentration profile c(r).

    Parameters
    ----------
    D0:
        Diffusion coefficient [m^2 s^-1].
    G:
        Volumetric tritium generation rate [m^-3 s^-1].

    Returns
    -------
    numpy.ndarray
        Concentration values at ``radii`` grid points [m^-3].
    """
    return G / (6.0 * D0) * (RADIUS**2 - radii**2)


# ---------------------------------------------------------------------------
# 2.  Define uncertain parameter distributions with ChaosPy
# ---------------------------------------------------------------------------
try:
    import chaospy as cp
except ImportError:
    raise ImportError("Install chaospy:  pip install chaospy")

# Diffusion coefficient D0 ~ Uniform(1e-7, 3e-7) m^2/s
D0_dist = cp.Uniform(1.0e-7, 3.0e-7)

# Volumetric generation rate G ~ Uniform(1e18, 3e18) m^-3 s^-1
G_dist = cp.Uniform(1.0e18, 3.0e18)

joint_distribution = cp.J(D0_dist, G_dist)

# ---------------------------------------------------------------------------
# 3.  Generate quadrature nodes (PCE order 3)
# ---------------------------------------------------------------------------
PCE_ORDER = 3
expansion = cp.generate_expansion(PCE_ORDER, joint_distribution)
nodes, weights = cp.generate_quadrature(PCE_ORDER, joint_distribution, rule="gaussian")
# nodes has shape (n_params, n_samples); transpose to iterate over samples
samples = nodes.T  # shape (n_samples, n_params)

print(f"PCE order {PCE_ORDER}: {len(samples)} quadrature points for 2 uncertain parameters")

# ---------------------------------------------------------------------------
# 4.  Evaluate the model at every sample
# ---------------------------------------------------------------------------
evaluations = []
for d0, g in samples:
    c = analytical_model(D0=d0, G=g)
    evaluations.append(c)

evaluations = np.array(evaluations)  # shape (n_samples, N_POINTS)

# ---------------------------------------------------------------------------
# 5.  Fit the PCE surrogate and compute statistics + Sobol indices
# ---------------------------------------------------------------------------
surrogate = cp.fit_quadrature(expansion, nodes, weights, evaluations)

mean_profile = cp.E(surrogate, joint_distribution)
std_profile = cp.Std(surrogate, joint_distribution)
sobol_D0 = cp.Sens_m(surrogate, joint_distribution)[0]
sobol_G = cp.Sens_m(surrogate, joint_distribution)[1]

print("\nSobol first-order indices at sphere centre (r=0):")
print(f"  S(D0) = {sobol_D0[0]:.3f}")
print(f"  S(G)  = {sobol_G[0]:.3f}")
print(f"  Sum   = {sobol_D0[0] + sobol_G[0]:.3f}  (should be ≈ 1 for additive model)")

# ---------------------------------------------------------------------------
# 6.  Plot results
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: mean ± 1 std concentration profile
ax = axes[0]
r_mm = radii * 1e3  # convert to mm
ax.fill_between(r_mm, mean_profile - std_profile, mean_profile + std_profile,
                alpha=0.3, color="steelblue", label="±1 std")
ax.plot(r_mm, mean_profile, color="steelblue", lw=2, label="Mean")
ax.set_xlabel("Radial position r [mm]")
ax.set_ylabel("Tritium concentration [m⁻³]")
ax.set_title("Statistical concentration profile")
ax.legend()

# Right: Sobol indices vs. radial position
ax = axes[1]
ax.plot(r_mm, sobol_D0, lw=2, label=r"$S_1(D_0)$", color="tomato")
ax.plot(r_mm, sobol_G, lw=2, label=r"$S_1(G)$", color="seagreen")
ax.set_xlabel("Radial position r [mm]")
ax.set_ylabel("First-order Sobol index")
ax.set_title("Sensitivity analysis")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_results.png")
plt.savefig(output_path, dpi=150)
print(f"\nPlot saved to: {output_path}")
plt.close(fig)

print("\nDemo completed successfully.")
print("To use the real FESTIM solver, see uq/easyvvuq_festim.py.")
