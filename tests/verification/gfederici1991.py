import numpy as np
import math


def _radius_grid(a=1.0, m=128):
    """Create a radial grid for spherical 1D verification solutions."""
    return np.linspace(0.0, a, m)

def CarlsJaeger1959(t, D=1.0, G=1.0, a=1.0, n=32, m=128):
    """
    /**
     * @brief Solution for non-stationary 1D gas diffusion with spherical symmetry.
     *
     * This function implements the analytical solution presented in Carlslaw and Jaeger (1959)
     * for a non-stationary 1D gas diffusion problem with spherical symmetry.
     * 
     * The boundary and initial conditions are:
     *      C(r,0) = 0
     *      C(a,t) = 0
     *      dC(r,t)/dr @r=0 = 0
     *      G = const
     *
     * @param D Diffusion coefficient (default: 1.0)
     * @param G Source term (default: 1.0)
     * @param a Sphere radius (default: 1.0)
     * @param n Number of terms in the series (default: 32)
     * @param m Number of points in the radial grid (default: 128)
     * @param t Time variable
     * @return Solution value at time t
     */
    """

    if t <= 0:
        raise ValueError("Time t must be greater than 0")

    r = _radius_grid(a=a, m=m)
    sum_terms = np.zeros_like(r)

    sum_terms = (G / (6 * D)) * (a**2 - r**2) # Initialize sum_terms to accumulate the series with the first term

    # Avoid division by zero in the series term; apply analytic centre limit after the loop.
    r_safe = np.where(r == 0.0, 1.0, r)

    for k in range(1, n + 1):
        term = (math.pow(-1, k)/k**3) * np.exp(-k**2 * np.pi**2 * D * t / a**2) * np.sin(k * np.pi * r / a)
        sum_terms += ((2 * G) * a**3 / (D * np.pi * r_safe)) * term

    # Avoid division by zero for r = 0
    sum_terms[r == 0] = (G / (6 * D)) * a**2    

    solution = sum_terms
    return solution


def PlotCarlsJaeger1959(
    t=1.0,
    D=1.0,
    G=1.0,
    a=1.0,
    n=32,
    m=128,
    output_path=None,
    show=False,
):
    """Plot the Carslaw-Jaeger analytical profile at time *t*.

    Returns
    -------
    tuple
        ``(r, c)`` arrays used for the plot.
    """
    import matplotlib.pyplot as plt

    r = _radius_grid(a=a, m=m)
    c = CarlsJaeger1959(t=t, D=D, G=G, a=a, n=n, m=m)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r, c, label=f"CJ1959 analytic (t={t:.2e}s)", color="tab:blue")
    ax.set_title("Carslaw-Jaeger 1959 Analytical Solution")
    ax.set_xlabel("Radius r [m]")
    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend(loc="best")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return r, c


def PlotCarlsJaeger1959WithSimulation(
    r_sim,
    c_sim,
    t,
    D=1.0,
    G=1.0,
    a=1.0,
    n=32,
    output_path=None,
    show=False,
):
    """Overlay simulation profile and CJ1959 analytical profile at the same time."""
    import matplotlib.pyplot as plt

    r_sim = np.asarray(r_sim)
    c_sim = np.asarray(c_sim)

    if len(r_sim) == 0 or len(c_sim) == 0 or len(r_sim) != len(c_sim):
        raise ValueError("Simulation radius/concentration arrays must be non-empty and same length.")

    r_ana = r_sim
    c_ana = CarlsJaeger1959(t=t, D=D, G=G, a=a, n=n, m=len(r_sim))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(r_ana, c_ana, label="CJ1959 analytic", color="tab:blue")
    ax.plot(r_sim, c_sim, label="Simulation", color="tab:orange", linestyle="--")
    ax.set_title(f"CJ1959 Verification Overlay at t={t:.2e}s")
    ax.set_xlabel("Radius r [m]")
    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend(loc="best")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return r_ana, c_ana

def Crank1975(t, D=1.0, c_0=1.0, a=1.0, n=32, m=128):
    """
    /**
     * @brief Solution for non-stationary 1D gas diffusion with spherical symmetry.
     *
     * This function implements the analytical solution presented in Crank (1975)
     * for a non-stationary 1D gas diffusion problem with spherical symmetry.
     * 
     * The boundary and initial conditions are:
     *      C(r,0) = c_0
     *      C(a,t) = 0
     *      dC(r,t)/dr @r=0 = 0
     *      G = 0
     *
     * @param D   Diffusion coefficient (default: 1.0)
     * @param c_0 Uniform initial concentration (default: 1.0)
     * @param a   Sphere radius (default: 1.0)
     * @param n   Number of terms in the series (default: 32)
     * @param m   Number of points in the radial grid (default: 128)
     * @param t   Time variable
     * @return Solution value at time t
     */
    """

    if t <= 0:
        raise ValueError("Time t must be greater than 0")

    r = np.linspace(0, a, m)
    sum_terms = np.zeros_like(r)

    for k in range(1, n + 1):
        term = math.pow(-1, k) * np.exp(-k**2 * np.pi**2 * D * t / a**2) * np.sin(k * np.pi * r / a)
        sum_terms += (-2 * a / (np.pi * r)) * term

    sum_terms *= c_0

    # Avoid division by zero at r = 0: apply L'Hôpital's rule limit
    # lim_{r->0} sum_k (-1)^k exp(...) sin(k pi r / a) / r = sum_k (-1)^k exp(...) * k pi / a
    limit_r0 = 0.0
    for k in range(1, n + 1):
        limit_r0 += math.pow(-1, k) * np.exp(-k**2 * np.pi**2 * D * t / a**2) * (k * np.pi / a)
    sum_terms[r == 0] = c_0 * (-2 * a / np.pi) * limit_r0

    solution = sum_terms
    return solution

def PlotResults(test_function):
    """
    Plot the results of a function
    """
    import matplotlib.pyplot as plt

    t = 1.0  # Example time value
    r = np.linspace(0, 1, 128)

    # Calculate solutions
    y = test_function(t=t)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(r, y, label=test_function.__name__, color='blue')
    plt.title('Gas Diffusion Solutions')
    plt.xlabel('Radius (r)')
    plt.ylabel('Concentration (C)')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

def RunTestOfTests():
    """
    Run a test of the CarlsJaeger91 function with a sample time value.
    """
    for test_function in [CarlsJaeger1959, Crank1975]:

        print(f"Running test for {test_function.__name__}")
        try:
            t = 1.0  # Example time value
            result = test_function(t=t)
            print(f"Test passed for {test_function.__name__}, result:", result)

            PlotResults(test_function)

        except ValueError as e:
            print(f"Test failed for {test_function.__name__}:", e)

    print("All tests completed.")

if __name__ == "__main__":
    RunTestOfTests()
