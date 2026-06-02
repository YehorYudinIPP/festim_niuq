import numpy as np
import math


def _radius_grid(a=1.0, m=128):
    """Create a radial grid for spherical 1D verification solutions."""
    return np.linspace(0.0, a, m)

def CarlsJaeger1959(t, D=1.0, G=1.0, a=1.0, n=32, m=128, r=None):
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

    if r is None:
        r = _radius_grid(a=a, m=m)
    else:
        r = np.asarray(r, dtype=float)
        if r.ndim != 1 or r.size == 0:
            raise ValueError("Radius array r must be a non-empty 1D array.")
    sum_terms = np.zeros_like(r)

    sum_terms = (G / (6 * D)) * (a**2 - r**2) # Initialize sum_terms to accumulate the series with the first term

    # Avoid division by zero in the series term; apply analytic centre limit after the loop.
    r_safe = np.where(r == 0.0, 1.0, r)

    # Center-point transient-series contribution limit:
    #   C(0,t) = G*a^2/(6D) + (2*G*a^2/D) * sum_{k=1..n} [(-1)^k/k^2 * exp(-k^2*pi^2*D*t/a^2)]
    center_series_sum = 0.0

    for k in range(1, n + 1):
        exp_k = np.exp(-k**2 * np.pi**2 * D * t / a**2)
        term = (math.pow(-1, k)/k**3) * exp_k * np.sin(k * np.pi * r / a)
        sum_terms += ((2 * G) * a**3 / (D * np.pi * r_safe)) * term
        center_series_sum += (math.pow(-1, k) / k**2) * exp_k

    # Exact r->0 limit (do not drop transient series contribution at center).
    sum_terms[r == 0] = (G / (6 * D)) * a**2 + ((2 * G) * a**2 / D) * center_series_sum

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
    c_ana = CarlsJaeger1959(t=t, D=D, G=G, a=a, n=n, r=r_ana)

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


def _read_profile_txt(profile_path):
    """Read FESTIM profile txt file with header: x,t=...s,... into arrays."""
    profile_path = str(profile_path)
    with open(profile_path, "r") as f:
        header = f.readline().strip()

    header_cols = [h.strip() for h in header.split(",")]
    if len(header_cols) < 2 or header_cols[0] != "x":
        raise ValueError(f"Unexpected profile header in {profile_path}: {header}")

    times = []
    for col in header_cols[1:]:
        if not (col.startswith("t=") and col.endswith("s")):
            raise ValueError(f"Unexpected time column '{col}' in {profile_path}")
        times.append(float(col.split("=", 1)[1].rstrip("s")))

    data = np.loadtxt(profile_path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Profile data malformed in {profile_path}")

    x = np.asarray(data[:, 0], dtype=float)
    c_sim = np.asarray(data[:, 1:], dtype=float)

    if c_sim.shape[1] != len(times):
        raise ValueError(
            f"Number of time columns in data ({c_sim.shape[1]}) does not match header ({len(times)})"
        )

    return x, np.asarray(times, dtype=float), c_sim


def PlotCJ1959VerificationDashboardFromProfile(
    profile_path,
    cfg_path=None,
    output_path=None,
    D=1.0,
    G=1.0,
    a=1.0,
    T=300.0,
    E_D=0.0,
    D_0=None,
    n=64,
    show=False,
):
    """Build a 2x2 CJ1959 verification dashboard from an existing profile file.

    Panels:
    1) concentration vs radius at final time (simulation vs analytic)
    2) squared error vs radius at final time
    3) concentration vs time at r=0 (simulation vs analytic)
    4) L2 error vs time
    """
    import matplotlib.pyplot as plt

    # Optionally pull parameters from YAML configuration.
    if cfg_path is not None:
        import yaml

        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        mat = (cfg.get("materials") or [{}])[0]
        D_0_cfg = (mat.get("D_0") or {}).get("mean", None)
        E_D_cfg = (mat.get("E_D") or {}).get("mean", E_D)
        T_cfg = ((((cfg.get("initial_conditions") or {}).get("temperature") or {}).get("value") or {}).get("mean", T))
        G_cfg = ((((cfg.get("source_terms") or {}).get("concentration") or {}).get("value") or {}).get("mean", G))
        a_cfg = ((((cfg.get("geometry") or {}).get("domains") or [{}])[0]).get("length", a))

        if D_0 is None:
            D_0 = D_0_cfg
        E_D = float(E_D_cfg)
        T = float(T_cfg)
        G = float(G_cfg)
        a = float(a_cfg)

    if D_0 is not None:
        k_b_ev_per_k = 8.617333262145e-5
        D = float(D_0) * np.exp(-float(E_D) / (k_b_ev_per_k * float(T))) if float(T) > 0.0 else float(D_0)
    else:
        D = float(D)

    x, times, c_sim = _read_profile_txt(profile_path)

    # Analytic profiles on the exact simulation grid.
    c_ana = np.zeros_like(c_sim)
    l2_err = np.zeros(len(times), dtype=float)
    for j, t in enumerate(times):
        c_ana[:, j] = CarlsJaeger1959(t=float(t), D=D, G=float(G), a=float(a), n=n, r=x)
        err = c_sim[:, j] - c_ana[:, j]
        l2_err[j] = np.sqrt(np.trapz(err**2, x=x))

    # Final-time fields and error.
    c_sim_last = c_sim[:, -1]
    c_ana_last = c_ana[:, -1]
    root_sq_err_last = np.sqrt((c_sim_last - c_ana_last) ** 2)

    # Center time series.
    center_idx = int(np.argmin(np.abs(x)))
    c_sim_center = c_sim[center_idx, :]
    c_ana_center = c_ana[center_idx, :]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1,1): concentration vs radius at final time
    ax = axes[0, 0]
    ax.plot(x, c_ana_last, label="CJ1959 analytic", color="tab:blue")
    ax.plot(x, c_sim_last, "--", label="FESTIM Simulation", color="tab:orange")
    ax.set_title(f"(1,1) Concentration vs Radius at t={times[-1]:.2e}s")
    ax.set_xlabel("Radius r [arbitrary units]")
    ax.set_ylabel("Concentration [arbitrary units]")
    ax.set_ylim(bottom=0.0)
    ax.grid(True)
    ax.legend(loc="best")

    # (1,2): root squared error vs radius at final time
    ax = axes[0, 1]
    err_floor = np.finfo(float).tiny
    root_sq_err_last_pos = np.maximum(root_sq_err_last, err_floor)
    ax.plot(x, root_sq_err_last_pos, color="tab:red")
    ax.set_title(f"(1,2) Root Squared Error vs Radius at t={times[-1]:.2e}s")
    ax.set_xlabel("Radius r [arbitrary units]")
    ax.set_ylabel("Root Squared Error [arbitrary units]")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1.0e-18)
    ax.grid(True)

    # (2,1): concentration vs time at center r=0
    ax = axes[1, 0]
    ax.plot(times, c_ana_center, marker="o", label="CJ1959 analytic", color="tab:blue")
    ax.plot(times, c_sim_center, "--", marker="s", label="FESTIM Simulation", color="tab:orange")
    ax.set_title("(2,1) Concentration vs Time at r=0")
    ax.set_xlabel("Time [arbitrary units]")
    ax.set_ylabel("Concentration [arbitrary units]")
    y_top_21 = float(max(np.max(c_ana_center), np.max(c_sim_center)))
    if y_top_21 <= 0.0:
        y_top_21 = 1.0
    ax.set_ylim(0.0, y_top_21 * 1.08)
    ax.grid(True)
    ax.legend(loc="best")

    # (2,2): L2 error vs time
    ax = axes[1, 1]
    l2_err_pos = np.maximum(l2_err, err_floor)
    ax.plot(times, l2_err_pos, marker="d", color="tab:red")
    ax.set_title("(2,2) L2 Error vs Time")
    ax.set_xlabel("Time [arbitrary units]")
    ax.set_ylabel("L2 Error [arbitrary units]")
    ax.set_yscale("log")
    y_min = float(np.min(l2_err_pos))
    y_max = float(np.max(l2_err_pos))
    if y_max <= y_min:
        y_max = y_min * 10.0
    ax.set_ylim(y_min * 0.8, y_max * 1.3)
    ax.grid(True)

    fig.suptitle("CJ1959 Verification Dashboard", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])

    if output_path:
        fig.savefig(output_path, dpi=170, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return {
        "times": times,
        "x": x,
        "c_sim_last": c_sim_last,
        "c_ana_last": c_ana_last,
        "sq_err_last": root_sq_err_last,
        "c_sim_center": c_sim_center,
        "c_ana_center": c_ana_center,
        "l2_err": l2_err,
        "D": D,
        "G": G,
        "a": a,
    }

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
