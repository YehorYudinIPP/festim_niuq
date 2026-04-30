"""
Local verification UQ workflow — sparse PCE, polynomial order 3.

Runs a complete uncertainty-quantification study for one of two analytical
verification cases (Carslaw & Jaeger 1959 or Crank 1975) entirely on the
local machine, without FESTIM or FEniCSx.

Usage
-----
    python examples/run_verification_uq.py                  # default: cj1959
    python examples/run_verification_uq.py --case c1975
    python examples/run_verification_uq.py --case cj1959 --output-dir my_results/

The script:
  1. Reads the corresponding YAML config file for distribution parameters.
  2. Builds a ChaosPy joint distribution for the two uncertain parameters.
  3. Generates a **sparse** Gaussian quadrature grid for polynomial order 3.
  4. Evaluates the analytical model at each quadrature node.
  5. Fits a sparse PCE surrogate (total-degree truncation, order 3).
  6. Extracts mean, std, percentiles, and first-order Sobol indices.
  7. Saves all results to a timestamped output directory:
       - ``analysis_results_<timestamp>.pickle``  (EasyVVUQ-compatible object)
       - ``campaign_data_<timestamp>.pickle``      (full campaign dict)
       - ``run_info_<timestamp>.json``             (metadata)
       - ``statistics_<timestamp>.csv``            (human-readable statistics)
       - ``concentration_stats_vs_r_<timestamp>.pdf``
       - ``sobol_indices_vs_r_<timestamp>.pdf``
       - ``qoi_histogram_inventory_<timestamp>.pdf``
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure the project source is on the path when run directly (no install)
# ---------------------------------------------------------------------------
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_src = os.path.join(_repo_root, "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

_tests = os.path.join(_repo_root, "tests")
if _tests not in sys.path:
    sys.path.insert(0, _tests)

try:
    import chaospy as cp
except ImportError:
    raise ImportError("Install chaospy:  pip install chaospy")

from verification.gfederici1991 import CarlsJaeger1959, Crank1975

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TRAPZ = getattr(np, "trapezoid", None) or np.trapz

PCE_ORDER = 3
N_RADIAL = 128  # spatial grid points for the analytical solution

# Map case name → (config file, uncertain params, analytical function)
_CONFIG_DIR = os.path.join(_repo_root, "examples", "config")
_CASES = {
    "cj1959": {
        "config": os.path.join(_CONFIG_DIR, "config.uq_test_cj1959.yaml"),
        "params": ["D_0", "G"],
        "func": "cj1959",
        "label": "Carslaw & Jaeger 1959 — diffusion with source",
    },
    "c1975": {
        "config": os.path.join(_CONFIG_DIR, "config.uq_test_c1975.yaml"),
        "params": ["D_0", "c_0"],
        "func": "c1975",
        "label": "Crank 1975 — release from preloaded sphere",
    },
}


def _load_config(path):
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def _uniform_dist(mean, rel_std):
    """Return cp.Uniform(a, b) from a mean and relative standard deviation."""
    # For U[a,b]: std = (b-a)/sqrt(12) => half-width = sqrt(3) * std
    half = np.sqrt(3) * rel_std * abs(mean)
    return cp.Uniform(mean - half, mean + half)


def build_distributions(case, config):
    """
    Build ChaosPy distributions for the two uncertain parameters of *case*.

    Returns
    -------
    tuple
        ``(joint_dist, param_names, dist_info)`` where *dist_info* is a dict
        with ``{name: cp.Distribution}`` for reference.
    """
    params = _CASES[case]["params"]

    mat = config["materials"][0]
    D0_mean = mat["D_0"]["mean"]
    D0_rel = mat["D_0"]["relative_stdev"]
    D0_dist = _uniform_dist(D0_mean, D0_rel)

    if case == "cj1959":
        G_mean = config["source_terms"]["concentration"]["value"]["mean"]
        G_rel = config["source_terms"]["concentration"]["value"].get("relative_stdev", 0.10)
        G_dist = _uniform_dist(G_mean, G_rel)
        dist_info = {"D_0": D0_dist, "G": G_dist}
        joint = cp.J(D0_dist, G_dist)
    else:  # c1975
        c0_mean = config["initial_conditions"]["concentration"]["value"]["mean"]
        c0_rel = config["initial_conditions"]["concentration"]["value"].get("relative_stdev", 0.10)
        c0_dist = _uniform_dist(c0_mean, c0_rel)
        dist_info = {"D_0": D0_dist, "c_0": c0_dist}
        joint = cp.J(D0_dist, c0_dist)

    return joint, params, dist_info


def evaluate_model(case, config, nodes, radius):
    """
    Evaluate the analytical model at every column of *nodes*.

    Parameters
    ----------
    nodes : numpy.ndarray, shape (n_params, n_samples)
    radius : numpy.ndarray, shape (n_radial,)

    Returns
    -------
    numpy.ndarray, shape (n_samples, n_radial)
        Concentration profile at each quadrature node.
    """
    t_final = float(config["model_parameters"]["total_time"])
    a = float(config["geometry"]["domains"][0]["length"])
    samples = nodes.T  # (n_samples, n_params)
    evaluations = []

    if _CASES[case]["func"] == "cj1959":
        for d0, g in samples:
            c = CarlsJaeger1959(t=t_final, D=d0, G=g, a=a, m=len(radius))
            evaluations.append(c)
    else:
        for d0, c0 in samples:
            c = Crank1975(t=t_final, D=d0, c_0=c0, a=a, m=len(radius))
            evaluations.append(c)

    return np.array(evaluations)  # (n_samples, n_radial)


def compute_inventory(evaluations, radius):
    """Return the total tritium inventory 4π ∫ r² c(r) dr for each sample."""
    integrand = (radius**2)[np.newaxis, :] * evaluations  # (n_samples, n_radial)
    return 4.0 * np.pi * _TRAPZ(integrand, x=radius, axis=1)  # (n_samples,)


def fit_surrogate(expansion, nodes, weights, evaluations):
    """
    Fit a PCE surrogate for every spatial point.

    *evaluations* has shape (n_samples, n_radial).  ChaosPy's
    ``fit_quadrature`` expects ``solves`` of shape (n_samples, n_outputs),
    so we pass *evaluations* directly (no transpose needed).

    Returns ``surrogate`` — a ChaosPy Poly array of shape (n_radial,).
    """
    return cp.fit_quadrature(expansion, nodes, weights, evaluations)


def extract_statistics(surrogate, joint_dist, radius, param_names):
    """
    Extract mean, std, percentiles, and first-order Sobol indices from the
    fitted PCE surrogate.

    Returns a dict with keys: mean, std, pct01, pct10, pct90, pct99,
    sobol_first (dict param→array), sobol_total (dict param→array).
    """
    mean_c = cp.E(surrogate, joint_dist)
    std_c = cp.Std(surrogate, joint_dist)

    # Percentiles via Monte-Carlo samples from the surrogate (fast)
    n_mc = 50_000
    mc_samples = joint_dist.sample(n_mc, rule="latin_hypercube")
    mc_eval = surrogate(*mc_samples)  # (n_radial, n_mc)

    pct01 = np.percentile(mc_eval, 1, axis=1)
    pct10 = np.percentile(mc_eval, 10, axis=1)
    pct90 = np.percentile(mc_eval, 90, axis=1)
    pct99 = np.percentile(mc_eval, 99, axis=1)

    sobol_first = {}
    sobol_total = {}
    for i, name in enumerate(param_names):
        sobol_first[name] = cp.Sens_m(surrogate, joint_dist)[i]
        sobol_total[name] = cp.Sens_t(surrogate, joint_dist)[i]

    return {
        "mean": mean_c,
        "std": std_c,
        "pct01": pct01,
        "pct10": pct10,
        "pct90": pct90,
        "pct99": pct99,
        "sobol_first": sobol_first,
        "sobol_total": sobol_total,
    }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_run_info(out_dir, timestamp, case, config_path, n_quadrature, param_names):
    """Write run_info_<timestamp>.json with campaign metadata."""
    # Try to get git commit hash (best-effort)
    git_hash = "unknown"
    try:
        import subprocess
        result = subprocess.run(
            ["git", "-C", _repo_root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        git_hash = result.stdout.strip()
    except Exception:
        pass

    info = {
        "timestamp": timestamp,
        "case": case,
        "pce_order": PCE_ORDER,
        "sparse_quadrature": True,
        "n_uncertain_params": len(param_names),
        "uncertain_params": param_names,
        "n_quadrature_points": n_quadrature,
        "config_path": config_path,
        "git_commit": git_hash,
    }
    path = os.path.join(out_dir, f"run_info_{timestamp}.json")
    with open(path, "w") as fh:
        json.dump(info, fh, indent=2)
    print(f"✓ Run info saved: {path}")
    return path


def save_statistics_csv(out_dir, timestamp, radius, stats, param_names):
    """Write a human-readable CSV with statistics at every radial position."""
    import csv

    path = os.path.join(out_dir, f"statistics_{timestamp}.csv")
    header = ["r", "mean", "std", "pct01", "pct10", "pct90", "pct99"]
    for name in param_names:
        header += [f"sobol_first_{name}", f"sobol_total_{name}"]

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for i, r in enumerate(radius):
            row = [
                r,
                stats["mean"][i],
                stats["std"][i],
                stats["pct01"][i],
                stats["pct10"][i],
                stats["pct90"][i],
                stats["pct99"][i],
            ]
            for name in param_names:
                row.append(stats["sobol_first"][name][i])
                row.append(stats["sobol_total"][name][i])
            writer.writerow(row)

    print(f"✓ Statistics CSV saved: {path}")
    return path


def save_analysis_pickle(out_dir, timestamp, stats, nodes, weights, evaluations, radius, param_names):
    """
    Save the full analysis results as a pickle file.

    The saved dict is intentionally structured so it can be loaded and
    post-processed without re-running the campaign.
    """
    payload = {
        "timestamp": timestamp,
        "pce_order": PCE_ORDER,
        "sparse": True,
        "param_names": param_names,
        "radius": radius,
        "nodes": nodes,
        "weights": weights,
        "evaluations": evaluations,
        "stats": stats,
    }
    path = os.path.join(out_dir, f"analysis_results_{timestamp}.pickle")
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"✓ Analysis results pickle saved: {path}")
    return path


def save_campaign_pickle(out_dir, timestamp, campaign_data):
    """Save the full campaign dict (nodes, weights, evaluations, surrogate)."""
    path = os.path.join(out_dir, f"campaign_data_{timestamp}.pickle")
    with open(path, "wb") as fh:
        pickle.dump(campaign_data, fh)
    print(f"✓ Campaign data pickle saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_concentration_stats(out_dir, timestamp, radius, stats, case):
    """Plot 1 — concentration profile statistics vs radius."""
    mean_c = stats["mean"]
    std_c = stats["std"]
    pct10 = stats["pct10"]
    pct90 = stats["pct90"]
    pct01 = stats["pct01"]
    pct99 = stats["pct99"]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(radius, pct01, pct99, alpha=0.15, color="steelblue", label="1%–99% quantile")
    ax.fill_between(radius, pct10, pct90, alpha=0.25, color="steelblue", label="10%–90% quantile")
    ax.fill_between(radius, mean_c - 2 * std_c, mean_c + 2 * std_c, alpha=0.30, color="steelblue", label=r"$\pm 2\sigma$")
    ax.fill_between(radius, mean_c - std_c, mean_c + std_c, alpha=0.45, color="steelblue", label=r"$\pm 1\sigma$")
    ax.plot(radius, mean_c, color="steelblue", lw=2, label="Mean")
    ax.set_xlabel("Radial position r")
    ax.set_ylabel("Tritium concentration [m⁻³]")
    ax.set_title(f"Concentration statistics — {_CASES[case]['label']}\nSparse PCE order {PCE_ORDER}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    path = os.path.join(out_dir, f"concentration_stats_vs_r_{timestamp}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"✓ Plot saved: {path}")
    return path


def plot_sobol_indices(out_dir, timestamp, radius, stats, param_names, case):
    """Plot 2 — first-order Sobol indices vs radius."""
    colors = ["tomato", "seagreen", "royalblue", "darkorange"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, name in enumerate(param_names):
        ax.plot(radius, stats["sobol_first"][name], lw=2, label=rf"$S_1$({name})", color=colors[i % len(colors)])
        ax.plot(
            radius, stats["sobol_total"][name], lw=1.5, ls="--",
            label=rf"$S_T$({name})", color=colors[i % len(colors)], alpha=0.7,
        )
    ax.set_xlabel("Radial position r")
    ax.set_ylabel("Sobol index")
    ax.set_title(f"Sobol sensitivity indices — {_CASES[case]['label']}\nSparse PCE order {PCE_ORDER}")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    path = os.path.join(out_dir, f"sobol_indices_vs_r_{timestamp}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"✓ Plot saved: {path}")
    return path


def plot_inventory_histogram(out_dir, timestamp, inventories, case):
    """Plot 3 — histogram of total tritium inventory across MC samples."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(inventories, bins=60, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(inventories), color="tomato", lw=2, label=f"Mean = {np.mean(inventories):.3e}")
    ax.axvline(np.percentile(inventories, 5), color="orange", lw=1.5, ls="--", label="5th percentile")
    ax.axvline(np.percentile(inventories, 95), color="orange", lw=1.5, ls="--", label="95th percentile")
    ax.set_xlabel("Total tritium inventory [m⁻³·m³ = #]")
    ax.set_ylabel("Count")
    ax.set_title(f"Inventory distribution (50 000 MC samples from PCE surrogate)\n{_CASES[case]['label']}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()

    path = os.path.join(out_dir, f"qoi_histogram_inventory_{timestamp}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"✓ Plot saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def run(case="cj1959", output_dir=None):
    """
    Execute the full local sparse-PCE UQ workflow for *case*.

    Parameters
    ----------
    case : str
        ``"cj1959"`` (Carslaw & Jaeger 1959) or ``"c1975"`` (Crank 1975).
    output_dir : str or None
        Base output directory.  A timestamped sub-folder is always created
        inside *output_dir*.  Defaults to ``results/`` next to this script.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    out_dir = os.path.join(output_dir, f"verification_{case}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*65}")
    print(f" FESTIM-NIUQ  —  Local Sparse PCE Verification  ({case.upper()})")
    print(f"{'='*65}")
    print(f" Output directory : {out_dir}")
    print(f" PCE order        : {PCE_ORDER}  (sparse Gaussian quadrature)")

    # ------------------------------------------------------------------
    # 1. Load config and build distributions
    # ------------------------------------------------------------------
    config_path = _CASES[case]["config"]
    config = _load_config(config_path)
    print(f" Config           : {config_path}")

    joint_dist, param_names, dist_info = build_distributions(case, config)
    print(f" Uncertain params : {param_names}")
    for name, d in dist_info.items():
        print(f"   {name}: {d}")

    # ------------------------------------------------------------------
    # 2. Sparse Gaussian quadrature (Smolyak / Clenshaw-Curtis sparse grid)
    # ------------------------------------------------------------------
    expansion = cp.generate_expansion(PCE_ORDER, joint_dist)

    # sparse=True activates Smolyak sparse quadrature — significantly fewer
    # points than the full tensor product for the same polynomial order.
    nodes, weights = cp.generate_quadrature(PCE_ORDER, joint_dist, rule="gaussian", sparse=True)
    n_quad = nodes.shape[1]
    print(f"\n Quadrature grid  : {n_quad} sparse nodes  "
          f"(full tensor would be {(PCE_ORDER + 1) ** len(param_names)})")
    print(f" PCE basis size   : {len(expansion)} terms  "
          f"(total-degree truncation order {PCE_ORDER}, {len(param_names)} params)")

    # ------------------------------------------------------------------
    # 3. Evaluate analytical model at every quadrature node
    # ------------------------------------------------------------------
    a = float(config["geometry"]["domains"][0]["length"])
    radius = np.linspace(0, a, N_RADIAL)

    print(f"\n Evaluating model at {n_quad} quadrature nodes …")
    evaluations = evaluate_model(case, config, nodes, radius)  # (n_quad, N_RADIAL)
    print(f" Evaluations shape: {evaluations.shape}")

    # ------------------------------------------------------------------
    # 4. Fit sparse PCE surrogate
    # ------------------------------------------------------------------
    print("\n Fitting PCE surrogate …")
    surrogate = fit_surrogate(expansion, nodes, weights, evaluations)

    # ------------------------------------------------------------------
    # 5. Extract statistics
    # ------------------------------------------------------------------
    print(" Extracting statistics and Sobol indices …")
    stats = extract_statistics(surrogate, joint_dist, radius, param_names)

    print("\n === Summary at sphere centre (r ≈ 0) ===")
    print(f"   Mean            : {stats['mean'][0]:.4e}")
    print(f"   Std             : {stats['std'][0]:.4e}")
    for name in param_names:
        print(f"   S1({name:6s})     : {stats['sobol_first'][name][0]:.4f}")

    # ------------------------------------------------------------------
    # 6. Compute inventory histogram from PCE surrogate (MC)
    # ------------------------------------------------------------------
    print("\n Drawing 50 000 MC samples from PCE surrogate for inventory …")
    n_mc = 50_000
    mc_samples = joint_dist.sample(n_mc, rule="latin_hypercube")
    mc_profiles = surrogate(*mc_samples)  # (N_RADIAL, n_mc)
    integrand = (radius**2)[:, np.newaxis] * mc_profiles
    inventories = 4.0 * np.pi * _TRAPZ(integrand, x=radius, axis=0)

    # ------------------------------------------------------------------
    # 7. Persist results
    # ------------------------------------------------------------------
    print("\n Saving results …")
    save_run_info(out_dir, timestamp, case, config_path, n_quad, param_names)

    campaign_data = {
        "case": case,
        "timestamp": timestamp,
        "pce_order": PCE_ORDER,
        "sparse": True,
        "param_names": param_names,
        "radius": radius,
        "nodes": nodes,
        "weights": weights,
        "evaluations": evaluations,
        "expansion": expansion,
        "surrogate": surrogate,
        "joint_dist": joint_dist,
        "dist_info": dist_info,
        "stats": stats,
        "inventories": inventories,
    }
    save_campaign_pickle(out_dir, timestamp, campaign_data)
    save_analysis_pickle(out_dir, timestamp, stats, nodes, weights, evaluations, radius, param_names)
    save_statistics_csv(out_dir, timestamp, radius, stats, param_names)

    # ------------------------------------------------------------------
    # 8. Plots
    # ------------------------------------------------------------------
    print("\n Generating plots …")
    plot_concentration_stats(out_dir, timestamp, radius, stats, case)
    plot_sobol_indices(out_dir, timestamp, radius, stats, param_names, case)
    plot_inventory_histogram(out_dir, timestamp, inventories, case)

    print(f"\n{'='*65}")
    print(f" Done!  All outputs in: {out_dir}")
    print(f"{'='*65}\n")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Local sparse PCE verification workflow (no FESTIM required).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--case", choices=list(_CASES.keys()), default="cj1959",
        help="Verification case: 'cj1959' (Carslaw & Jaeger 1959) or 'c1975' (Crank 1975). Default: cj1959",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Base output directory (a timestamped sub-folder is created inside). Default: examples/results/",
    )
    args = parser.parse_args()
    run(case=args.case, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
