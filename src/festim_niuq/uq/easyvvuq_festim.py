"""
Main EasyVVUQ campaign script for FESTIM uncertainty quantification.

This module orchestrates parameter sampling, FESTIM model execution,
result collection, and statistical analysis (PCE / QMC).  It is the
primary entry point for running a UQ campaign:

    python easyvvuq_festim.py --config config/config.uq_test_cj1959.yaml

See :func:`perform_uq_festim` for the top-level workflow.
"""

import argparse
import os
import sys
import logging
import traceback
import math
from datetime import datetime
import numpy as np
from pathlib import Path

import pickle

# consider import visualisation libraries optional
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid display connection issues
import matplotlib.pyplot as plt

# Import custom YAML encoders
from .util.Encoder import YAMLEncoder, AdvancedYAMLEncoder
from .util.Decoder import MultiOutputDecoder

import chaospy as cp
import easyvvuq as uq

from easyvvuq.actions import Encode, Decode, ExecuteLocal, Actions, CreateRunDirectory
from easyvvuq.actions import QCGPJPool, EasyVVUQBasicTemplate, EasyVVUQParallelTemplate

# local imports
from .util.utils import load_config, add_timestamp_to_filename, get_festim_python, validate_execution_setup
from .util.plotting import UQPlotter
from .interactive_run_viewer import discover_run_dirs, read_run_data, generate_html

logger = logging.getLogger(__name__)


def _parse_transient_time_label(qoi: str) -> float | None:
    if not isinstance(qoi, str):
        return None
    if not (qoi.startswith("t=") and qoi.endswith("s")):
        return None
    try:
        return float(qoi.split("=", 1)[1].rstrip("s"))
    except Exception:
        return None


def _parse_flux_time_label(qoi: str) -> float | None:
    if not isinstance(qoi, str):
        return None
    if not (qoi.startswith("flux_t=") and qoi.endswith("s")):
        return None
    try:
        return float(qoi.split("=", 1)[1].rstrip("s"))
    except Exception:
        return None


def _plot_flux_uq_vs_time(results, distributions, flux_qois, plot_folder_name, plot_timestamp):
    """Plot outer-surface flux uncertainty and Sobol indices versus time."""
    entries = []
    for q in flux_qois:
        t = _parse_flux_time_label(q)
        if t is not None:
            entries.append((t, q))
    if not entries:
        return
    entries.sort(key=lambda item: item[0])

    times = []
    mean_s = []
    std_s = []
    p01_s = []
    p99_s = []
    sobol_series = {p: [] for p in distributions.keys()}

    def _to_scalar(val):
        arr = np.asarray(val, dtype=float)
        return float(arr.reshape(-1)[0]) if arr.size > 0 else np.nan

    for t, q in entries:
        try:
            mean_v = _to_scalar(results.describe(q, "mean"))
        except Exception:
            continue

        try:
            std_v = _to_scalar(results.describe(q, "std"))
        except Exception:
            std_v = 0.0

        try:
            p01_v = _to_scalar(results.describe(q, "1%"))
        except Exception:
            p01_v = np.nan

        try:
            p99_v = _to_scalar(results.describe(q, "99%"))
        except Exception:
            p99_v = np.nan

        times.append(float(t))
        mean_s.append(mean_v)
        std_s.append(std_v)
        p01_s.append(p01_v)
        p99_s.append(p99_v)

        try:
            s1 = results.sobols_first(q)
        except Exception:
            s1 = None

        for p in sobol_series.keys():
            vals = (s1 or {}).get(p, None) if isinstance(s1, dict) else None
            if vals is None:
                sobol_series[p].append(np.nan)
            else:
                sobol_series[p].append(_to_scalar(vals))

    if not times:
        return

    t_arr = np.asarray(times, dtype=float)
    mean_arr = np.asarray(mean_s, dtype=float)
    std_arr = np.asarray(std_s, dtype=float)
    p01_arr = np.asarray(p01_s, dtype=float)
    p99_arr = np.asarray(p99_s, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_arr, mean_arr, marker="o", label="flux mean")
    ax.fill_between(t_arr, mean_arr - std_arr, mean_arr + std_arr, alpha=0.2, label="+/- STD")
    if np.all(np.isfinite(p01_arr)) and np.all(np.isfinite(p99_arr)):
        ax.fill_between(t_arr, p01_arr, p99_arr, alpha=0.12, color="tab:green", label="1%-99%")
    ax.set_title("Outer-surface Flux vs Time (UQ)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Total hydrogen flux at R=R_max")
    ax.grid(True)
    if np.all(t_arr > 0):
        ax.set_xscale("log")
    ax.legend(loc="best")
    out_unc = os.path.join(
        plot_folder_name,
        add_timestamp_to_filename("flux_rmax_uncertainty_vs_time.png", plot_timestamp),
    )
    fig.savefig(out_unc, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    any_curve = False
    for p, y in sobol_series.items():
        y_arr = np.asarray(y, dtype=float)
        if np.all(np.isnan(y_arr)):
            continue
        any_curve = True
        ax.plot(t_arr, y_arr, marker="o", label=f"S1({p})")
    if any_curve:
        ax.set_title("First-order Sobol for Flux vs Time")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Sobol index")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)
        if np.all(t_arr > 0):
            ax.set_xscale("log")
        ax.legend(loc="best")
        out_s1 = os.path.join(
            plot_folder_name,
            add_timestamp_to_filename("flux_rmax_sobols_vs_time.png", plot_timestamp),
        )
        fig.savefig(out_s1, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _generate_uq_dashboard_for_campaign(campaign_timestamp: str, output_folder: str):
    """Generate interactive HTML dashboard for the campaign run set."""
    campaign_prefix = f"festim_campaign_{campaign_timestamp}_"
    cwd = os.getcwd()
    candidate_dirs = []
    for d in os.listdir(cwd):
        p = os.path.join(cwd, d)
        if os.path.isdir(p) and d.startswith(campaign_prefix):
            if os.path.isdir(os.path.join(p, "runs")):
                candidate_dirs.append(p)

    if not candidate_dirs:
        print(f"Skipping dashboard generation: no campaign run directory found for prefix '{campaign_prefix}'.")
        return

    # Use the most recently modified campaign folder if multiple are present.
    campaign_dir = max(candidate_dirs, key=lambda p: os.path.getmtime(p))
    runs_root = os.path.join(campaign_dir, "runs")

    found = discover_run_dirs(runs_root)
    if not found:
        print(f"Skipping dashboard generation: no run directories under {runs_root}")
        return

    run_data_list = []
    for run_id, run_dir in found:
        params, x, profiles = read_run_data(run_dir)
        if profiles:
            run_data_list.append((run_id, params, x, profiles))

    if not run_data_list:
        print("Skipping dashboard generation: no usable run profile data found.")
        return

    os.makedirs(output_folder, exist_ok=True)
    out_html = os.path.join(output_folder, f"uq_campaign_dashboard_{campaign_timestamp}.html")
    generate_html(run_data_list, out_html)
    print(f"✓ UQ dashboard saved: {out_html}")


def save_statistics_log(results, qois, plot_folder_name, plot_timestamp):
    """
    Save a log file with all computed UQ statistics.

    Args:
        results: EasyVVUQ analysis results object.
        qois (list[str]): List of QoI names (excluding 'x').
        plot_folder_name (str): Folder to save the log file.
        plot_timestamp (str): Timestamp for the log file name.
    """
    log_filename = os.path.join(plot_folder_name, f"uq_statistics_log_{plot_timestamp}.txt")

    logger = logging.getLogger(f"uq_stats_{plot_timestamp}")
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("UQ STATISTICS LOG")
    logger.info(f"Timestamp: {plot_timestamp}")
    logger.info("=" * 70)

    for qoi in qois:
        logger.info(f"\n--- Statistics for QoI: {qoi} ---")
        for stat_name in ["mean", "std"]:
            try:
                values = results.describe(qoi, stat_name)
                if values is not None:
                    if hasattr(values, "__len__") and len(values) > 0:
                        logger.info(
                            f"  {stat_name}: min={np.min(values):.6e}, max={np.max(values):.6e}, avg={np.mean(values):.6e}"
                        )
                        logger.debug(f"  {stat_name} (full): {values}")
                    else:
                        logger.info(f"  {stat_name}: {values}")
            except Exception as e:
                logger.debug(f"  {stat_name}: not available ({e})")

        for pct in ["1%", "10%", "median", "90%", "99%"]:
            try:
                values = results.describe(qoi, pct)
                if values is not None and not np.all(values == 0):
                    if hasattr(values, "__len__") and len(values) > 0:
                        logger.info(f"  {pct}: min={np.min(values):.6e}, max={np.max(values):.6e}")
                    else:
                        logger.info(f"  {pct}: {values}")
            except Exception:
                pass

        try:
            sobols_first = results.sobols_first(qoi)
            if sobols_first:
                non_zero = {k: v for k, v in sobols_first.items() if v is not None and not np.all(np.array(v) == 0)}
                if non_zero:
                    logger.info(f"  Sobol first-order indices:")
                    for param, vals in non_zero.items():
                        vals_arr = np.array(vals)
                        logger.info(
                            f"    {param}: min={np.min(vals_arr):.6e}, max={np.max(vals_arr):.6e}, avg={np.mean(vals_arr):.6e}"
                        )
        except Exception:
            pass

        try:
            derivs = results._get_derivatives_first(qoi)
            if derivs:
                logger.info(f"  Derivative-based sensitivity indices:")
                for param, vals in derivs.items():
                    if vals is not None:
                        vals_arr = np.array(vals)
                        logger.info(
                            f"    {param}: min={np.min(vals_arr):.6e}, max={np.max(vals_arr):.6e}, avg={np.mean(vals_arr):.6e}"
                        )
        except Exception:
            pass

    logger.info("=" * 70)
    logger.info("END OF STATISTICS LOG")
    logger.info("=" * 70)

    # Clean up handler to avoid duplicate logs on repeated calls
    logger.removeHandler(fh)
    fh.close()

    print(f"Statistics log saved to: {log_filename}")


def _plot_cj1959_uq_dashboard_2x2(results, qois, rs, cfg, plot_folder_name):
    """Create CJ1959 UQ 2x2 dashboard for campaign outputs when case matches."""
    try:
        verification_case = str((cfg.get("model_parameters", {}) or {}).get("verification_case", "")).lower()
    except Exception:
        verification_case = ""
    if "cj1959" not in verification_case:
        return

    repo_root = Path(__file__).resolve().parents[3]
    tests_dir = repo_root / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    try:
        from verification.gfederici1991 import CarlsJaeger1959
    except Exception as exc:
        print(f"Skipping CJ1959 2x2 dashboard: cannot import verification module ({exc})")
        return

    geometry_cfg = cfg.get("geometry", {}) or {}
    domains = geometry_cfg.get("domains", [{}]) or [{}]
    materials = cfg.get("materials", []) or [{}]
    domain_material_id = (domains[0] or {}).get("material", None)
    material_cfg = next((m for m in materials if m.get("material_id", None) == domain_material_id), materials[0] if materials else {})

    d0 = float((material_cfg.get("D_0", {}) or {}).get("mean", 1.0))
    e_d = float((material_cfg.get("E_D", {}) or {}).get("mean", 0.0))
    temperature = float(
        (((cfg.get("initial_conditions", {}) or {}).get("temperature", {}) or {}).get("value", {}) or {}).get(
            "mean", (cfg.get("model_parameters", {}) or {}).get("T_0", 300.0)
        )
    )
    k_b_ev_per_k = 8.617333262145e-5
    D = d0 * math.exp(-e_d / (k_b_ev_per_k * temperature)) if temperature > 0.0 else d0
    source_terms_cfg = cfg.get("source_terms", {}) or {}
    concentration_cfg = source_terms_cfg.get("concentration", {}) or {}
    source_value_cfg = concentration_cfg.get("value", {}) or {}
    source_mean = source_value_cfg.get("mean", 1.0)
    G = float(source_mean)
    a = float((domains[0] or {}).get("length", 1.0))

    transient_qois = []
    for q in qois:
        if isinstance(q, str) and q.startswith("t=") and q.endswith("s"):
            try:
                transient_qois.append((float(q.split("=", 1)[1].rstrip("s")), q))
            except Exception:
                pass
    if not transient_qois:
        return
    transient_qois.sort(key=lambda it: it[0])

    r = np.asarray(rs, dtype=float)
    if r.ndim != 1 or r.size == 0:
        return
    if r.size > 1 and np.any(np.diff(r) < 0.0):
        order = np.argsort(r)
        r = r[order]
    else:
        order = np.arange(r.size, dtype=int)
    center_idx = int(np.argmin(np.abs(r)))

    t_last, q_last = transient_qois[-1]
    try:
        mean_last = np.asarray(results.describe(q_last, "mean"), dtype=float)
        std_last = np.asarray(results.describe(q_last, "std"), dtype=float)
    except Exception:
        return
    if mean_last.size != r.size:
        return
    if std_last.shape != mean_last.shape:
        std_last = np.zeros_like(mean_last)

    mean_last = mean_last[order]
    std_last = std_last[order]
    ana_last = np.asarray(CarlsJaeger1959(t=t_last, D=D, G=G, a=a, r=r), dtype=float)

    err_last = np.abs(mean_last - ana_last)
    err_last_lo = np.abs((mean_last - std_last) - ana_last)
    err_last_hi = np.abs((mean_last + std_last) - ana_last)
    err_last_min = np.minimum(err_last_lo, err_last_hi)
    err_last_max = np.maximum(err_last_lo, err_last_hi)

    times = []
    c_mean = []
    c_std = []
    c_ana = []
    l2_err = []
    l2_err_min = []
    l2_err_max = []
    for t, q in transient_qois:
        try:
            mean_prof = np.asarray(results.describe(q, "mean"), dtype=float)
            std_prof = np.asarray(results.describe(q, "std"), dtype=float)
        except Exception:
            continue
        if mean_prof.size != r.size:
            continue
        if std_prof.shape != mean_prof.shape:
            std_prof = np.zeros_like(mean_prof)
        mean_prof = mean_prof[order]
        std_prof = std_prof[order]
        ana_prof = np.asarray(CarlsJaeger1959(t=t, D=D, G=G, a=a, r=r), dtype=float)
        l2_c = float(np.sqrt(np.trapz((mean_prof - ana_prof) ** 2, x=r)))
        l2_lo = float(np.sqrt(np.trapz(((mean_prof - std_prof) - ana_prof) ** 2, x=r)))
        l2_hi = float(np.sqrt(np.trapz(((mean_prof + std_prof) - ana_prof) ** 2, x=r)))
        times.append(float(t))
        c_mean.append(float(mean_prof[center_idx]))
        c_std.append(float(std_prof[center_idx]))
        c_ana.append(float(ana_prof[center_idx]))
        l2_err.append(l2_c)
        l2_err_min.append(min(l2_lo, l2_hi))
        l2_err_max.append(max(l2_lo, l2_hi))

    if not times:
        return

    t_arr = np.asarray(times, dtype=float)
    c_mean = np.asarray(c_mean, dtype=float)
    c_std = np.asarray(c_std, dtype=float)
    c_ana = np.asarray(c_ana, dtype=float)

    l2_err = np.asarray(l2_err, dtype=float)
    l2_err_min = np.asarray(l2_err_min, dtype=float)
    l2_err_max = np.asarray(l2_err_max, dtype=float)

    floor = np.finfo(float).tiny
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(r, ana_last, color="tab:blue", label="CJ1959 analytic")
    ax.plot(r, mean_last, "--", color="tab:orange", label="UQ mean")
    ax.fill_between(r, mean_last - std_last, mean_last + std_last, alpha=0.2, color="tab:orange", label="+/- STD")
    ax.set_title(f"(1,1) Concentration vs Radius at t={t_last:.2e}s")
    ax.set_xlabel("Radius r [m]")
    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(r, np.maximum(err_last, floor), color="tab:red", label="|UQ mean - analytic|")
    ax.fill_between(r, np.maximum(err_last_min, floor), np.maximum(err_last_max, floor), alpha=0.2, color="tab:red", label="error from +/- STD")
    ax.set_title(f"(1,2) Error vs Radius at t={t_last:.2e}s")
    ax.set_xlabel("Radius r [m]")
    ax.set_ylabel("Absolute Error")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(t_arr, c_ana, marker="o", color="tab:blue", label="CJ1959 analytic @ r=0")
    ax.plot(t_arr, c_mean, "--", marker="s", color="tab:orange", label="UQ mean @ r=0")
    ax.fill_between(t_arr, c_mean - c_std, c_mean + c_std, alpha=0.2, color="tab:orange", label="+/- STD")
    ax.set_title("(2,1) Concentration vs Time at r=0")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend(loc="best")

    ax = axes[1, 1]
    ax.plot(t_arr, np.maximum(l2_err, floor), marker="d", color="tab:red", label="L2 error")
    ax.fill_between(
        t_arr,
        np.maximum(l2_err_min, floor),
        np.maximum(l2_err_max, floor),
        alpha=0.2,
        color="tab:red",
        label="L2 error from +/- STD",
    )
    ax.set_title("(2,2) L2 Error vs Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Absolute Error")
    ax.set_yscale("log")
    ax.grid(True)
    ax.legend(loc="best")

    fig.suptitle("CJ1959 Verification Dashboard (UQ 2x2)", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.97])
    out = os.path.join(plot_folder_name, "cj1959_verification_dashboard_2x2.png")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ CJ1959 UQ dashboard saved: {out}")


def visualisation_of_results(results, distributions, qois, plot_folder_name, plot_timestamp, runs_info=None, config=None):
    """
    Visualise the results of the EasyVVUQ campaign.
    This function is a placeholder for future visualisation methods.
    """

    print("Visualising results...")
    # Plot the results: error bar for each QoI + other plots

    # Create a common timestamp for all plots from this run, if none
    if not plot_timestamp:
        plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a folder to save plots
    # Assumes that by default there is no folder with plots, and creates a new one
    if not os.path.exists("plots_festim_uq_" + plot_timestamp):
        plot_folder_name = "plots_festim_uq_" + plot_timestamp
        os.makedirs(plot_folder_name)
        # Save plots in this folder

    # Read the vertices from the results
    vertices = results.describe("x", "mean")  # Assuming 'x' is the vertex coordinate in results
    if vertices is None:
        print("No vertices found in the results. Using a simple range for plotting.")
        rs = np.linspace(
            0.0, 1.0, len(qois)
        )  # Assuming a simple range for x-axis - false, qois is number of checkpoints + 1
    else:
        rs = vertices
    # TODO mesh can be individual for each QoI, potentially each simulation, so read it from the results

    # Read runs from database to a list
    if runs_info is not None:
        print(f"Type of runs_info: {type(runs_info)}")
        if isinstance(runs_info, str):
            print(f"Loading runs info from an EasyVVUQ campaign under path: {runs_info}")
            campaign = uq.Campaign(dp_path=runs_info)
            runs_info = list(campaign.campaign_db.runs())
        else:
            print(f"Using provided runs info: {runs_info}")
            runs_info = list(runs_info)

    concentration_qois = [q for q in qois[1:] if _parse_transient_time_label(q) is not None or q == "t=steady"]
    flux_qois = [q for q in qois[1:] if _parse_flux_time_label(q) is not None]

    # Plotting statistics of concentration QoIs as a function of radius
    uqplotter = UQPlotter()
    if concentration_qois:
        uqplotter.plot_stats_vs_r(results, concentration_qois, plot_folder_name, plot_timestamp, rs=rs, runs_info=runs_info)

    # Bespoke plotting of uncertainty and Sobol indices as a function of time
    # for transient QoIs (e.g. t=...s columns).
    transient_qois = [q for q in concentration_qois if _parse_transient_time_label(q) is not None]
    if transient_qois:
        uqplotter.plot_stats_vs_t(
            results,
            distributions,
            transient_qois,
            plot_folder_name,
            plot_timestamp,
            rs=rs,
        )

    if flux_qois:
        _plot_flux_uq_vs_time(results, distributions, flux_qois, plot_folder_name, plot_timestamp)

    if config is not None:
        _plot_cj1959_uq_dashboard_2x2(results, qois[1:], rs, config, plot_folder_name)

    # Save statistics log file
    save_statistics_log(results, concentration_qois + flux_qois, plot_folder_name, plot_timestamp)

    print(f"Plots saved in folder: {plot_folder_name}")
    return 0


def define_parameter_uncertainty(config, CoV=None, distribution=None):
    """
    Define the uncertain parameters and their distributions for the FESTIM model and EasyVVUQ uncertainty propagations.
    """

    print(f"Defining uncertain parameters with CoV={CoV} and distributions={distribution}")

    # # Trial 1) Vary parameter of Arrhenius law
    # parameters_used = ['D_0', 'E_D', 'T_0',]
    # # - add BC and Sources
    # parameters_used += ['source_concentration_value', 'right_bc_concentration_value']

    # Trial 2) Vary parameters for thermal conduction
    # parameters_used = ['thermal_conductivity']

    def _nested(node, *keys):
        cur = node
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
            if cur is None:
                return None
        return cur

    # Resolve the material config using material_id (not list index).
    materials = config.get("materials", []) or []
    domain0 = (config.get("geometry", {}).get("domains", [{}]) or [{}])[0]
    material_id = domain0.get("material", None)
    material_cfg = next((m for m in materials if m.get("material_id", None) == material_id), materials[0] if materials else {})

    # Candidate uncertain parameters. Keep only those fully defined in YAML.
    parameter_specs = {
        "D_0": {
            "mean": _nested(material_cfg, "D_0", "mean"),
            "relative_stdev": _nested(material_cfg, "D_0", "relative_stdev"),
            "pdf": _nested(material_cfg, "D_0", "pdf") or "uniform",
        },
        "kappa": {
            "mean": _nested(material_cfg, "thermal_conductivity", "mean"),
            "relative_stdev": _nested(material_cfg, "thermal_conductivity", "relative_stdev"),
            "pdf": _nested(material_cfg, "thermal_conductivity", "pdf") or "uniform",
        },
        "G": {
            "mean": _nested(config, "source_terms", "concentration", "value", "mean"),
            "relative_stdev": _nested(config, "source_terms", "concentration", "value", "relative_stdev"),
            "pdf": _nested(config, "source_terms", "concentration", "value", "pdf") or "uniform",
        },
        "Q": {
            "mean": _nested(config, "source_terms", "heat", "value", "mean"),
            "relative_stdev": _nested(config, "source_terms", "heat", "value", "relative_stdev"),
            "pdf": _nested(config, "source_terms", "heat", "value", "pdf") or "uniform",
        },
        "E_kr": {
            "mean": _nested(config, "boundary_conditions", "concentration", "right", "E_kr", "mean"),
            "relative_stdev": _nested(config, "boundary_conditions", "concentration", "right", "E_kr", "relative_stdev"),
            "pdf": _nested(config, "boundary_conditions", "concentration", "right", "E_kr", "pdf") or "uniform",
        },
        "h_conv": {
            "mean": _nested(config, "boundary_conditions", "temperature", "right", "h_conv", "mean"),
            "relative_stdev": _nested(config, "boundary_conditions", "temperature", "right", "h_conv", "relative_stdev"),
            "pdf": _nested(config, "boundary_conditions", "temperature", "right", "h_conv", "pdf") or "uniform",
        },
    }

    if CoV is not None and (CoV < 0.0 or CoV > 1.0):
        raise ValueError("Coefficient of variation (CoV) must be in the range [0, 1].")

    parameters_used = [
        name
        for name, spec in parameter_specs.items()
        if spec["mean"] is not None and ((CoV is not None) or (spec["relative_stdev"] is not None))
    ]

    if not parameters_used:
        raise ValueError("No uncertain parameters could be parsed from config for UQ campaign.")

    means = {name: parameter_specs[name]["mean"] for name in parameters_used}
    relative_stds = {
        name: (CoV if CoV is not None else parameter_specs[name]["relative_stdev"])
        for name in parameters_used
    }
    distributions = {
        name: (distribution if distribution is not None else parameter_specs[name]["pdf"])
        for name in parameters_used
    }

    logger.debug(f" >>> Parsed uncertain parameters: {parameters_used}")
    logger.debug(f" >>> Mean values for uncertain parameters: {means}")
    logger.debug(f" >>> Relative STDs for uncertain parameters: {relative_stds}")

    logger.debug(f" >>> Distributions for uncertain parameters: {distributions}")

    # Define coefficients to recalculate distribution defining parameters - absolute bounds of the (uniform) distribution is a function of the mean and CoV
    # - for COV of U[a,b]: STD = (b-a)/sqrt(12) , meaning it should be a=mean*(1-sqrt(3)CoV), b=mean*(1+sqrt(3)CoV)
    expansion_factor_lookup = {
        "normal": 1.0,
        "uniform": np.sqrt(3),
    }

    distribution_lookup = {
        "normal": cp.Normal,
        "uniform": cp.Uniform,
        "lognormal": cp.LogNormal,
        "beta": cp.Beta,
        "gamma": cp.Gamma,
        "exponential": cp.Exponential,
    }

    # Create the distributions for the parameters
    parameters_distributions = {
        name: distribution_lookup[distributions[name]](
            (
                (means[name] * (1.0 - expansion_factor_lookup[distributions[name]] * relative_stds[name]))
                if distributions[name] == "uniform"
                else (means[name]) if distributions[name] == "normal" else means[name]
            ),  # lower bound for uniform, mean for normal
            (
                (means[name] * (1.0 + expansion_factor_lookup[distributions[name]] * relative_stds[name]))
                if distributions[name] == "uniform"
                else (
                    (means[name] * relative_stds[name])
                    if distributions[name] == "normal"
                    else means[name] * relative_stds[name]
                )
            ),  # upper bound for uniform, mean for normal
        )
        for name in parameters_used
    }
    # TODO: tackle not implemented distributions, e.g. lognormal, beta, gamma, exponential
    # TODO: tackle different distribution specifications, e.g. normal with mean and std, uniform with bounds, etc.

    return parameters_distributions


def define_festim_model_parameters(config=None):
    """
    Define the FESTIM model parameters and their default values.
    """

    # Define the model input parameters
    parameters = {
        "D_0": {
            "type": "float",
            "default": 1.0e-7,
        },
        "kappa": {
            "type": "float",
            "default": 1.0,
        },
        "G": {
            "type": "float",
            "default": 1.0e6,
        },
        "Q": {
            "type": "float",
            "default": 1000.0,
        },
        "E_kr": {
            "type": "float",
            "default": 1.0,
        },
        "h_conv": {
            "type": "float",
            "default": 1000.0,
        },
        # "E_D": {"type": "float", "default": 0.1,},
        # "T_0": {"type": "float", "default": 300.0,},
        # "source_concentration_value": {"type": "float", "default": 1.0e20,},
        # "left_bc_concentration_value": {"type": "float", "default": 0.0},  # Boundary condition value: better to specify centre at r=0.0
        # "right_bc_concentration_value": {"type": "float", "default": 1.0e15},  # Boundary condition value at the right (outer) surface of the sample
    }

    # Define output parameters / QoIs from config where possible.
    qois = ["x"]
    model_params = (config or {}).get("model_parameters", {}) if isinstance(config, dict) else {}
    transient = bool(model_params.get("transient", False))
    milestone_times = ((config or {}).get("simulation", {}) or {}).get("milestone_times", []) if isinstance(config, dict) else []
    if transient and milestone_times:
        qois.extend([f"t={float(t):.2e}s" for t in milestone_times])
        qois.extend([f"flux_t={float(t):.2e}s" for t in milestone_times])
    else:
        qois.append("t=steady")

    # print(f"Model parameters defined: {parameters}")
    # print(f"QoIs defined: {qois}")
    return parameters, qois


def prepare_execution_command():
    """
    Prepare the execution command for the EasyVVUQ campaign.
    """

    # Get the Python executable and script path - validate and setup environment
    python_exe, script_path = validate_execution_setup()

    # Use the filename that the encoder creates (config.uq.yaml)
    config_suffix = f" --config config.yaml --campaign-mode "

    # - Assuring correct environment:
    # - - running activation command
    env_name = "festim2-env"  # Name of the conda environment to activate
    env_prefix = ""  # f"conda activate {env_name} && "

    # - option 1) run via calling the python3 command - be sure that the correct environment is used
    exec_command_line = f"{env_prefix} {python_exe} {script_path} {config_suffix}"
    # - option 2) run using shebang in the python script (as an executable)

    print(f"Execution command line: {exec_command_line}")

    # Execute locally. Passing stdout/stderr filenames triggers a known
    # EasyVVUQ execute_local bug in some versions (missing close symbol).
    execute = ExecuteLocal(exec_command_line)

    return execute


def prepare_uq_campaign(config, config_file, fixed_params=None, uq_params=None):
    """
    Prepare the uncertainty quantification (UQ) campaign by creating necessary steps: set-up, parameter definitions, encoders, decoders, and actions.
    """

    # Define the model input and output parameters
    parameters, qois = define_festim_model_parameters(config=config)

    # TODO rearange FESTIM data output, figure out how to specify multiple quantities at different times, all vs a coordinate

    # Set up necessary elements for the EasyVVUQ campaign

    # TODO: provide parameters with fixed values for the entire campaign -  could be done: (1) in the file at the harddrive, (2) in the YAML object read by the encoder, (3) as an UQ parameter that has to be set to a new default value (a CopyEncoder + MultiEncoder can be applied for this...)

    # Option 1) Modify the parameters in a hard-drive file - skipping
    # Option 2) Before running the campaign, substitute the parameters in the template YAML file

    # print(f" >> Preparing the encoder with parameters: {fixed_params}")

    # Create an Encoder object

    # Option 1): Simple YAML Encoder
    # encoder = YAMLEncoder(
    #     template_fname="festim_yaml.template",
    #     target_filename="config.yaml",
    #     delimiter="$"
    # )

    # Option 2): Advanced YAML Encoder
    encoder = AdvancedYAMLEncoder(
        template_fname=config_file,
        target_filename="config.yaml",
        parameter_map={  # TODO: store the YAML schema as a separate file; ideally, read from an existing YAML config file
            # ATTENTION: for lists, pattern will always choose the first element, e.g. for materials, domains
            "D_0": "materials.D_0.mean",
            "kappa": "materials.thermal_conductivity.mean",
            "G": "source_terms.concentration.value.mean",
            "Q": "source_terms.heat.value.mean",
            "E_kr": "boundary_conditions.concentration.right.E_kr.mean",
            "h_conv": "boundary_conditions.temperature.right.h_conv.mean",
            # "E_D": "materials.E_D.mean",
            # "T": "initial_conditions.temperature.value.mean",
            # "source_concentration_value": "source_terms.concentration.source_value",
            # "left_bc_concentration_value": "boundary_conditions.concentration.left.value",
            # "right_bc_concentration_value": "boundary_conditions.concentration.right.value",
            "length": "geometry.domains.length",
            "tritium_transport_absolute_tolerance": "simulation.tolerances.absolute_tolerance.tritium_transport",
        },
        type_conversions={
            "D_0": float,
            "kappa": float,
            "G": float,
            "Q": float,
            "E_kr": float,
            "h_conv": float,
            "E_D": float,
            "T": float,
            "source_concentration_value": float,
            "left_bc_concentration_value": float,
            "right_bc_concentration_value": float,
            "length": float,
            "tritium_transport_absolute_tolerance": float,
        },
        fixed_parameters=fixed_params,  # Pass the dictionary of parameters to be fixed to the encoder
    )

    # Option 3): Use built-in EasyVVUQ encoder
    # encoder = uq.encoders.JinjaEncoder(
    #     template_fname="festim.template",
    #     target_filename="config.yaml"
    # )

    # TODO modify the YAML more arbitrartly - pass a parameter value from this function e.g. for the sample length scan

    # print(f"Using encoder: {encoder.__class__.__name__}")
    print(f"Encoder prepared: {encoder}")

    # Create a decoder object
    # The decoder will read the results from the output file and extract the quantities of interest (QoIs)

    # TODO change the output and decoder to YAML (for UQ derived quantities) or other format (?)

    output_dir = str(config.get("simulation", {}).get("output_directory", "results/test")).rstrip("/")
    concentration_qois = [q for q in qois if q == "x" or _parse_transient_time_label(q) is not None or q == "t=steady"]
    flux_qois = [q for q in qois if _parse_flux_time_label(q) is not None]
    decoder = MultiOutputDecoder(
        profile_filename=f"{output_dir}/results_tritium_concentration.txt",
        flux_filename=f"{output_dir}/total_hydrogen_flux_rmax.txt",
        concentration_qois=concentration_qois,
        flux_qois=flux_qois,
    )

    print(f"Decoder prepared: {decoder}")

    # Prepare execution command action
    # This command will be used to run the FESTIM model with the generated configuration file
    execute = prepare_execution_command()

    # Set up actions for the campaign
    actions = Actions(
        CreateRunDirectory("run_dir"),
        Encode(encoder),
        execute,
        Decode(decoder),
    )

    print(f"Sequence of actions prepared: {actions}")

    # Define the campaign parameters and actions
    # This includes the model parameters, quantities of interest (QoIs), and the actions to be performed during the campaign
    # Note: The campaign name can be customized as needed
    # Define the campaign
    campaign_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Campaign timestamp: {campaign_timestamp}")

    campaign = uq.Campaign(
        name=f"festim_campaign_{campaign_timestamp}_",
        params=parameters,
        # qois=qois,
        actions=actions,
    )

    # Define uncertain parameters distributions
    distributions = define_parameter_uncertainty(config)
    print(f"Uncertain parameters distributions defined: {distributions}")

    # Define sampling method and create a sampler for the campaign
    # This sampler will generate samples based on the defined distributions

    if uq_params is not None:
        if "uq_scheme" in uq_params:
            if uq_params["uq_scheme"] == "pce":
                # Option A) Polynomial Chaos Expansion (PCE) sampler
                # - define polynomial order for PC expansion
                if "p_order" in uq_params:
                    p_order = uq_params["p_order"]
                else:
                    p_order = 3  # default: maximal order 3

                # Sparse quadrature reduces the number of model evaluations
                # significantly for high-dimensional problems while retaining
                # accuracy for total-degree truncated polynomial bases.
                sparse = uq_params.get("sparse", True)

                logger.debug(f"Using UQ scheme: {uq_params['uq_scheme']} with polynomial order: {p_order}, sparse: {sparse}")

                sampler = uq.sampling.PCESampler(
                    vary=distributions,
                    polynomial_order=p_order,
                    sparse=sparse,
                )

            elif uq_params["uq_scheme"] == "qmc":
                # Option B) quasi-Monte Carlo sampler
                # - define number of samples
                if "n_samples" in uq_params:
                    n_samples = uq_params["n_samples"]
                else:
                    n_samples = 128  # Default number of samples if not specified

                logger.debug(f"Using UQ scheme: {uq_params['uq_scheme']} with number of samples: {n_samples}")

                sampler = uq.sampling.QMCSampler(
                    vary=distributions,
                    n_mc_samples=n_samples,  # Number of samples to generate
                )

            else:
                raise ValueError(
                    f"Unsupported UQ scheme: {uq_params['uq_scheme']}. Supported schemes are 'pce' and 'qmc'."
                )
        else:
            raise ValueError(
                "UQ scheme not specified in uq_params. Please provide 'uq_scheme' as either 'pce' or 'qmc'."
            )
    else:
        # Default to sparse PCE sampler with polynomial order 3
        print("No UQ parameters provided, defaulting to sparse PCE sampler with polynomial order 3.")
        sampler = uq.sampling.PCESampler(
            vary=distributions,
            polynomial_order=3,
            sparse=True,
        )

    campaign.set_sampler(sampler)
    print(f"Sampler prepared and set for the campaign: {sampler}")

    print(f"Campaign and its elements are prepared!")
    return campaign, qois, distributions, campaign_timestamp, sampler


def run_uq_campaign(campaign, resource_pool=None, local=True):
    """
    Run the UQ campaign.

    Parameters
    ----------
    campaign : easyvvuq.Campaign
        Prepared EasyVVUQ campaign object.
    resource_pool : optional
        A QCGPJPool (or compatible) resource pool for HPC execution.
        When *None* and *local* is ``True`` (the default), the campaign is
        executed directly on the local machine without any scheduler.
    local : bool
        If ``True`` (default) and no *resource_pool* is provided, run
        sequentially on the local machine using ``campaign.execute()``.
        Set to ``False`` to fall back to the QCGPJPool scheduler.
    """
    if resource_pool is None and local:
        # Local execution — no scheduler, no pool required.
        print(" >> Running campaign locally (no HPC scheduler)…")
        campaign_results = campaign.execute()
        print(">>> Local execution completed! Collating results…")
        campaign_results.collate()
    else:
        # HPC execution via QCGPJPool (or a user-provided pool)
        if resource_pool is None:
            # Make sure the right parameters are passed to the pool: virtual environment, working directory, etc.
            template = EasyVVUQBasicTemplate()
            template_params = {
                "venv": os.environ.get("CONDA_PREFIX", sys.prefix),
            }

            # By default, run with resource pool by QCG-PJ
            resource_pool = QCGPJPool(
                template=template,
                template_params=template_params,
            )

        print(f" >> Prepared the resource pool to run the campaign: {resource_pool}")
        # Execute the campaign
        with resource_pool as pool:

            print(f">>> Running the campaign with resource pool: {pool}")

            campaign_results = campaign.execute(pool=pool)

            print(">>> Execution completed! Collating the results...")

            campaign_results.collate()

    return campaign, campaign_results


def analyse_uq_results(campaign, qois, sampler, uq_params=None, output_folder=None):
    """
    Perform analysis on the UQ results.

    Parameters
    ----------
    campaign : easyvvuq.Campaign
    qois : list of str
    sampler : easyvvuq sampler
    uq_params : dict, optional
    output_folder : str, optional
        Directory where the analysis results pickle will be saved.
        Defaults to the current working directory.
    """
    if uq_params is not None:
        if "uq_scheme" in uq_params:
            logger.debug(f"Performing analysis for UQ scheme: {uq_params['uq_scheme']}")
            if uq_params["uq_scheme"] == "pce":
                # Perform PCE analysis on the campaign results
                analysis = uq.analysis.PCEAnalysis(sampler=sampler, qoi_cols=qois)
            elif uq_params["uq_scheme"] == "qmc":
                # Perform QMC analysis on the campaign results
                analysis = uq.analysis.QMCAnalysis(sampler=sampler, qoi_cols=qois)
            else:
                raise ValueError(
                    f"Unsupported UQ scheme: {uq_params['uq_scheme']}. Supported schemes are 'pce' and 'qmc'."
                )
        else:
            print("UQ scheme not specified in uq_params. Proceeding with default analysis.")
    else:
        print("No UQ parameters provided. Proceeding with default analysis.")

    campaign.apply_analysis(analysis)

    # Get the last analysis results
    results = campaign.get_last_analysis()

    logger.debug(f"\n >>> Analysis completed. Results:\n{results}")

    # Save the analysis results to a file in output_folder (or CWD)
    result_filename_base = "analysis_results_uq_campaign.pickle"
    results_basename = add_timestamp_to_filename(result_filename_base)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        results_filename = os.path.join(output_folder, results_basename)
    else:
        results_filename = results_basename
    logger.debug(f">> Saving the campaign results into {results_filename}")
    with open(results_filename, "wb") as _fh:
        pickle.dump(results, _fh)
    print(f"✓ Analysis results saved: {results_filename}")

    # Display the results of the analysis
    for qoi in qois[1:]:
        print(f"Results for {qoi}:")
        try:
            print(results.describe(qoi))
        except Exception as exc:
            # Some EasyVVUQ/Chaospy combinations may fail to build output
            # distributions for a QoI; keep campaign flow alive and print
            # available scalar/vector statistics instead.
            logger.warning(f"Could not print full describe() for {qoi}: {exc}")
            print(f"  Full describe unavailable: {exc}")
            printed_any = False
            for stat_name in ["mean", "std", "min", "max", "1%", "10%", "median", "90%", "99%"]:
                try:
                    stat_values = results.describe(qoi, stat_name)
                except Exception:
                    continue
                if stat_values is None:
                    continue
                printed_any = True
                arr = np.asarray(stat_values)
                if arr.ndim == 0:
                    print(f"  {stat_name}: {arr.item()}")
                else:
                    print(
                        f"  {stat_name}: shape={arr.shape}, min={np.min(arr):.6e}, max={np.max(arr):.6e}, avg={np.mean(arr):.6e}"
                    )
            if not printed_any:
                print("  No fallback statistics available for this QoI.")
        print("\n")

    return results


def perform_uq_festim(config=None, fixed_params=None):
    """
    Main function to perform the UQ campaign for FESTIM.
    This function orchestrates the preparation, execution, and analysis of the UQ campaign.

    The campaign runs **locally** (no HPC scheduler) by default, using a
    **sparse PCE sampler** with polynomial order 3 and a total-degree truncated
    polynomial basis.  Both the campaign database state file and the analysis
    results pickle are written to a timestamped output folder.
    """
    # EasyVVUQ script to be executed as a function
    print(" \n ! Starting FESTIM UQ campaign !.. \n")
    print(f" time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Prepare the UQ campaign
    # This includes defining parameters, encoders, decoders, and actions

    # Check if config exists
    if config is None:

        print("No config file provided, reading from arguments")

        # Read the configuration file from the command line argument or use a default one
        parser = argparse.ArgumentParser(description="Run FESTIM model with YAML configuration")

        parser.add_argument(
            "--config", "-c", default="config.yaml", help="Path to YAML configuration file (default: config.yaml)"
        )
        parser.add_argument(
            "--p-order",
            type=int,
            default=3,
            help="PCE polynomial order (default: 3)",
        )
        parser.add_argument(
            "--full-tensor",
            action="store_true",
            help="Use full tensor-product quadrature for PCE (default: sparse Smolyak)",
        )

        args = parser.parse_args()
        print(f"> Using arguments file: {args.config}")

        # Load configuration from YAML file
        config = load_config(args.config)

        print(f" > Loaded configuration from: {args.config}")

    if config is None:
        print("No config file provided, quitting...")
        return

    logger.debug(f" >> Passing parameters fixed to the campaign: {fixed_params}")

    # Define UQ parameters — sparse PCE, polynomial order configurable via CLI
    uq_params = {
        "uq_scheme": "pce",   # 'pce' or 'qmc'
        "p_order": int(args.p_order),  # maximal polynomial order (total-degree truncation)
        "sparse": not bool(args.full_tensor),  # sparse Smolyak by default; full tensor if requested
        "n_samples": 8,       # only used for QMC fallback
    }

    campaign, qois, distributions, campaign_timestamp, sampler = prepare_uq_campaign(
        config, config_file=args.config, fixed_params=fixed_params, uq_params=uq_params
    )

    print("\n===== PRE-EXECUTION SUMMARY =====")
    print(f"Config file: {args.config}")
    print(f"Campaign timestamp: {campaign_timestamp}")
    print(f"UQ scheme: {uq_params.get('uq_scheme', 'unknown')}")
    print(f"PCE order: {uq_params.get('p_order', 'n/a')}")
    print(f"Sparse grid: {uq_params.get('sparse', 'n/a')}")
    print(f"Uncertain parameters: {list(distributions.keys())}")
    print(f"QoIs: {qois}")
    print("Execution mode: local (ExecuteLocal)")
    print("Per-run logs: stdout.txt, stderr.txt")
    print("===============================\n")

    # Create a dedicated output folder for this campaign run
    output_folder = f"festim_uq_results_{campaign_timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    print(f" >> Output folder: {output_folder}")

    # Run the campaign locally (no HPC scheduler)
    print(f" >> Now running the UQ campaign locally…")
    campaign, campaign_results = run_uq_campaign(campaign, local=True)
    logger.debug(f" >> Campaign run completed. Campaign results: {campaign_results}")

    # Persist the campaign database state to a portable JSON file
    campaign_state_filename = os.path.join(output_folder, f"campaign_state_{campaign_timestamp}.json")
    try:
        campaign.save_state(campaign_state_filename)
        print(f"✓ Campaign state saved: {campaign_state_filename}")
    except Exception as _e:
        logger.warning(f"Could not save campaign state via save_state(): {_e}")
        # Some EasyVVUQ versions have fragile DB dump/close paths.
        # Keep campaign execution alive and continue to analysis.
        print("i Continuing without campaign state dump due to save_state fallback issue.")

    # Perform the analysis — saves analysis_results pickle to output_folder
    results = analyse_uq_results(campaign, qois, sampler, uq_params=uq_params, output_folder=output_folder)
    logger.debug(f" >> Analysis of results completed. Results: {results}")

    # Save campaign configuration and parameter distributions to output_folder
    config_filename = os.path.join(output_folder, add_timestamp_to_filename("uq_campaign_config.pickle", campaign_timestamp))
    with open(config_filename, "wb") as _fh:
        pickle.dump(config, _fh)
    print(f"✓ Campaign configuration saved: {config_filename}")

    # Get the individual results from the campaign
    runs = campaign.campaign_db.runs()  # return an iterator over runs in the campaign

    # Visualize the results
    plot_folder = os.path.join(output_folder, "plots")
    os.makedirs(plot_folder, exist_ok=True)
    visualisation_of_results(
        results,
        distributions,
        qois,
        plot_folder,
        plot_timestamp=campaign_timestamp,
        runs_info=runs,
        config=config,
    )

    # Generate interactive dashboard from individual campaign runs.
    _generate_uq_dashboard_for_campaign(campaign_timestamp, output_folder)

    print(f"\nFESTIM UQ campaign completed successfully!")
    print(f"✓ All outputs in: {output_folder}")
    return 0


if __name__ == "__main__":
    """
    Main entry point for the script.
    This will execute the UQ campaign when the script is run directly.
    """
    try:
        # Run with parameters from the provided YAML config by default.
        perform_uq_festim()

    except Exception as e:
        print(f"An error occurred during the UQ campaign: {e}")
        traceback.print_exc()
        sys.exit(1)
