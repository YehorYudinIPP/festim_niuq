#!/usr/bin/env python3
"""
Reanalyse an existing EasyVVUQ FESTIM campaign with corrected transient QoI columns.

This script is designed for campaigns that already executed all FESTIM runs but
failed at decode/collation due to mismatched decoder columns (e.g. expecting
`t=steady` while output contains transient `t=...s` headers).

Workflow
--------
1. Read one existing run `config.yaml` and build the expected QoI columns from
   `simulation.milestone_times`.
2. Try EasyVVUQ restart/recollate path on the existing campaign DB:
   - reopen campaign with `change_to_state=True`
   - replace app actions with decode-only action using corrected decoder
   - call `campaign.recollate()`
3. If restart recollation is unavailable/broken in current EasyVVUQ version,
   fall back to reconstruction from existing run folders only (no reruns):
   - create a fresh campaign DB
   - import existing run inputs/outputs via `add_external_runs`
   - reuse the original sampler from the old campaign DB
4. Perform analysis and plotting, and write outputs to a dedicated folder.

Usage
-----
python -m festim_niuq.uq.reanalyse_existing_campaign \
    --campaign-dir festim_campaign_20260527_123624_zvlh933n
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import io
import json
import os
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
import numpy as np
import matplotlib
import math

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import easyvvuq as uq
from easyvvuq.actions import Actions, Decode

from .easyvvuq_festim import define_parameter_uncertainty
from .interactive_run_viewer import discover_run_dirs, read_run_data, generate_html
from .util.Decoder import MultiOutputDecoder
from .util.utils import add_timestamp_to_filename


TIME_PLOT_MIN_S = 1.0


class YAMLParamDecoder:
    """Decoder for campaign input `config.yaml` to recover uncertain parameters."""

    def __init__(self, defaults: dict[str, Any], target_filename: str = "config.yaml"):
        self.defaults = defaults
        self.target_filename = target_filename

    @staticmethod
    def _nested(node: dict[str, Any], *keys: str) -> Any:
        cur: Any = node
        for key in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(key)
            if cur is None:
                return None
        return cur

    def parse_sim_output(self, run_info: dict[str, Any] | None = None) -> dict[str, Any]:
        run_info = run_info or {}
        run_dir = Path(run_info["run_dir"])
        cfg_path = run_dir / self.target_filename
        with cfg_path.open("r") as fh:
            cfg = yaml.safe_load(fh)

        materials = cfg.get("materials", []) or []
        domain0 = (cfg.get("geometry", {}).get("domains", [{}]) or [{}])[0]
        material_id = domain0.get("material", None)
        material_cfg = next((m for m in materials if m.get("material_id", None) == material_id), materials[0] if materials else {})

        vals = {
            "D_0": self._nested(material_cfg, "D_0", "mean"),
            "kappa": self._nested(material_cfg, "thermal_conductivity", "mean"),
            "G": self._nested(cfg, "source_terms", "concentration", "value", "mean"),
            "Q": self._nested(cfg, "source_terms", "heat", "value", "mean"),
            "E_kr": self._nested(cfg, "boundary_conditions", "concentration", "right", "E_kr", "mean"),
            "h_conv": self._nested(cfg, "boundary_conditions", "temperature", "right", "h_conv", "mean"),
        }

        for key, default_value in self.defaults.items():
            if vals.get(key) is None:
                vals[key] = default_value

        return vals


def _load_sampler_from_db(campaign_db: Path):
    con = sqlite3.connect(str(campaign_db))
    sampler_blob = con.execute("SELECT sampler FROM sampler LIMIT 1").fetchone()[0]
    con.close()
    return pickle.loads(codecs.decode(sampler_blob.encode(), "base64"))


def _load_app_params_defaults(campaign_db: Path) -> tuple[str, dict[str, Any]]:
    con = sqlite3.connect(str(campaign_db))
    row = con.execute("SELECT name, params FROM app LIMIT 1").fetchone()
    con.close()

    app_name = row[0]
    params_json = json.loads(row[1])
    defaults = {k: v.get("default", None) for k, v in params_json.items()}
    return app_name, params_json, defaults


def _find_run_dirs(campaign_dir: Path) -> list[Path]:
    run_dirs = sorted(p.parent for p in campaign_dir.glob("runs/**/run_*/config.yaml"))
    return run_dirs


def _run_analysis(campaign, sampler, qois: list[str], output_folder: Path):
    """Run PCE analysis robustly and save results pickle."""
    output_folder.mkdir(parents=True, exist_ok=True)

    analysis = uq.analysis.PCEAnalysis(
        sampler=sampler,
        qoi_cols=qois,
        CorrelationMatrices=False,
        OutputDistributions=False,
    )
    # EasyVVUQ's PCE derivative post-processing may print non-fatal tracebacks
    # for some Chaospy versions; capture stderr to keep terminal/log output clean.
    analysis_stderr = io.StringIO()
    with contextlib.redirect_stderr(analysis_stderr):
        campaign.apply_analysis(analysis)

    stderr_text = analysis_stderr.getvalue().strip()
    if stderr_text:
        stderr_log = output_folder / add_timestamp_to_filename("analysis_stderr.log")
        with stderr_log.open("w") as fh:
            fh.write(stderr_text + "\n")
        print(f"i Analysis warnings captured in: {stderr_log}")

    results = campaign.get_last_analysis()

    results_name = add_timestamp_to_filename("analysis_results_uq_campaign.pickle")
    results_path = output_folder / results_name
    with results_path.open("wb") as fh:
        pickle.dump(results, fh)
    print(f"✓ Analysis results saved: {results_path}")

    return results, results_path


def _first_existing_x_from_runs(run_dirs: list[Path]) -> np.ndarray | None:
    """Read radial coordinates from FESTIM concentration outputs (CSV first)."""
    candidate_patterns = (
        "results/**/results_tritium_concentration.csv",
        "results/**/results_tritium_concentration.txt",
    )

    for run_dir in run_dirs:
        for pattern in candidate_patterns:
            for out in sorted(run_dir.glob(pattern)):
                if not out.exists():
                    continue
                try:
                    data = np.genfromtxt(out, delimiter=",", skip_header=1)
                except Exception:
                    continue
                if data.ndim != 2 or data.shape[1] < 1:
                    continue
                x_vals = np.asarray(data[:, 0], dtype=float)
                if x_vals.size == 0 or not np.all(np.isfinite(x_vals)):
                    continue
                return x_vals
    return None


def _plot_reanalysis(results, qois: list[str], run_dirs: list[Path], plot_dir: Path):
    """Create robust mean/std + quantile plots per QoI over radius/index."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Prefer true mesh coordinates exported by FESTIM outputs.
    x_vals = _first_existing_x_from_runs(run_dirs)
    if x_vals is None:
        try:
            x_vals = np.asarray(results.describe("x", "mean"))
        except Exception:
            x_vals = None

    concentration_qois = [q for q in qois if _parse_transient_time(q) is not None]
    flux_qois = [q for q in qois if _parse_flux_time(q) is not None]

    for qoi in concentration_qois:
        try:
            mean = np.asarray(results.describe(qoi, "mean"))
        except Exception as exc:
            print(f"Skipping plot for {qoi}: mean unavailable ({exc})")
            continue

        try:
            std = np.asarray(results.describe(qoi, "std"))
            if std.shape != mean.shape:
                std = np.zeros_like(mean)
        except Exception:
            std = np.zeros_like(mean)

        try:
            p01 = np.asarray(results.describe(qoi, "1%"))
            if p01.shape != mean.shape:
                p01 = np.full_like(mean, np.nan, dtype=float)
        except Exception:
            p01 = np.full_like(mean, np.nan, dtype=float)

        try:
            p99 = np.asarray(results.describe(qoi, "99%"))
            if p99.shape != mean.shape:
                p99 = np.full_like(mean, np.nan, dtype=float)
        except Exception:
            p99 = np.full_like(mean, np.nan, dtype=float)

        if x_vals is None or len(x_vals) != len(mean):
            x = np.arange(len(mean))
            xlabel = "Index"
        else:
            x = x_vals
            xlabel = "x"
            if x.ndim == 1 and x.size > 1 and np.any(np.diff(x) < 0.0):
                order = np.argsort(x)
                x = x[order]
                mean = mean[order]
                std = std[order]
                p01 = p01[order]
                p99 = p99[order]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, mean, label="mean")
        # Always render the uncertainty band so plot structure is consistent
        # across QoIs, even when the current std is zero.
        ax.fill_between(x, mean - std, mean + std, alpha=0.2, label="+/- STD")
        ax.plot(x, mean - std, alpha=0.35, linewidth=0.8, linestyle=":", label="_nolegend_")
        ax.plot(x, mean + std, alpha=0.35, linewidth=0.8, linestyle=":", label="_nolegend_")
        if np.all(np.isfinite(p01)) and np.all(np.isfinite(p99)):
            ax.fill_between(x, p01, p99, alpha=0.12, color="tab:green", label="1%-99%")
            ax.plot(x, p01, alpha=0.35, linewidth=0.8, linestyle="--", color="tab:green", label="_nolegend_")
            ax.plot(x, p99, alpha=0.35, linewidth=0.8, linestyle="--", color="tab:green", label="_nolegend_")
        ax.set_title(f"Reanalysis: {qoi}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(qoi)
        ax.grid(True)
        ax.legend(loc="best")

        out = plot_dir / f"reanalysis_{qoi.replace('=', '_').replace('+', 'p').replace('-', 'm')}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if flux_qois:
        entries: list[tuple[float, str]] = []
        for q in flux_qois:
            t = _parse_flux_time(q)
            if t is not None:
                entries.append((t, q))
        entries.sort(key=lambda it: it[0])

        t_arr = []
        mean_arr = []
        std_arr = []
        p01_arr = []
        p99_arr = []

        def _scalar(v):
            arr = np.asarray(v, dtype=float)
            return float(arr.reshape(-1)[0]) if arr.size > 0 else np.nan

        for t, q in entries:
            try:
                mean_v = _scalar(results.describe(q, "mean"))
            except Exception:
                continue

            try:
                std_v = _scalar(results.describe(q, "std"))
            except Exception:
                std_v = 0.0

            try:
                p01_v = _scalar(results.describe(q, "1%"))
            except Exception:
                p01_v = np.nan

            try:
                p99_v = _scalar(results.describe(q, "99%"))
            except Exception:
                p99_v = np.nan

            t_arr.append(float(t))
            mean_arr.append(mean_v)
            std_arr.append(std_v)
            p01_arr.append(p01_v)
            p99_arr.append(p99_v)

        if t_arr:
            t_vals = np.asarray(t_arr, dtype=float)
            m_vals = np.asarray(mean_arr, dtype=float)
            s_vals = np.asarray(std_arr, dtype=float)
            q01_vals = np.asarray(p01_arr, dtype=float)
            q99_vals = np.asarray(p99_arr, dtype=float)

            mask = t_vals >= TIME_PLOT_MIN_S
            t_vals = t_vals[mask]
            m_vals = m_vals[mask]
            s_vals = s_vals[mask]
            q01_vals = q01_vals[mask]
            q99_vals = q99_vals[mask]
            if t_vals.size > 0:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(t_vals, m_vals, marker="o", label="flux mean")
                ax.fill_between(t_vals, m_vals - s_vals, m_vals + s_vals, alpha=0.2, label="+/- STD")
                if np.all(np.isfinite(q01_vals)) and np.all(np.isfinite(q99_vals)):
                    ax.fill_between(t_vals, q01_vals, q99_vals, alpha=0.12, color="tab:green", label="1%-99%")
                ax.set_title("Reanalysis: outer-surface flux vs time")
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Total hydrogen flux at R=R_max")
                ax.grid(True)
                if np.all(t_vals > 0):
                    ax.set_xscale("log")
                ax.legend(loc="best")
                fig.savefig(plot_dir / "reanalysis_flux_rmax_vs_time.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

    print(f"✓ Plots saved under: {plot_dir}")


def _generate_uq_dashboard(runs_root: Path, output_html: Path):
    """Generate interactive UQ dashboard HTML from campaign run directories."""
    if not runs_root.exists():
        print(f"Skipping dashboard generation: runs root not found: {runs_root}")
        return

    found = discover_run_dirs(str(runs_root))
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

    output_html.parent.mkdir(parents=True, exist_ok=True)
    generate_html(run_data_list, str(output_html))
    print(f"✓ UQ dashboard saved: {output_html}")


def _parse_transient_time(qoi: str) -> float | None:
    """Return transient time value for labels like 't=1.00e-01s'."""
    if not isinstance(qoi, str):
        return None
    if not (qoi.startswith("t=") and qoi.endswith("s")):
        return None
    try:
        return float(qoi.split("=", 1)[1].rstrip("s"))
    except Exception:
        return None


def _parse_flux_time(qoi: str) -> float | None:
    """Return transient flux time value for labels like 'flux_t=1.00e-01s'."""
    if not isinstance(qoi, str):
        return None
    if not (qoi.startswith("flux_t=") and qoi.endswith("s")):
        return None
    try:
        return float(qoi.split("=", 1)[1].rstrip("s"))
    except Exception:
        return None


def _transient_qois_sorted(qois: list[str]) -> list[tuple[float, str]]:
    """Collect transient QoI labels sorted by time."""
    transient: list[tuple[float, str]] = []
    for q in qois:
        t = _parse_transient_time(q)
        if t is not None:
            transient.append((t, q))
    transient.sort(key=lambda it: it[0])
    return transient


def _filter_time_entries(entries: list[tuple[float, str]], min_time: float = TIME_PLOT_MIN_S) -> list[tuple[float, str]]:
    """Filter (time, qoi) entries to times >= min_time."""
    return [(t, q) for (t, q) in entries if float(t) >= float(min_time)]


def _plot_cj1959_sobol_bar_summary(results, qois: list[str], x_vals: np.ndarray, plot_dir: Path):
    """Plot average first-order Sobol indices for concentration and flux.

    Concentration Sobols are averaged over all time slices (t >= TIME_PLOT_MIN_S)
    and then across the spatial dimension. Flux Sobols are averaged over the
    available flux time slices (t >= TIME_PLOT_MIN_S).
    """
    transient_qois = _filter_time_entries(_transient_qois_sorted(qois))
    flux_qois = _filter_time_entries([(t, q) for q in qois if (t := _parse_flux_time(q)) is not None])

    if not transient_qois and not flux_qois:
        print("Skipping Sobol bar summary: no valid transient concentration or flux QoIs found.")
        return

    param_names: list[str] = []
    for _, q in transient_qois + flux_qois:
        try:
            s1 = results.sobols_first(q)
        except Exception:
            s1 = None
        if isinstance(s1, dict):
            for p in s1.keys():
                if p not in param_names:
                    param_names.append(p)

    if not param_names:
        print("Skipping Sobol bar summary: no Sobol parameter names available.")
        return

    x = np.asarray(x_vals, dtype=float).reshape(-1)
    if x.size > 1 and np.any(np.diff(x) < 0.0):
        x_order = np.argsort(x)
    else:
        x_order = np.arange(x.size, dtype=int)

    conc_s1_summary: dict[str, list[float]] = {p: [] for p in param_names}
    conc_st_summary: dict[str, list[float]] = {p: [] for p in param_names}
    flux_s1_summary: dict[str, list[float]] = {p: [] for p in param_names}
    flux_st_summary: dict[str, list[float]] = {p: [] for p in param_names}

    # Average concentration Sobols over time and space.
    for _, q in transient_qois:
        try:
            s1 = results.sobols_first(q)
        except Exception:
            s1 = None
        try:
            st = results.sobols_total(q)
        except Exception:
            st = None
        if not isinstance(s1, dict):
            s1 = {}
        if not isinstance(st, dict):
            st = {}

        for p in param_names:
            vals_s1 = s1.get(p, None)
            if vals_s1 is not None:
                arr = np.asarray(vals_s1, dtype=float).reshape(-1)
                if arr.size > 0:
                    if arr.size == x.size:
                        arr = arr[x_order]
                    conc_s1_summary[p].append(float(np.mean(arr)))

            vals_st = st.get(p, None)
            if vals_st is not None:
                arr = np.asarray(vals_st, dtype=float).reshape(-1)
                if arr.size > 0:
                    if arr.size == x.size:
                        arr = arr[x_order]
                    conc_st_summary[p].append(float(np.mean(arr)))

    # Average flux Sobols over available flux time slices.
    for _, q in flux_qois:
        try:
            s1 = results.sobols_first(q)
        except Exception:
            s1 = None
        try:
            st = results.sobols_total(q)
        except Exception:
            st = None
        if not isinstance(s1, dict):
            s1 = {}
        if not isinstance(st, dict):
            st = {}

        for p in param_names:
            vals_s1 = s1.get(p, None)
            if vals_s1 is not None:
                arr = np.asarray(vals_s1, dtype=float).reshape(-1)
                if arr.size > 0:
                    flux_s1_summary[p].append(float(np.mean(arr)))

            vals_st = st.get(p, None)
            if vals_st is not None:
                arr = np.asarray(vals_st, dtype=float).reshape(-1)
                if arr.size > 0:
                    flux_st_summary[p].append(float(np.mean(arr)))

    def _mean_ci95(samples: list[float]) -> tuple[float, float]:
        if not samples:
            return np.nan, np.nan
        arr = np.asarray(samples, dtype=float)
        mean = float(np.mean(arr))
        if arr.size < 2:
            return mean, 0.0
        sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
        return mean, 1.96 * sem

    conc_s1_vals = []
    conc_s1_ci = []
    conc_st_vals = []
    conc_st_ci = []
    flux_s1_vals = []
    flux_s1_ci = []
    flux_st_vals = []
    flux_st_ci = []

    for p in param_names:
        m, c = _mean_ci95(conc_s1_summary[p])
        conc_s1_vals.append(m)
        conc_s1_ci.append(c)
        m, c = _mean_ci95(conc_st_summary[p])
        conc_st_vals.append(m)
        conc_st_ci.append(c)
        m, c = _mean_ci95(flux_s1_summary[p])
        flux_s1_vals.append(m)
        flux_s1_ci.append(c)
        m, c = _mean_ci95(flux_st_summary[p])
        flux_st_vals.append(m)
        flux_st_ci.append(c)

    conc_s1_vals = np.asarray(conc_s1_vals, dtype=float)
    conc_st_vals = np.asarray(conc_st_vals, dtype=float)
    flux_s1_vals = np.asarray(flux_s1_vals, dtype=float)
    flux_st_vals = np.asarray(flux_st_vals, dtype=float)
    conc_s1_ci = np.asarray(conc_s1_ci, dtype=float)
    conc_st_ci = np.asarray(conc_st_ci, dtype=float)
    flux_s1_ci = np.asarray(flux_s1_ci, dtype=float)
    flux_st_ci = np.asarray(flux_st_ci, dtype=float)

    if np.all(np.isnan(conc_s1_vals)) and np.all(np.isnan(flux_s1_vals)) and np.all(np.isnan(conc_st_vals)) and np.all(np.isnan(flux_st_vals)):
        print("Skipping Sobol bar summary: no finite summary values available.")
        return

    x_idx = np.arange(len(param_names), dtype=float)
    width = 0.38

    def _param_label(p: str) -> str:
        if p == "D_0":
            return r"$D_0$"
        if p == "G":
            return r"$G$"
        return p

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax = axes[0]
    ax.bar(
        x_idx - width / 2,
        np.nan_to_num(conc_s1_vals, nan=0.0),
        width=width,
        yerr=np.nan_to_num(conc_s1_ci, nan=0.0),
        capsize=3,
        color="black",
        alpha=0.9,
        label=r"$S_1$",
    )
    ax.bar(
        x_idx + width / 2,
        np.nan_to_num(conc_st_vals, nan=0.0),
        width=width,
        yerr=np.nan_to_num(conc_st_ci, nan=0.0),
        capsize=3,
        color="0.55",
        alpha=0.9,
        label=r"$S_T$",
    )
    ax.set_title("Concentration average Sobol", fontsize=14)
    ax.set_xlabel("Parameter", fontsize=13)
    ax.set_ylabel("Sobol index", fontsize=13)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([_param_label(p) for p in param_names], rotation=0, ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)

    ax = axes[1]
    ax.bar(
        x_idx - width / 2,
        np.nan_to_num(flux_s1_vals, nan=0.0),
        width=width,
        yerr=np.nan_to_num(flux_s1_ci, nan=0.0),
        capsize=3,
        color="black",
        alpha=0.9,
        label=r"$S_1$",
    )
    ax.bar(
        x_idx + width / 2,
        np.nan_to_num(flux_st_vals, nan=0.0),
        width=width,
        yerr=np.nan_to_num(flux_st_ci, nan=0.0),
        capsize=3,
        color="0.55",
        alpha=0.9,
        label=r"$S_T$",
    )
    ax.set_title("Outward flux average Sobol", fontsize=14)
    ax.set_xlabel("Parameter", fontsize=13)
    ax.set_xticks(x_idx)
    ax.set_xticklabels([_param_label(p) for p in param_names], rotation=0, ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    fig.savefig(plot_dir / "cj1959_sobol_summary_1x2.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_sobols_reanalysis(results, qois: list[str], run_dirs: list[Path], plot_dir: Path):
    """Create Sobol plots requested for reanalysis outputs.

    1) Sobol indices vs radius at the last time point.
    2) Sobol indices vs time at R=0.
    3) Sobol heatmap blocks on (time, radius): rows=input parameters, cols=outputs.
    """
    plot_dir.mkdir(parents=True, exist_ok=True)

    x_vals = _first_existing_x_from_runs(run_dirs)
    if x_vals is None:
        try:
            x_vals = np.asarray(results.describe("x", "mean"), dtype=float)
        except Exception:
            x_vals = None

    if x_vals is None or x_vals.ndim != 1 or x_vals.size == 0:
        print("Skipping Sobol plotting: x/radius coordinates unavailable.")
        return

    transient = _filter_time_entries(_transient_qois_sorted(qois))
    flux_transient: list[tuple[float, str]] = []
    for q in qois:
        t_flux = _parse_flux_time(q)
        if t_flux is not None:
            flux_transient.append((t_flux, q))
    flux_transient.sort(key=lambda it: it[0])
    flux_transient = _filter_time_entries(flux_transient)
    if not transient:
        print("Skipping Sobol plotting: no transient QoI labels found.")
        return

    # Ensure monotonic radius for cleaner plots/heatmaps.
    x = np.asarray(x_vals, dtype=float)
    radius_order = np.arange(len(x), dtype=int)
    if x.size > 1 and np.any(np.diff(x) < 0.0):
        radius_order = np.argsort(x)
        x = x[radius_order]

    # Determine uncertain input parameter names from available Sobol data.
    param_names: list[str] = []
    for _, q in transient:
        try:
            s1 = results.sobols_first(q)
        except Exception:
            s1 = None
        if isinstance(s1, dict):
            for p in s1.keys():
                if p not in param_names:
                    param_names.append(p)

    if not param_names:
        print("Skipping Sobol plotting: first-order Sobol indices unavailable.")
        return

    # 1) Sobol indices as function of radius at the last time point.
    t_last, q_last = transient[-1]
    try:
        s1_last = results.sobols_first(q_last)
    except Exception as exc:
        s1_last = None
        print(f"Skipping Sobol-vs-radius@last-time plot: {exc}")

    if isinstance(s1_last, dict):
        fig, ax = plt.subplots(figsize=(10, 6))
        for p in param_names:
            arr = s1_last.get(p, None)
            if arr is None:
                continue
            y = np.asarray(arr, dtype=float).reshape(-1)
            if y.size == len(x_vals):
                y = y[radius_order]
            elif y.size > 1:
                y = np.interp(
                    np.linspace(0.0, 1.0, len(x)),
                    np.linspace(0.0, 1.0, y.size),
                    y,
                )
            else:
                continue
            ax.plot(x, y, label=p)

        ax.set_title(f"First-order Sobol vs Radius at t={t_last:.3e} s")
        ax.set_xlabel("Radius [m]")
        ax.set_ylabel("Sobol index")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)
        ax.legend(loc="best")
        fig.savefig(plot_dir / "sobols_first_vs_radius_last_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2) Sobol indices as function of time at R=0.
    center_idx = int(np.argmin(np.abs(np.asarray(x_vals, dtype=float))))
    times = np.asarray([t for t, _ in transient], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6))
    any_curve = False

    for p in param_names:
        series = []
        for _, q in transient:
            try:
                s1 = results.sobols_first(q)
            except Exception:
                s1 = None
            vals = (s1 or {}).get(p, None) if isinstance(s1, dict) else None
            if vals is None:
                series.append(np.nan)
                continue
            v = np.asarray(vals, dtype=float).reshape(-1)
            if center_idx < len(v):
                series.append(float(v[center_idx]))
            elif v.size > 0:
                series.append(float(v[0]))
            else:
                series.append(np.nan)

        y = np.asarray(series, dtype=float)
        if np.all(np.isnan(y)):
            continue
        any_curve = True
        ax.plot(times, y, marker="o", label=p)

    if any_curve:
        ax.set_title("First-order Sobol vs Time at R=0")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Sobol index")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True)
        ax.legend(loc="best")
        fig.savefig(plot_dir / "sobols_first_vs_time_r0.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Flux Sobol indices as function of time (scalar QoI).
    if flux_transient:
        t_flux = np.asarray([t for t, _ in flux_transient], dtype=float)
        fig, ax = plt.subplots(figsize=(10, 6))
        any_flux_curve = False

        for p in param_names:
            series = []
            for _, q in flux_transient:
                try:
                    s1 = results.sobols_first(q)
                except Exception:
                    s1 = None

                vals = (s1 or {}).get(p, None) if isinstance(s1, dict) else None
                if vals is None:
                    series.append(np.nan)
                    continue

                arr = np.asarray(vals, dtype=float).reshape(-1)
                series.append(float(arr[0]) if arr.size > 0 else np.nan)

            y = np.asarray(series, dtype=float)
            if np.all(np.isnan(y)):
                continue
            any_flux_curve = True
            ax.plot(t_flux, y, marker="o", label=p)

        if any_flux_curve:
            ax.set_title("First-order Sobol vs Time for Outer Flux")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Sobol index")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True)
            if np.all(t_flux > 0):
                ax.set_xscale("log")
            ax.legend(loc="best")
            fig.savefig(plot_dir / "sobols_first_flux_vs_time.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3) Sobol heatmap blocks on (time, radius): rows=input params, cols=outputs.
    # Current transient QoIs represent a single output group.
    groups: dict[str, list[tuple[float, str]]] = {"tritium_concentration": transient}
    n_rows = len(param_names)
    n_cols = len(groups)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.8 * n_cols, 3.8 * n_rows), squeeze=False)

    for col_idx, (group_name, entries) in enumerate(groups.items()):
        t_vals = np.asarray([t for t, _ in entries], dtype=float)
        for row_idx, p in enumerate(param_names):
            ax = axes[row_idx, col_idx]
            z = np.full((len(entries), len(x)), np.nan, dtype=float)

            for i_t, (_, q) in enumerate(entries):
                try:
                    s1 = results.sobols_first(q)
                except Exception:
                    s1 = None
                vals = (s1 or {}).get(p, None) if isinstance(s1, dict) else None
                if vals is None:
                    continue

                v = np.asarray(vals, dtype=float).reshape(-1)
                if v.size == len(x_vals):
                    z[i_t, :] = v[radius_order]
                elif v.size > 1:
                    z[i_t, :] = np.interp(
                        np.linspace(0.0, 1.0, len(x)),
                        np.linspace(0.0, 1.0, v.size),
                        v,
                    )

            z_plot = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            m = ax.pcolormesh(x, t_vals, z_plot, shading="auto", vmin=0.0, vmax=1.0, cmap="viridis")
            cbar = fig.colorbar(m, ax=ax)
            cbar.set_label("Sobol index")

            if row_idx == 0:
                ax.set_title(f"QoI: {group_name}")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Radius [m]")
            ax.set_ylabel(f"Time [s]\n{p}")
            ax.grid(False)

    fig.suptitle("First-order Sobol heatmaps on (time, radius)", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    fig.savefig(plot_dir / "sobols_first_heatmap_blocks_t_x_r.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Sobol plots saved under: {plot_dir}")


def _plot_pce_pdf_reconstruction_last_time(results, qois: list[str], run_dirs: list[Path], plot_dir: Path, sampler):
    """Reconstruct PDFs at final time in linear and log10 spaces using the PCE surrogate."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    transient = _filter_time_entries(_transient_qois_sorted(qois))
    flux_transient = _filter_time_entries([(t, q) for q in qois if (t := _parse_flux_time(q)) is not None])
    if not transient or not flux_transient:
        print("Skipping PCE-PDF reconstruction: missing transient concentration/flux QoIs.")
        return

    x_vals = _first_existing_x_from_runs(run_dirs)
    if x_vals is None:
        try:
            x_vals = np.asarray(results.describe("x", "mean"), dtype=float)
        except Exception:
            x_vals = None
    if x_vals is None or x_vals.ndim != 1 or x_vals.size == 0:
        print("Skipping PCE-PDF reconstruction: x/radius coordinates unavailable.")
        return

    t_last_c, q_last_c = transient[-1]
    t_last_f, q_last_f = flux_transient[-1]
    center_idx = int(np.argmin(np.abs(np.asarray(x_vals, dtype=float))))

    try:
        pce_surrogate = results.surrogate()
    except Exception as exc:
        print(f"Skipping PCE-PDF reconstruction: surrogate unavailable ({exc}).")
        return

    try:
        param_names = list(sampler.vary.get_keys())
        joint_dist = sampler.distribution
    except Exception as exc:
        print(f"Skipping PCE-PDF reconstruction: sampler distributions unavailable ({exc}).")
        return

    n_mc = 20000
    try:
        draws = np.asarray(joint_dist.sample(n_mc, rule="sobol"), dtype=float)
    except Exception:
        draws = np.asarray(joint_dist.sample(n_mc), dtype=float)

    if draws.ndim == 1:
        draws = draws.reshape(1, -1)
    if draws.shape[0] != len(param_names):
        print("Skipping PCE-PDF reconstruction: sampled parameter shape mismatch.")
        return

    surrogate_in = {p: draws[i, :] for i, p in enumerate(param_names)}
    try:
        surrogate_out = pce_surrogate(surrogate_in)
    except Exception as exc:
        print(f"Skipping PCE-PDF reconstruction: surrogate evaluation failed ({exc}).")
        return

    c_all = np.asarray(surrogate_out.get(q_last_c, []), dtype=float)
    f_all = np.asarray(surrogate_out.get(q_last_f, []), dtype=float)
    if c_all.ndim != 2 or c_all.shape[1] <= center_idx:
        print("Skipping PCE-PDF reconstruction: concentration surrogate output shape mismatch.")
        return
    if f_all.ndim == 2 and f_all.shape[1] >= 1:
        f_samples = f_all[:, 0]
    elif f_all.ndim == 1:
        f_samples = f_all
    else:
        print("Skipping PCE-PDF reconstruction: flux surrogate output shape mismatch.")
        return

    c_samples = c_all[:, center_idx]
    c_samples = c_samples[np.isfinite(c_samples)]
    f_samples = f_samples[np.isfinite(f_samples)]
    if c_samples.size < 20 or f_samples.size < 20:
        print("Skipping PCE-PDF reconstruction: not enough finite surrogate samples.")
        return

    def _log10_positive(arr: np.ndarray) -> tuple[np.ndarray, float]:
        arr = np.asarray(arr, dtype=float).reshape(-1)
        positive_mask = arr > 0.0
        if arr.size == 0:
            return np.asarray([], dtype=float), 0.0
        excluded_fraction = 1.0 - float(np.count_nonzero(positive_mask)) / float(arr.size)
        return np.log10(arr[positive_mask]), excluded_fraction

    c_log10, c_excluded = _log10_positive(c_samples)
    f_log10, f_excluded = _log10_positive(f_samples)
    if c_log10.size < 20 or f_log10.size < 20:
        print("Skipping PCE-PDF reconstruction: not enough positive surrogate samples for log-space plot.")
        return

    c_pos = c_samples[c_samples > 0.0]
    f_pos = f_samples[f_samples > 0.0]
    c_excluded_lin = 1.0 - float(c_pos.size) / float(c_samples.size)
    f_excluded_lin = 1.0 - float(f_pos.size) / float(f_samples.size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(c_pos, bins=70, density=True, alpha=0.8, color="black", edgecolor="0.35")
    ax.set_title(f"PCE PDF (linear): concentration at r = 0, t = {t_last_c:.1f} s", fontsize=13)
    ax.set_xlabel("Mobile concentration [arb. units]")
    ax.set_ylabel("PDF")
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.98,
        f"excluded <= 0: {100.0 * c_excluded_lin:.2f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    ax = axes[1]
    ax.hist(f_pos, bins=70, density=True, alpha=0.8, color="0.45", edgecolor="0.2")
    ax.set_title(f"PCE PDF (linear): outward flux at t = {t_last_f:.1f} s", fontsize=13)
    ax.set_xlabel("Outward flux [arb. units]")
    ax.set_ylabel("PDF")
    ax.set_xlim(left=0.0)
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.98,
        f"excluded <= 0: {100.0 * f_excluded_lin:.2f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    fig.tight_layout()
    fig.savefig(plot_dir / "pce_pdf_reconstruction_last_time_center_flux.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(c_log10, bins=70, density=True, alpha=0.8, color="black", edgecolor="0.35")
    ax.set_title(f"PCE PDF (log10): concentration at r = 0, t = {t_last_c:.1f} s", fontsize=13)
    ax.set_xlabel(r"$\log_{10}$(mobile concentration)")
    ax.set_ylabel("PDF")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.98,
        f"excluded <= 0: {100.0 * c_excluded:.2f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    ax = axes[1]
    ax.hist(f_log10, bins=70, density=True, alpha=0.8, color="0.45", edgecolor="0.2")
    ax.set_title(f"PCE PDF (log10): outward flux at t = {t_last_f:.1f} s", fontsize=13)
    ax.set_xlabel(r"$\log_{10}$(outward flux)")
    ax.set_ylabel("PDF")
    ax.grid(True, alpha=0.3)
    ax.text(
        0.02,
        0.98,
        f"excluded <= 0: {100.0 * f_excluded:.2f}%",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="0.8"),
    )

    fig.tight_layout()
    fig.savefig(plot_dir / "pce_pdf_reconstruction_last_time_center_flux_log10.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_last_time_sobol_bars_center_and_flux(results, qois: list[str], run_dirs: list[Path], plot_dir: Path):
    """Plot last-time Sobol bar charts for center concentration and outward flux."""
    plot_dir.mkdir(parents=True, exist_ok=True)

    transient = _filter_time_entries(_transient_qois_sorted(qois))
    flux_transient = _filter_time_entries([(t, q) for q in qois if (t := _parse_flux_time(q)) is not None])
    if not transient or not flux_transient:
        print("Skipping last-time Sobol bars: missing transient concentration/flux QoIs.")
        return

    x_vals = _first_existing_x_from_runs(run_dirs)
    if x_vals is None:
        try:
            x_vals = np.asarray(results.describe("x", "mean"), dtype=float)
        except Exception:
            x_vals = None
    if x_vals is None or x_vals.ndim != 1 or x_vals.size == 0:
        print("Skipping last-time Sobol bars: x/radius coordinates unavailable.")
        return

    center_idx = int(np.argmin(np.abs(np.asarray(x_vals, dtype=float))))
    t_last_c, q_last_c = transient[-1]
    t_last_f, q_last_f = flux_transient[-1]

    try:
        s1_c = results.sobols_first(q_last_c)
        st_c = results.sobols_total(q_last_c)
        s1_f = results.sobols_first(q_last_f)
        st_f = results.sobols_total(q_last_f)
    except Exception as exc:
        print(f"Skipping last-time Sobol bars: Sobol data unavailable ({exc}).")
        return

    param_names: list[str] = []
    for dct in (s1_c, st_c, s1_f, st_f):
        if isinstance(dct, dict):
            for p in dct.keys():
                if p not in param_names:
                    param_names.append(p)
    if not param_names:
        print("Skipping last-time Sobol bars: no Sobol parameter names available.")
        return

    def _scalar_from(vals: Any, idx: int | None = None) -> float:
        if vals is None:
            return float("nan")
        arr = np.asarray(vals, dtype=float).reshape(-1)
        if arr.size == 0:
            return float("nan")
        if idx is not None and idx < arr.size:
            return float(arr[idx])
        return float(arr[0])

    c_s1_vals = np.asarray([_scalar_from((s1_c or {}).get(p, None), center_idx) for p in param_names], dtype=float)
    c_st_vals = np.asarray([_scalar_from((st_c or {}).get(p, None), center_idx) for p in param_names], dtype=float)
    f_s1_vals = np.asarray([_scalar_from((s1_f or {}).get(p, None), None) for p in param_names], dtype=float)
    f_st_vals = np.asarray([_scalar_from((st_f or {}).get(p, None), None) for p in param_names], dtype=float)

    x_idx = np.arange(len(param_names), dtype=float)
    width = 0.38

    def _param_label(p: str) -> str:
        if p == "D_0":
            return r"$D_0$"
        if p == "G":
            return r"$G$"
        return p

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    ax = axes[0]
    ax.bar(x_idx - width / 2, np.nan_to_num(c_s1_vals, nan=0.0), width=width, color="black", alpha=0.9, label=r"$S_1$")
    ax.bar(x_idx + width / 2, np.nan_to_num(c_st_vals, nan=0.0), width=width, color="0.55", alpha=0.9, label=r"$S_T$")
    ax.set_title(f"Center concentration Sobol at t = {t_last_c:.1f} s", fontsize=13)
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Sobol index")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([_param_label(p) for p in param_names], rotation=0, ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.bar(x_idx - width / 2, np.nan_to_num(f_s1_vals, nan=0.0), width=width, color="black", alpha=0.9, label=r"$S_1$")
    ax.bar(x_idx + width / 2, np.nan_to_num(f_st_vals, nan=0.0), width=width, color="0.55", alpha=0.9, label=r"$S_T$")
    ax.set_title(f"Outward flux Sobol at t = {t_last_f:.1f} s", fontsize=13)
    ax.set_xlabel("Parameter")
    ax.set_xticks(x_idx)
    ax.set_xticklabels([_param_label(p) for p in param_names], rotation=0, ha="center")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(plot_dir / "sobols_last_time_center_and_flux_bars.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_cj1959_verification(cfg: dict[str, Any], results, qois: list[str], run_dirs: list[Path], plot_dir: Path):
    """Create analytical-only and analytical-vs-simulation plots for CJ1959 case."""
    if not _is_cj1959_case(cfg):
        print("Skipping CJ1959 verification plotting: case-gate not matched.")
        return

    # Lazy import to avoid hard dependency on tests package for unrelated runs.
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    tests_dir = repo_root / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))

    try:
        from verification.gfederici1991 import CarlsJaeger1959, PlotCarlsJaeger1959
    except Exception as exc:
        print(f"Skipping CJ1959 verification plotting: cannot import verification module ({exc})")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    geometry_cfg = cfg.get("geometry", {}) or {}
    domains = geometry_cfg.get("domains", [{}]) or [{}]
    domain0 = domains[0] if domains else {}

    materials = cfg.get("materials", []) or [{}]
    domain_material_id = domain0.get("material", None)
    material_cfg = next(
        (m for m in materials if m.get("material_id", None) == domain_material_id),
        materials[0] if materials else {},
    )

    d0 = float((material_cfg.get("D_0", {}) or {}).get("mean", 1.0))
    e_d = float((material_cfg.get("E_D", {}) or {}).get("mean", 0.0))
    temperature = float(
        (((cfg.get("initial_conditions", {}) or {}).get("temperature", {}) or {}).get("value", {}) or {}).get(
            "mean", (cfg.get("model_parameters", {}) or {}).get("T_0", 300.0)
        )
    )

    # FESTIM materials use Arrhenius diffusivity: D(T) = D0 * exp(-E_D / (k_B * T)).
    k_b_ev_per_k = 8.617333262145e-5
    D = d0 * math.exp(-e_d / (k_b_ev_per_k * temperature)) if temperature > 0.0 else d0

    G = float(
        (((cfg.get("source_terms", {}) or {}).get("concentration", {}) or {}).get("value", {}) or {}).get("mean", 1.0)
    )
    a = float((domains[0] or {}).get("length", 1.0))

    print(
        "CJ1959 verification parameters from config: "
        f"material_id={domain_material_id}, D0={d0:.6e}, E_D={e_d:.6e}, T={temperature:.3f} K, "
        f"D_used={D:.6e}, G={G:.6e}, a={a:.6e}"
    )

    # Prefer mesh coordinates from FESTIM output files to preserve refined spacing.
    x_vals = _first_existing_x_from_runs(run_dirs)
    if x_vals is None:
        try:
            x_vals = np.asarray(results.describe("x", "mean"))
        except Exception:
            x_vals = None

    # Standalone analytical profiles per transient QoI time.
    for q in [q for q in qois if q.startswith("t=") and q.endswith("s")]:
        try:
            t = float(q.split("=", 1)[1].rstrip("s"))
        except Exception:
            continue
        out = plot_dir / f"cj1959_analytic_{q.replace('=', '_').replace('+', 'p').replace('-', 'm')}.png"
        PlotCarlsJaeger1959(t=t, D=D, G=G, a=a, m=len(x_vals) if x_vals is not None else 128, output_path=out)

    # Overlay analytical and simulation-mean profiles.
    if x_vals is None:
        print("Skipping CJ1959 overlay plots: x coordinate unavailable.")
        return

    for q in [q for q in qois if q.startswith("t=") and q.endswith("s")]:
        try:
            t = float(q.split("=", 1)[1].rstrip("s"))
            c_mean = np.asarray(results.describe(q, "mean"), dtype=float)
            if c_mean.ndim != 1 or len(c_mean) != len(x_vals):
                continue
        except Exception:
            continue

        try:
            c_std = np.asarray(results.describe(q, "std"), dtype=float)
            if c_std.shape != c_mean.shape:
                c_std = np.zeros_like(c_mean)
        except Exception:
            c_std = np.zeros_like(c_mean)

        try:
            c_p01 = np.asarray(results.describe(q, "1%"), dtype=float)
            if c_p01.shape != c_mean.shape:
                c_p01 = np.full_like(c_mean, np.nan, dtype=float)
        except Exception:
            c_p01 = np.full_like(c_mean, np.nan, dtype=float)

        try:
            c_p99 = np.asarray(results.describe(q, "99%"), dtype=float)
            if c_p99.shape != c_mean.shape:
                c_p99 = np.full_like(c_mean, np.nan, dtype=float)
        except Exception:
            c_p99 = np.full_like(c_mean, np.nan, dtype=float)

        r_sim = np.asarray(x_vals)
        if r_sim.ndim == 1 and r_sim.size > 1 and np.any(np.diff(r_sim) < 0.0):
            order = np.argsort(r_sim)
            r_sim = r_sim[order]
            c_mean = c_mean[order]
            c_std = c_std[order]
            c_p01 = c_p01[order]
            c_p99 = c_p99[order]

        c_ana = np.asarray(CarlsJaeger1959(t=t, D=D, G=G, a=a, r=r_sim), dtype=float)
        has_q01_q99 = np.all(np.isfinite(c_p01)) and np.all(np.isfinite(c_p99))

        out = plot_dir / f"cj1959_overlay_{q.replace('=', '_').replace('+', 'p').replace('-', 'm')}.png"
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(r_sim, c_ana, label="CJ1959 analytic", color="tab:blue")
        ax.plot(r_sim, c_mean, label="UQ mean", color="tab:orange", linestyle="--")
        ax.fill_between(r_sim, c_mean - c_std, c_mean + c_std, alpha=0.2, label="+/- STD")
        ax.plot(r_sim, c_mean - c_std, alpha=0.35, linewidth=0.8, linestyle=":", color="tab:orange", label="_nolegend_")
        ax.plot(r_sim, c_mean + c_std, alpha=0.35, linewidth=0.8, linestyle=":", color="tab:orange", label="_nolegend_")
        if has_q01_q99:
            ax.fill_between(r_sim, c_p01, c_p99, alpha=0.12, color="tab:green", label="1%-99%")
            ax.plot(r_sim, c_p01, alpha=0.35, linewidth=0.8, linestyle="--", color="tab:green", label="_nolegend_")
            ax.plot(r_sim, c_p99, alpha=0.35, linewidth=0.8, linestyle="--", color="tab:green", label="_nolegend_")
        ax.set_title(f"CJ1959 Verification Overlay at t={t:.2e}s")
        ax.set_xlabel("Radius r [m]")
        ax.set_ylabel("Concentration")
        ax.grid(True)
        ax.legend(loc="best")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _plot_cj1959_center_vs_time(
        results=results,
        qois=qois,
        x_vals=x_vals,
        D=D,
        G=G,
        a=a,
        analytic_solver=CarlsJaeger1959,
        plot_dir=plot_dir,
    )
    _plot_cj1959_uq_dashboard_2x2(
        results=results,
        qois=qois,
        x_vals=np.asarray(x_vals, dtype=float),
        D=D,
        G=G,
        a=a,
        analytic_solver=CarlsJaeger1959,
        plot_dir=plot_dir,
    )
    _plot_cj1959_sobol_bar_summary(results, qois, np.asarray(x_vals, dtype=float), plot_dir)

    print(f"✓ CJ1959 verification plots saved under: {plot_dir}")


def _plot_cj1959_center_vs_time(
    results,
    qois: list[str],
    x_vals: np.ndarray,
    D: float,
    G: float,
    a: float,
    analytic_solver,
    plot_dir: Path,
):
    """Plot center-point (r=0) concentration vs time for verification and UQ."""
    transient_qois: list[tuple[float, str]] = []
    for q in qois:
        if not (q.startswith("t=") and q.endswith("s")):
            continue
        try:
            t = float(q.split("=", 1)[1].rstrip("s"))
            transient_qois.append((t, q))
        except Exception:
            continue

    if not transient_qois:
        print("Skipping center-vs-time plots: no transient QoI columns found.")
        return

    transient_qois.sort(key=lambda pair: pair[0])
    transient_qois = _filter_time_entries(transient_qois)

    if not transient_qois:
        print(f"Skipping center-vs-time plots: no times >= {TIME_PLOT_MIN_S:g}s.")
        return

    center_idx = int(np.argmin(np.abs(np.asarray(x_vals))))

    times: list[float] = []
    uq_center_mean: list[float] = []
    uq_center_std: list[float] = []
    uq_center_p01: list[float] = []
    uq_center_p99: list[float] = []
    analytic_center: list[float] = []

    for t, q in transient_qois:
        try:
            mean_profile = np.asarray(results.describe(q, "mean"))
            if mean_profile.ndim != 1 or center_idx >= len(mean_profile):
                continue
        except Exception:
            continue

        try:
            std_profile = np.asarray(results.describe(q, "std"))
            if std_profile.shape != mean_profile.shape:
                std_profile = np.zeros_like(mean_profile)
        except Exception:
            std_profile = np.zeros_like(mean_profile)

        # Quantiles may be unavailable for some PCE analysis settings.
        try:
            p01_profile = np.asarray(results.describe(q, "1%"))
            if p01_profile.shape != mean_profile.shape:
                p01_profile = np.full_like(mean_profile, np.nan, dtype=float)
        except Exception:
            p01_profile = np.full_like(mean_profile, np.nan, dtype=float)

        try:
            p99_profile = np.asarray(results.describe(q, "99%"))
            if p99_profile.shape != mean_profile.shape:
                p99_profile = np.full_like(mean_profile, np.nan, dtype=float)
        except Exception:
            p99_profile = np.full_like(mean_profile, np.nan, dtype=float)

        ana_profile = np.asarray(analytic_solver(t=t, D=D, G=G, a=a, m=len(x_vals)))

        times.append(float(t))
        uq_center_mean.append(float(mean_profile[center_idx]))
        uq_center_std.append(float(std_profile[center_idx]))
        uq_center_p01.append(float(p01_profile[center_idx]))
        uq_center_p99.append(float(p99_profile[center_idx]))
        analytic_center.append(float(ana_profile[center_idx]))

    if not times:
        print("Skipping center-vs-time plots: no valid center-point data extracted.")
        return

    t_arr = np.asarray(times)
    uq_mean_arr = np.asarray(uq_center_mean)
    uq_std_arr = np.asarray(uq_center_std)
    uq_p01_arr = np.asarray(uq_center_p01)
    uq_p99_arr = np.asarray(uq_center_p99)
    ana_arr = np.asarray(analytic_center)

    valid_q01_q99 = np.all(np.isfinite(uq_p01_arr)) and np.all(np.isfinite(uq_p99_arr))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_arr, ana_arr, marker="o", label="CJ1959 analytic @ r=0", color="tab:blue")
    ax.set_title("Verification Center Value vs Time (r=0)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend(loc="best")
    fig.savefig(plot_dir / "cj1959_center_vs_time_verification.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_arr, uq_mean_arr, marker="o", label="UQ mean @ r=0", color="tab:orange")
    ax.fill_between(t_arr, uq_mean_arr - uq_std_arr, uq_mean_arr + uq_std_arr, alpha=0.2, label="+/- STD")
    ax.plot(t_arr, uq_mean_arr - uq_std_arr, alpha=0.35, linewidth=0.8, linestyle=":", color="tab:orange", label="_nolegend_")
    ax.plot(t_arr, uq_mean_arr + uq_std_arr, alpha=0.35, linewidth=0.8, linestyle=":", color="tab:orange", label="_nolegend_")
    if valid_q01_q99:
        ax.fill_between(t_arr, uq_p01_arr, uq_p99_arr, alpha=0.12, color="tab:green", label="1%-99%")
        ax.plot(t_arr, uq_p01_arr, alpha=0.35, linewidth=0.8, linestyle="--", color="tab:green", label="_nolegend_")
        ax.plot(t_arr, uq_p99_arr, alpha=0.35, linewidth=0.8, linestyle="--", color="tab:green", label="_nolegend_")
    ax.set_title("UQ Center Value vs Time (r=0)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend(loc="best")
    fig.savefig(plot_dir / "cj1959_center_vs_time_uq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_arr, ana_arr, marker="o", label="CJ1959 analytic @ r=0", color="tab:blue")
    ax.plot(t_arr, uq_mean_arr, marker="s", label="UQ mean @ r=0", color="tab:orange")
    ax.fill_between(t_arr, uq_mean_arr - uq_std_arr, uq_mean_arr + uq_std_arr, alpha=0.2, label="+/- STD")
    ax.plot(t_arr, uq_mean_arr - uq_std_arr, alpha=0.35, linewidth=0.8, linestyle=":", color="tab:orange", label="_nolegend_")
    ax.plot(t_arr, uq_mean_arr + uq_std_arr, alpha=0.35, linewidth=0.8, linestyle=":", color="tab:orange", label="_nolegend_")
    if valid_q01_q99:
        ax.fill_between(t_arr, uq_p01_arr, uq_p99_arr, alpha=0.12, color="tab:green", label="1%-99%")
        ax.plot(t_arr, uq_p01_arr, alpha=0.35, linewidth=0.8, linestyle="--", color="tab:green", label="_nolegend_")
        ax.plot(t_arr, uq_p99_arr, alpha=0.35, linewidth=0.8, linestyle="--", color="tab:green", label="_nolegend_")
    ax.set_title("Center Value Comparison vs Time (r=0)")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Concentration")
    ax.grid(True)
    ax.legend(loc="best")
    fig.savefig(plot_dir / "cj1959_center_vs_time_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cj1959_uq_dashboard_2x2(
    results,
    qois: list[str],
    x_vals: np.ndarray,
    D: float,
    G: float,
    a: float,
    analytic_solver,
    plot_dir: Path,
):
    """Create CJ1959 UQ 2x2 dashboard plot."""
    transient_qois: list[tuple[float, str]] = []
    for q in qois:
        if not (isinstance(q, str) and q.startswith("t=") and q.endswith("s")):
            continue
        try:
            transient_qois.append((float(q.split("=", 1)[1].rstrip("s")), q))
        except Exception:
            continue

    if not transient_qois:
        print("Skipping CJ1959 dashboard: no transient QoIs found.")
        return

    transient_qois.sort(key=lambda it: it[0])
    transient_qois = _filter_time_entries(transient_qois)

    if not transient_qois:
        print(f"Skipping CJ1959 dashboard: no transient QoIs at t>={TIME_PLOT_MIN_S:g}s.")
        return

    r = np.asarray(x_vals, dtype=float)
    if r.ndim != 1 or r.size == 0:
        print("Skipping CJ1959 dashboard: invalid radius array.")
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
    except Exception as exc:
        print(f"Skipping CJ1959 dashboard: final-time stats unavailable ({exc}).")
        return

    if mean_last.size != r.size:
        print("Skipping CJ1959 dashboard: final-time mean shape mismatch.")
        return
    if std_last.shape != mean_last.shape:
        std_last = np.zeros_like(mean_last)

    try:
        p01_last = np.asarray(results.describe(q_last, "1%"), dtype=float)
        if p01_last.shape != mean_last.shape:
            p01_last = np.full_like(mean_last, np.nan, dtype=float)
    except Exception:
        p01_last = np.full_like(mean_last, np.nan, dtype=float)

    try:
        p99_last = np.asarray(results.describe(q_last, "99%"), dtype=float)
        if p99_last.shape != mean_last.shape:
            p99_last = np.full_like(mean_last, np.nan, dtype=float)
    except Exception:
        p99_last = np.full_like(mean_last, np.nan, dtype=float)

    mean_last = mean_last[order]
    std_last = std_last[order]
    p01_last = p01_last[order]
    p99_last = p99_last[order]
    ana_last = np.asarray(analytic_solver(t=t_last, D=D, G=G, a=a, r=r), dtype=float)

    err_last = np.abs(mean_last - ana_last)
    err_last_lo = np.abs((mean_last - std_last) - ana_last)
    err_last_hi = np.abs((mean_last + std_last) - ana_last)
    err_last_min = np.minimum(err_last_lo, err_last_hi)
    err_last_max = np.maximum(err_last_lo, err_last_hi)
    has_q01_q99_last = np.all(np.isfinite(p01_last)) and np.all(np.isfinite(p99_last))

    times = []
    c_mean = []
    c_std = []
    c_ana = []
    c_p01 = []
    c_p99 = []
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

        try:
            p01_prof = np.asarray(results.describe(q, "1%"), dtype=float)
            if p01_prof.shape != mean_prof.shape:
                p01_prof = np.full_like(mean_prof, np.nan, dtype=float)
        except Exception:
            p01_prof = np.full_like(mean_prof, np.nan, dtype=float)

        try:
            p99_prof = np.asarray(results.describe(q, "99%"), dtype=float)
            if p99_prof.shape != mean_prof.shape:
                p99_prof = np.full_like(mean_prof, np.nan, dtype=float)
        except Exception:
            p99_prof = np.full_like(mean_prof, np.nan, dtype=float)

        mean_prof = mean_prof[order]
        std_prof = std_prof[order]
        p01_prof = p01_prof[order]
        p99_prof = p99_prof[order]
        ana_prof = np.asarray(analytic_solver(t=t, D=D, G=G, a=a, r=r), dtype=float)

        l2_c = float(np.sqrt(np.trapz((mean_prof - ana_prof) ** 2, x=r)))
        l2_lo = float(np.sqrt(np.trapz(((mean_prof - std_prof) - ana_prof) ** 2, x=r)))
        l2_hi = float(np.sqrt(np.trapz(((mean_prof + std_prof) - ana_prof) ** 2, x=r)))

        times.append(float(t))
        c_mean.append(float(mean_prof[center_idx]))
        c_std.append(float(std_prof[center_idx]))
        c_ana.append(float(ana_prof[center_idx]))
        c_p01.append(float(p01_prof[center_idx]))
        c_p99.append(float(p99_prof[center_idx]))
        l2_err.append(l2_c)
        l2_err_min.append(min(l2_lo, l2_hi))
        l2_err_max.append(max(l2_lo, l2_hi))

    if not times:
        print("Skipping CJ1959 dashboard: no valid center time series.")
        return

    t_arr = np.asarray(times, dtype=float)
    c_mean = np.asarray(c_mean, dtype=float)
    c_std = np.asarray(c_std, dtype=float)
    c_ana = np.asarray(c_ana, dtype=float)
    c_p01 = np.asarray(c_p01, dtype=float)
    c_p99 = np.asarray(c_p99, dtype=float)
    has_q01_q99_time = np.all(np.isfinite(c_p01)) and np.all(np.isfinite(c_p99))

    l2_err = np.asarray(l2_err, dtype=float)
    l2_err_min = np.asarray(l2_err_min, dtype=float)
    l2_err_max = np.asarray(l2_err_max, dtype=float)
    l2_err_max_std = np.maximum(l2_err_max, l2_err)

    floor = np.finfo(float).tiny

    def _nice_upper_and_step(y_max: float) -> tuple[float, float]:
        """Round up y_max to a step where the second significant digit is 0 or 5."""
        if not np.isfinite(y_max) or y_max <= 0.0:
            return 1.0, 0.5
        exponent = int(np.floor(np.log10(y_max)))
        step = 5.0 * (10.0 ** (exponent - 1))
        if step <= 0.0:
            step = 0.5
        y_top = step * np.ceil(y_max / step)
        if y_top <= 0.0:
            y_top = step
        return float(y_top), float(step)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(r, mean_last, "-", color="black", label="Simulation campaign")
    ax.fill_between(r, mean_last - std_last, mean_last + std_last, alpha=0.22, color="0.5", label=r"$\pm \sigma$")
    ax.plot(r, mean_last - std_last, alpha=0.35, linewidth=0.8, linestyle=":", color="0.35", label="_nolegend_")
    ax.plot(r, mean_last + std_last, alpha=0.35, linewidth=0.8, linestyle=":", color="0.35", label="_nolegend_")
    if has_q01_q99_last:
        ax.fill_between(r, p01_last, p99_last, alpha=0.12, color="0.75", label="1% - 99%")
        ax.plot(r, p01_last, alpha=0.35, linewidth=0.8, linestyle="--", color="0.45", label="_nolegend_")
        ax.plot(r, p99_last, alpha=0.35, linewidth=0.8, linestyle="--", color="0.45", label="_nolegend_")
    ax.plot(r, ana_last, linestyle="--", color="red", label="CJ1959 verification")
    ax.set_title(f"Mobile concentration vs Radius at t = {t_last:.1f} s", fontsize=14)
    ax.set_xlabel("Radius r [m]")
    ax.set_ylabel("Mobile concentration [arb. units]")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True)
    ax.legend(loc="lower left")

    ax = axes[0, 1]
    ax.plot(r, np.maximum(err_last, floor), color="black", label="|simulation campaign - verification|")
    ax.fill_between(r, np.maximum(err_last_min, floor), np.maximum(err_last_max, floor), alpha=0.22, color="0.5", label=r"error from $\pm \sigma$")
    ax.plot(
        r,
        np.maximum(err_last_max, floor),
        linestyle="--",
        color="red",
        label=r"Max error within $\pm \sigma$",
    )
    ax.set_title(f"Absolute error vs Radius at t = {t_last:.1f} s", fontsize=14)
    ax.set_xlabel("Radius r [m]")
    ax.set_ylabel("Absolute error [arb. units]")
    ax.set_yscale("log")
    err_top = max(float(np.nanmax(np.maximum(err_last, floor))), float(np.nanmax(np.maximum(err_last_max, floor))))
    ax.set_ylim(1.0e-15, max(err_top, 1.0e-14))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True)
    ax.legend(loc="lower left")

    ax = axes[1, 0]
    ax.plot(t_arr, c_mean, "-", marker="s", color="black", label="Simulation campaign at r = 0")
    ax.fill_between(t_arr, c_mean - c_std, c_mean + c_std, alpha=0.22, color="0.5", label=r"$\pm \sigma$")
    ax.plot(t_arr, c_mean - c_std, alpha=0.35, linewidth=0.8, linestyle=":", color="0.35", label="_nolegend_")
    ax.plot(t_arr, c_mean + c_std, alpha=0.35, linewidth=0.8, linestyle=":", color="0.35", label="_nolegend_")
    if has_q01_q99_time:
        ax.fill_between(t_arr, c_p01, c_p99, alpha=0.12, color="0.75", label="1% - 99%")
        ax.plot(t_arr, c_p01, alpha=0.35, linewidth=0.8, linestyle="--", color="0.45", label="_nolegend_")
        ax.plot(t_arr, c_p99, alpha=0.35, linewidth=0.8, linestyle="--", color="0.45", label="_nolegend_")
    ax.plot(t_arr, c_ana, linestyle="--", marker="o", color="red", label="CJ1959 verification at r = 0")
    ax.set_title("Mobile concentration vs Time at r = 0", fontsize=14)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Mobile concentration [arb. units]")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True)
    ax.legend(loc="lower left")

    ax = axes[1, 1]
    ax.plot(t_arr, np.maximum(l2_err, floor), marker="d", color="black", label="L2 error")
    ax.plot(
        t_arr,
        np.maximum(l2_err_max_std, floor),
        marker="^",
        linestyle="--",
        color="red",
        label=r"Max L2 within $\pm \sigma$",
    )
    ax.set_title("L2 error vs Time", fontsize=14)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Absolute error [arb. units]")
    ax.set_yscale("log")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True)
    ax.legend(loc="lower left")

    # Keep left-column ranges/ticks aligned.
    y_left_max = max(np.max(mean_last + std_last), np.max(c_mean + c_std))
    if has_q01_q99_last:
        y_left_max = max(y_left_max, np.max(p99_last))
    if has_q01_q99_time:
        y_left_max = max(y_left_max, np.max(c_p99))
    y_left_top, tick_step = _nice_upper_and_step(float(y_left_max))
    y_left_min = 0.0
    axes[0, 0].set_ylim(y_left_min, y_left_top)
    axes[1, 0].set_ylim(y_left_min, y_left_top)
    shared_ticks = np.arange(y_left_min, y_left_top + 0.5 * tick_step, tick_step)
    axes[0, 0].set_yticks(shared_ticks)
    axes[1, 0].set_yticks(shared_ticks)

    fig.tight_layout()
    fig.savefig(plot_dir / "cj1959_verification_dashboard_2x2.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def _is_cj1959_case(cfg: dict[str, Any]) -> bool:
    """Case gate for Carslaw-Jaeger 1959 verification post-processing."""
    text_tokens = [
        str(cfg.get("case_name", "")),
        str((cfg.get("case", {}) or {}).get("name", "")),
        str((cfg.get("model_parameters", {}) or {}).get("case_name", "")),
        str((cfg.get("model_parameters", {}) or {}).get("verification_case", "")),
    ]
    token_blob = " ".join(text_tokens).lower()
    if any(k in token_blob for k in ("cj1959", "carslaw", "jaeger")):
        return True

    geometry_cfg = cfg.get("geometry", {}) or {}
    coordinate_system = str(geometry_cfg.get("coordinate_system", "")).lower()

    qoi_names = ((cfg.get("simulation", {}) or {}).get("quantities_of_interest", []) or [])
    qoi_names = [str(v).lower() for v in qoi_names]

    left_bc = (((cfg.get("boundary_conditions", {}) or {}).get("concentration", {}) or {}).get("left", {}) or {})
    right_bc = (((cfg.get("boundary_conditions", {}) or {}).get("concentration", {}) or {}).get("right", {}) or {})

    left_type = str(left_bc.get("type", "")).lower()
    right_type = str(right_bc.get("type", "")).lower()

    left_val = ((left_bc.get("value", {}) or {}).get("mean", None))
    right_val = ((right_bc.get("value", {}) or {}).get("mean", None))

    concentration_source = (((cfg.get("source_terms", {}) or {}).get("concentration", {}) or {}))
    source_type = str(concentration_source.get("type", "")).lower()
    source_mean = ((concentration_source.get("value", {}) or {}).get("mean", None))

    initial_c = (((cfg.get("initial_conditions", {}) or {}).get("concentration", {}) or {}))
    initial_c_mean = ((initial_c.get("value", {}) or {}).get("mean", None))

    return (
        coordinate_system == "spherical"
        and "tritium_concentration" in qoi_names
        and left_type == "neumann"
        and right_type == "dirichlet"
        and left_val == 0.0
        and right_val == 0.0
        and source_type == "constant"
        and source_mean is not None
        and initial_c_mean == 0.0
    )


def _build_qois_from_config(config_yaml: Path) -> tuple[dict[str, Any], list[str]]:
    with config_yaml.open("r") as fh:
        cfg = yaml.safe_load(fh)

    transient = bool(cfg.get("model_parameters", {}).get("transient", False))
    milestone_times = cfg.get("simulation", {}).get("milestone_times", []) or []

    if transient and milestone_times:
        qois = ["x"] + [f"t={float(t):.2e}s" for t in milestone_times] + [f"flux_t={float(t):.2e}s" for t in milestone_times]
    else:
        qois = ["x", "t=steady"]

    return cfg, qois


def _restart_recollate_if_possible(campaign_dir: Path, qois: list[str]) -> tuple[bool, Any, Any]:
    """Try true restart path. Returns (ok, campaign, results_or_exception)."""
    campaign_db = campaign_dir / "campaign.db"
    app_name, _, _ = _load_app_params_defaults(campaign_db)

    campaign = uq.Campaign(
        name=app_name,
        db_location=f"sqlite:///{campaign_db.resolve()}",
        work_dir=str(campaign_dir.parent.resolve()),
        change_to_state=True,
    )

    concentration_qois = [q for q in qois if q == "x" or _parse_transient_time(q) is not None or q == "t=steady"]
    flux_qois = [q for q in qois if _parse_flux_time(q) is not None]
    decoder = MultiOutputDecoder(
        profile_filename="results/test/results_tritium_concentration.txt",
        flux_filename="results/test/total_hydrogen_flux_rmax.txt",
        concentration_qois=concentration_qois,
        flux_qois=flux_qois,
    )

    campaign.replace_actions(app_name, Actions(Decode(decoder)))

    try:
        campaign.recollate()
    except Exception as exc:  # Compatibility issue in some EasyVVUQ versions
        return False, campaign, exc

    sampler = campaign.get_active_sampler()
    results, _ = _run_analysis(
        campaign,
        sampler,
        qois,
        campaign_dir / "reanalysis_restart_outputs",
    )
    return True, campaign, results


def _fallback_rebuild_and_analyse(
    campaign_dir: Path,
    qois: list[str],
    cfg: dict[str, Any],
    verification_cfg: dict[str, Any] | None = None,
):
    """Fallback: build analysis-only campaign from existing run files (no reruns)."""
    campaign_db = campaign_dir / "campaign.db"
    app_name, params_json, defaults = _load_app_params_defaults(campaign_db)
    sampler = _load_sampler_from_db(campaign_db)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rebuilt_dir = campaign_dir / f"reanalysis_rebuild_{timestamp}"
    rebuilt_dir.mkdir(parents=True, exist_ok=True)

    campaign = uq.Campaign(
        name=f"{app_name}_reanalysis_{timestamp}",
        params=params_json,
        work_dir=str(rebuilt_dir.resolve()),
        db_location=f"sqlite:///{(rebuilt_dir / 'campaign.db').resolve()}",
    )

    campaign.add_app(name="reanalysis_app", params=params_json, actions=Actions())
    campaign.set_app("reanalysis_app")
    campaign.set_sampler(sampler)

    run_dirs = _find_run_dirs(campaign_dir)
    input_files = [str(p / "config.yaml") for p in run_dirs]
    output_files = [str(p / "results/test/results_tritium_concentration.txt") for p in run_dirs]

    input_decoder = YAMLParamDecoder(defaults=defaults, target_filename="config.yaml")
    concentration_qois = [q for q in qois if q == "x" or _parse_transient_time(q) is not None or q == "t=steady"]
    flux_qois = [q for q in qois if _parse_flux_time(q) is not None]
    output_decoder = MultiOutputDecoder(
        profile_filename="results_tritium_concentration.txt",
        flux_filename="total_hydrogen_flux_rmax.txt",
        concentration_qois=concentration_qois,
        flux_qois=flux_qois,
    )

    campaign.add_external_runs(input_files, output_files, input_decoder, output_decoder)

    results, _ = _run_analysis(
        campaign,
        sampler,
        qois,
        rebuilt_dir,
    )

    _plot_reanalysis(results, qois, run_dirs, rebuilt_dir / "plots")
    _plot_sobols_reanalysis(results, qois, run_dirs, rebuilt_dir / "plots")
    _plot_pce_pdf_reconstruction_last_time(results, qois, run_dirs, rebuilt_dir / "plots", sampler)
    _plot_last_time_sobol_bars_center_and_flux(results, qois, run_dirs, rebuilt_dir / "plots")
    _plot_cj1959_verification(verification_cfg or cfg, results, qois, run_dirs, rebuilt_dir / "plots")
    _generate_uq_dashboard(campaign_dir / "runs", rebuilt_dir / "plots" / "uq_campaign_dashboard.html")

    # Save config used for the analysis rerun.
    cfg_out = rebuilt_dir / add_timestamp_to_filename("reanalysis_config.pickle", timestamp)
    with cfg_out.open("wb") as fh:
        pickle.dump(cfg, fh)

    return rebuilt_dir


def main():
    parser = argparse.ArgumentParser(description="Reanalyse an existing FESTIM EasyVVUQ campaign")
    parser.add_argument("--campaign-dir", required=True, help="Path to campaign folder containing campaign.db")
    parser.add_argument(
        "--verification-config",
        default=None,
        help="Optional YAML config path to use for verification plotting parameters (D, E_D, T, G, geometry).",
    )
    args = parser.parse_args()

    campaign_dir = Path(args.campaign_dir).resolve()
    if not (campaign_dir / "campaign.db").exists():
        raise FileNotFoundError(f"campaign.db not found under {campaign_dir}")

    run_dirs = _find_run_dirs(campaign_dir)
    if not run_dirs:
        raise RuntimeError(f"No run directories with config.yaml found under {campaign_dir}")

    cfg, qois = _build_qois_from_config(run_dirs[0] / "config.yaml")
    print(f"Using QoI columns for decoding: {qois}")

    verification_cfg = cfg
    if args.verification_config:
        verification_cfg_path = Path(args.verification_config).resolve()
        if not verification_cfg_path.exists():
            raise FileNotFoundError(f"verification config not found: {verification_cfg_path}")
        with verification_cfg_path.open("r") as fh:
            verification_cfg = yaml.safe_load(fh)
        print(f"Using verification plotting parameters from: {verification_cfg_path}")
    else:
        print("Using verification plotting parameters from first run config.yaml in campaign runs.")

    ok, campaign, payload = _restart_recollate_if_possible(campaign_dir, qois)
    if ok:
        print("Restart recollation path succeeded.")
        plot_folder = campaign_dir / "reanalysis_restart_outputs" / "plots"
        sampler = campaign.get_active_sampler()
        _plot_reanalysis(payload, qois, run_dirs, plot_folder)
        _plot_sobols_reanalysis(payload, qois, run_dirs, plot_folder)
        _plot_pce_pdf_reconstruction_last_time(payload, qois, run_dirs, plot_folder, sampler)
        _plot_last_time_sobol_bars_center_and_flux(payload, qois, run_dirs, plot_folder)
        _plot_cj1959_verification(verification_cfg, payload, qois, run_dirs, plot_folder)
        _generate_uq_dashboard(campaign_dir / "runs", plot_folder / "uq_campaign_dashboard.html")
        print(f"Reanalysis outputs: {campaign_dir / 'reanalysis_restart_outputs'}")
    else:
        print(f"Restart recollation unavailable in this EasyVVUQ version: {type(payload).__name__}: {payload}")
        rebuilt_dir = _fallback_rebuild_and_analyse(campaign_dir, qois, cfg, verification_cfg=verification_cfg)
        print(f"Fallback reanalysis outputs: {rebuilt_dir}")


if __name__ == "__main__":
    main()
