#!/usr/bin/env python3
"""
Analyse dependence of UQ outputs on PCE polynomial order p.

The script reads existing reanalysis outputs from FESTIM EasyVVUQ campaign
folders and produces:
- Sobol first-order indices as a function of p
- Mean concentration at core for the latest transient time as a function of p
- Mean flux at the latest available flux time (proxy for steady state) vs p
- Standard deviations of the same physical QoIs as a function of p

Usage example
-------------
PYTHONPATH=src python -m festim_niuq.uq.polynomial_order_analysis \
  --campaign 1:festim_campaign_20260602_131019_lbqg4pjt \
  --campaign 2:festim_campaign_20260602_145158_h04c6132 \
  --campaign 3:festim_campaign_20260601_175742_z_h2jk8i
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_transient_time(qoi: str) -> float | None:
    if not isinstance(qoi, str):
        return None
    if not (qoi.startswith("t=") and qoi.endswith("s")):
        return None
    try:
        return float(qoi.split("=", 1)[1].rstrip("s"))
    except Exception:
        return None


def _parse_flux_time(qoi: str) -> float | None:
    if not isinstance(qoi, str):
        return None
    if not (qoi.startswith("flux_t=") and qoi.endswith("s")):
        return None
    try:
        return float(qoi.split("=", 1)[1].rstrip("s"))
    except Exception:
        return None


def _to_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def _find_latest_reanalysis_dir(campaign_dir: Path) -> Path:
    candidates = [p for p in campaign_dir.glob("reanalysis_rebuild_*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No reanalysis_rebuild_* directory found in {campaign_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_results_pickle(reanalysis_dir: Path) -> Path:
    candidates = sorted(reanalysis_dir.glob("analysis_results_uq_campaign_*.pickle"))
    if not candidates:
        raise FileNotFoundError(f"No analysis_results_uq_campaign_*.pickle in {reanalysis_dir}")
    return candidates[-1]


def _extract_qois_from_results(results: Any) -> list[str]:
    names = []
    for attr in ("qois", "_qois"):
        if hasattr(results, attr):
            try:
                names = list(getattr(results, attr))
            except Exception:
                names = []
            if names:
                return names
    return names


@dataclass
class POrderSnapshot:
    p: int
    campaign_dir: Path
    reanalysis_dir: Path
    results_pickle: Path
    q_conc_last: str
    q_flux_last: str
    t_conc_last: float
    t_flux_last: float
    mean_conc_core_last: float
    std_conc_core_last: float
    mean_flux_steady: float
    std_flux_steady: float
    sobol_conc_core_last: dict[str, float]
    sobol_flux_steady: dict[str, float]


def _collect_snapshot(p: int, campaign_dir: Path) -> POrderSnapshot:
    reanalysis_dir = _find_latest_reanalysis_dir(campaign_dir)
    results_pickle = _find_results_pickle(reanalysis_dir)

    with results_pickle.open("rb") as fh:
        results = pickle.load(fh)

    qois = _extract_qois_from_results(results)
    if not qois:
        raise RuntimeError(f"Could not infer QoIs from analysis results in {results_pickle}")

    transient = sorted([(t, q) for q in qois for t in [_parse_transient_time(q)] if t is not None], key=lambda x: x[0])
    fluxes = sorted([(t, q) for q in qois for t in [_parse_flux_time(q)] if t is not None], key=lambda x: x[0])

    if not transient:
        raise RuntimeError(f"No transient concentration QoIs found in {results_pickle}")
    if not fluxes:
        raise RuntimeError(f"No flux QoIs found in {results_pickle}")

    t_conc_last, q_conc_last = transient[-1]
    t_flux_last, q_flux_last = fluxes[-1]

    x = np.asarray(results.describe("x", "mean"), dtype=float)
    if x.size == 0:
        raise RuntimeError(f"No x-coordinate data in {results_pickle}")
    i_core = int(np.argmin(np.abs(x)))

    mean_conc = np.asarray(results.describe(q_conc_last, "mean"), dtype=float)
    std_conc = np.asarray(results.describe(q_conc_last, "std"), dtype=float)

    mean_conc_core_last = float(mean_conc.reshape(-1)[i_core])
    std_conc_core_last = float(std_conc.reshape(-1)[i_core])

    mean_flux_steady = _to_scalar(results.describe(q_flux_last, "mean"))
    std_flux_steady = _to_scalar(results.describe(q_flux_last, "std"))

    s1_conc_raw = results.sobols_first(q_conc_last) or {}
    s1_flux_raw = results.sobols_first(q_flux_last) or {}

    sobol_conc_core_last = {
        param: float(np.asarray(values, dtype=float).reshape(-1)[i_core])
        for param, values in s1_conc_raw.items()
        if values is not None and np.asarray(values).size > i_core
    }
    sobol_flux_steady = {
        param: _to_scalar(values)
        for param, values in s1_flux_raw.items()
        if values is not None
    }

    return POrderSnapshot(
        p=p,
        campaign_dir=campaign_dir,
        reanalysis_dir=reanalysis_dir,
        results_pickle=results_pickle,
        q_conc_last=q_conc_last,
        q_flux_last=q_flux_last,
        t_conc_last=float(t_conc_last),
        t_flux_last=float(t_flux_last),
        mean_conc_core_last=mean_conc_core_last,
        std_conc_core_last=std_conc_core_last,
        mean_flux_steady=mean_flux_steady,
        std_flux_steady=std_flux_steady,
        sobol_conc_core_last=sobol_conc_core_last,
        sobol_flux_steady=sobol_flux_steady,
    )


def _write_summary_csv(rows: list[POrderSnapshot], outdir: Path) -> Path:
    out_csv = outdir / "polynomial_order_summary.csv"
    with out_csv.open("w") as fh:
        fh.write(
            "p,campaign_dir,reanalysis_dir,q_conc_last,t_conc_last_s,mean_conc_core_last,std_conc_core_last,q_flux_last,t_flux_last_s,mean_flux_steady,std_flux_steady\n"
        )
        for row in rows:
            fh.write(
                f"{row.p},{row.campaign_dir},{row.reanalysis_dir},{row.q_conc_last},{row.t_conc_last:.6g},{row.mean_conc_core_last:.12e},{row.std_conc_core_last:.12e},{row.q_flux_last},{row.t_flux_last:.6g},{row.mean_flux_steady:.12e},{row.std_flux_steady:.12e}\n"
            )
    return out_csv


def _plot_sobol_vs_p(rows: list[POrderSnapshot], outdir: Path, which: str) -> Path:
    if which == "conc":
        title = "First-order Sobol at core, latest concentration time"
        getter = lambda r: r.sobol_conc_core_last
        out = outdir / "sobol_first_concentration_core_last_vs_p.png"
    elif which == "flux":
        title = "First-order Sobol at latest flux time"
        getter = lambda r: r.sobol_flux_steady
        out = outdir / "sobol_first_flux_steady_vs_p.png"
    else:
        raise ValueError("which must be 'conc' or 'flux'")

    params = sorted({k for r in rows for k in getter(r).keys()})
    p_vals = np.asarray([r.p for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    for param in params:
        y = [getter(r).get(param, np.nan) for r in rows]
        style = _param_style(param)
        ax.plot(p_vals, y, label=f"S1({param})", **style)

    ax.set_title(title.replace(" p", " $p$"), fontsize=14)
    ax.set_xlabel("Polynomial order $p$", fontsize=13)
    ax.set_ylabel("Sobol index", fontsize=13)
    ax.set_xticks(sorted(set(int(v) for v in p_vals)))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(loc="lower right", fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_mean_vs_p(rows: list[POrderSnapshot], outdir: Path) -> Path:
    p_vals = np.asarray([r.p for r in rows], dtype=float)
    conc_mean = np.asarray([r.mean_conc_core_last for r in rows], dtype=float)
    flux_mean = np.asarray([r.mean_flux_steady for r in rows], dtype=float)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    axs[0].plot(p_vals, conc_mean, marker="o", color="black", linewidth=2.0, label="Mean concentration")
    axs[0].set_title("Mean concentration at core (latest time)", fontsize=14)
    axs[0].set_xlabel("Polynomial order $p$", fontsize=13)
    axs[0].set_ylabel("Concentration mean", fontsize=13)
    axs[0].set_xticks(sorted(set(int(v) for v in p_vals)))
    axs[0].grid(True, alpha=0.3)
    axs[0].tick_params(axis="both", labelsize=12)
    axs[0].legend(loc="lower right", fontsize=11)

    axs[1].plot(p_vals, flux_mean, marker="o", color="red", linewidth=2.0, label="Mean outward flux")
    axs[1].set_title("Mean flux at latest flux time", fontsize=14)
    axs[1].set_xlabel("Polynomial order $p$", fontsize=13)
    axs[1].set_ylabel("Flux mean", fontsize=13)
    axs[1].set_xticks(sorted(set(int(v) for v in p_vals)))
    axs[1].grid(True, alpha=0.3)
    axs[1].tick_params(axis="both", labelsize=12)
    axs[1].legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    out = outdir / "mean_physical_qois_vs_p.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _plot_std_vs_p(rows: list[POrderSnapshot], outdir: Path) -> Path:
    p_vals = np.asarray([r.p for r in rows], dtype=float)
    conc_std = np.asarray([r.std_conc_core_last for r in rows], dtype=float)
    flux_std = np.asarray([r.std_flux_steady for r in rows], dtype=float)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))

    axs[0].plot(p_vals, conc_std, marker="o", color="black", linewidth=2.0, label="STD concentration")
    axs[0].set_title("STD concentration at core (latest time)", fontsize=14)
    axs[0].set_xlabel("Polynomial order $p$", fontsize=13)
    axs[0].set_ylabel("Concentration STD", fontsize=13)
    axs[0].set_xticks(sorted(set(int(v) for v in p_vals)))
    axs[0].grid(True, alpha=0.3)
    axs[0].tick_params(axis="both", labelsize=12)
    axs[0].legend(loc="lower right", fontsize=11)

    axs[1].plot(p_vals, flux_std, marker="o", color="red", linewidth=2.0, label="STD outward flux")
    axs[1].set_title("STD flux at latest flux time", fontsize=14)
    axs[1].set_xlabel("Polynomial order $p$", fontsize=13)
    axs[1].set_ylabel("Flux STD", fontsize=13)
    axs[1].set_xticks(sorted(set(int(v) for v in p_vals)))
    axs[1].grid(True, alpha=0.3)
    axs[1].tick_params(axis="both", labelsize=12)
    axs[1].legend(loc="lower right", fontsize=11)

    fig.tight_layout()
    out = outdir / "std_physical_qois_vs_p.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def _style_axes_for_paper(ax, p_vals: np.ndarray) -> None:
    ax.set_xticks(sorted(set(int(v) for v in p_vals)))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)


def _param_latex(param: str) -> str:
    if param == "D_0":
        return r"D_0"
    if param == "G":
        return r"G"
    return param


def _param_style(param: str) -> dict[str, Any]:
    # Keep G visually consistent with upper-panel STD style.
    if param == "G":
        return {
            "marker": "s",
            "linestyle": "--",
            "color": "red",
            "linewidth": 2.0,
            "markersize": 6,
        }
    return {
        "marker": "o",
        "linestyle": "-",
        "color": "black",
        "linewidth": 2.0,
        "markersize": 6,
    }


def _plot_dashboard_2x2(rows: list[POrderSnapshot], outdir: Path) -> tuple[Path, Path]:
    """Create a publication-friendly 2x2 dashboard for p-dependence trends."""
    p_vals = np.asarray([r.p for r in rows], dtype=float)
    conc_mean = np.asarray([r.mean_conc_core_last for r in rows], dtype=float)
    flux_mean = np.asarray([r.mean_flux_steady for r in rows], dtype=float)
    conc_std = np.asarray([r.std_conc_core_last for r in rows], dtype=float)
    flux_std = np.asarray([r.std_flux_steady for r in rows], dtype=float)

    fig, axs = plt.subplots(2, 2, figsize=(11.69, 8.27), constrained_layout=True)

    # (1,1) Concentration with std on twin y-axis
    conc_ax = axs[0, 0]
    conc_std_ax = conc_ax.twinx()
    conc_ax.plot(
        p_vals,
        conc_mean,
        marker="o",
        linewidth=2.2,
        markersize=6,
        color="black",
        label="Mean",
    )
    conc_std_ax.plot(
        p_vals,
        conc_std,
        marker="s",
        linewidth=2.0,
        markersize=5.5,
        color="red",
        linestyle="--",
        label="STD",
    )
    conc_ax.set_title("Mobile concentration at centre vs polynomial order $p$", fontsize=14)
    conc_ax.set_xlabel("Polynomial order $p$", fontsize=13)
    conc_ax.set_ylabel("Concentration [arb. units]", fontsize=13)
    conc_std_ax.set_ylabel("Concentration STD [arb. units]", fontsize=13)
    conc_ax.yaxis.label.set_color("black")
    conc_std_ax.yaxis.label.set_color("red")
    _style_axes_for_paper(conc_ax, p_vals)
    conc_ax.tick_params(axis="y", labelsize=12, colors="black")
    conc_std_ax.tick_params(axis="y", labelsize=12, colors="red")
    h1, l1 = conc_ax.get_legend_handles_labels()
    h2, l2 = conc_std_ax.get_legend_handles_labels()
    conc_ax.legend(h1 + h2, l1 + l2, fontsize=11, loc="lower right")

    # (1,2) Outward flux with std on twin y-axis
    flux_ax = axs[0, 1]
    flux_std_ax = flux_ax.twinx()
    flux_ax.plot(
        p_vals,
        flux_mean,
        marker="o",
        linewidth=2.2,
        markersize=6,
        color="black",
        label="Mean",
    )
    flux_std_ax.plot(
        p_vals,
        flux_std,
        marker="s",
        linewidth=2.0,
        markersize=5.5,
        color="red",
        linestyle="--",
        label="STD",
    )
    flux_ax.set_title("Outward flux at final time vs polynomial order $p$", fontsize=14)
    flux_ax.set_xlabel("Polynomial order $p$", fontsize=13)
    flux_ax.set_ylabel("Outward flux [arb. units]", fontsize=13)
    flux_std_ax.set_ylabel("Outward flux STD [arb. units]", fontsize=13)
    flux_ax.yaxis.label.set_color("black")
    flux_std_ax.yaxis.label.set_color("red")
    _style_axes_for_paper(flux_ax, p_vals)
    flux_ax.tick_params(axis="y", labelsize=12, colors="black")
    flux_std_ax.tick_params(axis="y", labelsize=12, colors="red")
    h1, l1 = flux_ax.get_legend_handles_labels()
    h2, l2 = flux_std_ax.get_legend_handles_labels()
    flux_ax.legend(h1 + h2, l1 + l2, fontsize=11, loc="lower right")

    # (2,1) Sobol concentration
    conc_params = sorted({k for r in rows for k in r.sobol_conc_core_last.keys()})
    for param in conc_params:
        y = [r.sobol_conc_core_last.get(param, np.nan) for r in rows]
        style = _param_style(param)
        axs[1, 0].plot(p_vals, y, label=rf"$S_1({_param_latex(param)})$", **style)
    axs[1, 0].set_title("Sobol indices for concentration at centre and final time", fontsize=14)
    axs[1, 0].set_xlabel("Polynomial order $p$", fontsize=13)
    axs[1, 0].set_ylabel(r"First-order Sobol index $S_1$", fontsize=13)
    axs[1, 0].set_ylim(-0.02, 1.05)
    _style_axes_for_paper(axs[1, 0], p_vals)
    axs[1, 0].legend(fontsize=11, loc="lower right")

    # (2,2) Sobol outward flux
    flux_params = sorted({k for r in rows for k in r.sobol_flux_steady.keys()})
    for param in flux_params:
        y = [r.sobol_flux_steady.get(param, np.nan) for r in rows]
        style = _param_style(param)
        axs[1, 1].plot(p_vals, y, label=rf"$S_1({_param_latex(param)})$", **style)
    axs[1, 1].set_title("Sobol indices for outward flux at final time", fontsize=14)
    axs[1, 1].set_xlabel("Polynomial order $p$", fontsize=13)
    axs[1, 1].set_ylabel(r"First-order Sobol index $S_1$", fontsize=13)
    axs[1, 1].set_ylim(-0.02, 1.05)
    _style_axes_for_paper(axs[1, 1], p_vals)
    axs[1, 1].legend(fontsize=11, loc="lower right")

    out_png = outdir / "poly_order_dashboard_2x2_a4.png"
    out_pdf = outdir / "poly_order_dashboard_2x2_a4.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png, out_pdf


def _parse_campaign_arg(value: str) -> tuple[int, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("Expected format '<p>:<campaign_dir>'")
    p_str, path_str = value.split(":", 1)
    try:
        p = int(p_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid p value '{p_str}'") from exc
    campaign_dir = Path(path_str).expanduser().resolve()
    if not campaign_dir.exists():
        raise argparse.ArgumentTypeError(f"Campaign dir does not exist: {campaign_dir}")
    return p, campaign_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Polynomial-order analysis for FESTIM UQ reanalysis outputs")
    parser.add_argument(
        "--campaign",
        action="append",
        required=True,
        help="Campaign mapping in format '<p>:<campaign_dir>'. Repeat for each p.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: polynomial_order_analysis_<timestamp>",
    )

    args = parser.parse_args()

    parsed = [_parse_campaign_arg(item) for item in args.campaign]
    parsed = sorted(parsed, key=lambda x: x[0])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd() / f"polynomial_order_analysis_{ts}"
    outdir.mkdir(parents=True, exist_ok=True)

    rows = [_collect_snapshot(p, campaign_dir) for p, campaign_dir in parsed]

    csv_path = _write_summary_csv(rows, outdir)
    sobol_conc_path = _plot_sobol_vs_p(rows, outdir, which="conc")
    sobol_flux_path = _plot_sobol_vs_p(rows, outdir, which="flux")
    mean_path = _plot_mean_vs_p(rows, outdir)
    std_path = _plot_std_vs_p(rows, outdir)
    dashboard_png, dashboard_pdf = _plot_dashboard_2x2(rows, outdir)

    print(f"Saved: {csv_path}")
    print(f"Saved: {sobol_conc_path}")
    print(f"Saved: {sobol_flux_path}")
    print(f"Saved: {mean_path}")
    print(f"Saved: {std_path}")
    print(f"Saved: {dashboard_png}")
    print(f"Saved: {dashboard_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
