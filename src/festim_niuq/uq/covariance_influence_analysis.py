"""
Covariance Influence Analysis Script for FESTIM UQ

This script analyses the influence of the covariance parameter (correlation
coefficient between uncertain parameters) on uncertainty quantification results.
It runs multiple correlated UQ campaigns across a range of covariance values
and collects first-order and total Sobol indices for each.

Supports both logarithmic and linear scales for the covariance scan.

Results are saved in a separate folder with a descriptor and timestamp in the name.

Usage:
    python covariance_influence_analysis.py --config config.yaml --scale log --n-points 5
    python covariance_influence_analysis.py --config config.yaml --scale linear --min-cov 0.01 --max-cov 0.9
    python covariance_influence_analysis.py --config config.yaml --scale log --p-order 3

Created by: Copilot Agent
"""

import argparse
import os
import sys
import json
import csv

import numpy as np

from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import chaospy as cp
import easyvvuq as uq

from easyvvuq.actions import Encode, Decode, ExecuteLocal, Actions, CreateRunDirectory
from easyvvuq.actions import QCGPJPool

# local imports
from .util.utils import load_config, add_timestamp_to_filename
from .util.Encoder import AdvancedYAMLEncoder
from .easyvvuq_festim_correlated import (
    define_parameter_uncertainty,
    define_festim_model_parameters,
    prepare_execution_command,
    run_uq_campaign,
)


def generate_covariance_values(scale, n_points, min_cov, max_cov):
    """
    Generate an array of covariance (correlation coefficient) values for scanning.

    Args:
        scale (str): Scale type — 'log' or 'linear'.
        n_points (int): Number of covariance values to generate.
        min_cov (float): Minimum covariance value (must be > 0 for log scale).
        max_cov (float): Maximum covariance value (must be < 1).

    Returns:
        np.ndarray: Array of covariance values.

    Raises:
        ValueError: If scale is not 'log' or 'linear', or if bounds are invalid.
    """
    if scale not in ("log", "linear"):
        raise ValueError(f"Scale must be 'log' or 'linear', got '{scale}'")
    if min_cov <= 0.0:
        raise ValueError(f"Minimum covariance must be > 0, got {min_cov}")
    if max_cov >= 1.0:
        raise ValueError(f"Maximum covariance must be < 1, got {max_cov}")
    if min_cov >= max_cov:
        raise ValueError(f"min_cov ({min_cov}) must be less than max_cov ({max_cov})")

    if scale == "log":
        cov_values = np.logspace(np.log10(min_cov), np.log10(max_cov), n_points)
    else:
        cov_values = np.linspace(min_cov, max_cov, n_points)

    return cov_values


def run_single_covariance_campaign(config, config_file, corr_value, p_order, fixed_params=None):
    """
    Run a single correlated UQ campaign for a given covariance value using PCE.

    Args:
        config (dict): Configuration dictionary.
        config_file (str): Path to YAML config file.
        corr_value (float): Correlation coefficient value.
        p_order (int): Polynomial order for PCE.
        fixed_params (dict, optional): Fixed parameters.

    Returns:
        tuple: (results, qois, distributions, param_names) from the analysis.
    """
    # Define model parameters and QoIs
    parameters, qois = define_festim_model_parameters(config)

    # Define uncertain parameter distributions with this covariance value
    distributions, distributions_joint = define_parameter_uncertainty(config, corr=corr_value)

    param_names = list(distributions.keys())

    # Set up encoder
    encoder = AdvancedYAMLEncoder(
        template_fname=config_file,
        target_filename="config.yaml",
        parameter_map={
            "D_0": "materials.D_0.mean",
            "thermal_conductivity": "materials.thermal_conductivity.mean",
            "length": "geometry.domains.length",
        },
        type_conversions={
            "D_0": float,
            "thermal_conductivity": float,
            "E_D": float,
            "T": float,
            "source_concentration_value": float,
            "left_bc_concentration_value": float,
            "right_bc_concentration_value": float,
            "length": float,
        },
        fixed_parameters=fixed_params,
    )

    # Set up decoder
    output_dir = config.get("simulation", {}).get("output_directory", "results")
    decoder = uq.decoders.SimpleCSV(
        target_filename=f"{output_dir}/results_tritium_concentration.txt",
        output_columns=qois,
    )

    # Set up execution and actions
    execute = prepare_execution_command()
    actions = Actions(
        CreateRunDirectory("run_dir"),
        Encode(encoder),
        execute,
        Decode(decoder),
    )

    campaign_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign = uq.Campaign(
        name=f"cov_scan_{corr_value:.4f}_{campaign_timestamp}_",
        params=parameters,
        actions=actions,
    )

    # Use PCE sampler (supports correlated distributions and computes Sobol indices)
    sampler = uq.sampling.PCESampler(
        vary=distributions,
        distribution=distributions_joint,
        polynomial_order=p_order,
    )
    campaign.set_sampler(sampler)

    # Run campaign
    campaign, campaign_results = run_uq_campaign(campaign)

    # Analyse with PCE (computes proper Sobol indices)
    analysis = uq.analysis.PCEAnalysis(sampler=sampler, qoi_cols=qois)
    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    return results, qois, distributions, param_names


def collect_sobol_indices(results, qois, param_names):
    """
    Extract first-order and total Sobol indices from PCE analysis results.

    Args:
        results: PCEAnalysis results object.
        qois (list[str]): QoI names (including 'x').
        param_names (list[str]): Names of uncertain parameters.

    Returns:
        dict: Nested dictionary with structure:
            {qoi: {param: {'first': float, 'total': float}}}
            Values are the mean Sobol index across spatial points.
    """
    sobol_data = {}

    for qoi in qois[1:]:  # Skip 'x'
        sobol_data[qoi] = {}
        for param in param_names:
            try:
                s_first = results._get_sobols_first(qoi, param)
                s_total = results._get_sobols_total(qoi, param)

                # Compute mean across spatial points for a scalar summary
                s_first_mean = float(np.mean(s_first)) if s_first is not None else 0.0
                s_total_mean = float(np.mean(s_total)) if s_total is not None else 0.0

                sobol_data[qoi][param] = {
                    "first": s_first_mean,
                    "total": s_total_mean,
                }
            except Exception as e:
                print(f"  Warning: Could not extract Sobol indices for " f"{qoi}/{param}: {e}")
                sobol_data[qoi][param] = {"first": 0.0, "total": 0.0}

    return sobol_data


def save_scan_results(results_folder, cov_values, all_sobol_data, param_names, qois, scan_metadata):
    """
    Save covariance scan results to CSV and JSON files.

    Args:
        results_folder (str): Path to results folder.
        cov_values (np.ndarray): Array of covariance values scanned.
        all_sobol_data (list[dict]): List of Sobol data dicts, one per covariance value.
        param_names (list[str]): Parameter names.
        qois (list[str]): QoI names (including 'x').
        scan_metadata (dict): Metadata about the scan configuration.
    """
    # Save metadata
    metadata_file = os.path.join(results_folder, "scan_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(scan_metadata, f, indent=2, default=str)
    print(f"  Metadata saved to: {metadata_file}")

    # Save per-QoI CSV files
    for qoi in qois[1:]:
        csv_file = os.path.join(results_folder, f"sobol_indices_{qoi}.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)

            # Header
            header = ["covariance"]
            for param in param_names:
                header.extend([f"{param}_first", f"{param}_total"])
            writer.writerow(header)

            # Data rows
            for i, cov_val in enumerate(cov_values):
                row = [f"{cov_val:.6e}"]
                sobol_data = all_sobol_data[i]
                for param in param_names:
                    if qoi in sobol_data and param in sobol_data[qoi]:
                        row.append(f"{sobol_data[qoi][param]['first']:.6e}")
                        row.append(f"{sobol_data[qoi][param]['total']:.6e}")
                    else:
                        row.extend(["0.0", "0.0"])
                writer.writerow(row)

        print(f"  Sobol indices saved to: {csv_file}")

    # Save combined JSON results
    combined_file = os.path.join(results_folder, "sobol_indices_all.json")
    combined_data = {
        "covariance_values": cov_values.tolist(),
        "param_names": param_names,
        "qois": qois[1:],
        "sobol_indices": all_sobol_data,
    }
    with open(combined_file, "w") as f:
        json.dump(combined_data, f, indent=2, default=str)
    print(f"  Combined results saved to: {combined_file}")


def plot_sobol_vs_covariance(results_folder, cov_values, all_sobol_data, param_names, qois, scale, timestamp):
    """
    Plot first-order and total Sobol indices as a function of covariance.

    Args:
        results_folder (str): Path to results folder.
        cov_values (np.ndarray): Array of covariance values.
        all_sobol_data (list[dict]): Sobol data for each covariance value.
        param_names (list[str]): Parameter names.
        qois (list[str]): QoI names (including 'x').
        scale (str): 'log' or 'linear' — used for x-axis.
        timestamp (str): Timestamp for filename.
    """
    for qoi in qois[1:]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot first-order Sobol indices
        ax1 = axes[0]
        for param in param_names:
            values = [all_sobol_data[i].get(qoi, {}).get(param, {}).get("first", 0.0) for i in range(len(cov_values))]
            if scale == "log":
                ax1.semilogx(cov_values, values, "o-", label=param)
            else:
                ax1.plot(cov_values, values, "o-", label=param)

        ax1.set_xlabel("Correlation coefficient")
        ax1.set_ylabel("First-order Sobol index")
        ax1.set_title(f"First-order Sobol indices — {qoi}")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=-0.05, top=1.05)

        # Plot total Sobol indices
        ax2 = axes[1]
        for param in param_names:
            values = [all_sobol_data[i].get(qoi, {}).get(param, {}).get("total", 0.0) for i in range(len(cov_values))]
            if scale == "log":
                ax2.semilogx(cov_values, values, "s-", label=param)
            else:
                ax2.plot(cov_values, values, "s-", label=param)

        ax2.set_xlabel("Correlation coefficient")
        ax2.set_ylabel("Total Sobol index")
        ax2.set_title(f"Total Sobol indices — {qoi}")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=-0.05, top=1.05)

        plt.tight_layout()
        plot_file = os.path.join(results_folder, f"sobol_vs_covariance_{qoi}_{timestamp}.pdf")
        plt.savefig(plot_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot saved: {plot_file}")

    # Summary plot: all QoIs on one figure (first-order only)
    if len(qois) > 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        for qoi in qois[1:]:
            for param in param_names:
                values = [
                    all_sobol_data[i].get(qoi, {}).get(param, {}).get("first", 0.0) for i in range(len(cov_values))
                ]
                label = f"{param} @ {qoi}"
                if scale == "log":
                    ax.semilogx(cov_values, values, "o-", label=label, markersize=4)
                else:
                    ax.plot(cov_values, values, "o-", label=label, markersize=4)

        ax.set_xlabel("Correlation coefficient")
        ax.set_ylabel("First-order Sobol index")
        ax.set_title("First-order Sobol indices — all QoIs")
        ax.legend(fontsize="small", ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-0.05, top=1.05)

        plt.tight_layout()
        summary_file = os.path.join(results_folder, f"sobol_vs_covariance_summary_{timestamp}.pdf")
        plt.savefig(summary_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary plot saved: {summary_file}")


def run_covariance_scan(config_file, scale="log", n_points=5, min_cov=0.01, max_cov=0.9, p_order=2, fixed_params=None):
    """
    Main function: scan covariance values and collect Sobol indices.

    Args:
        config_file (str): Path to YAML configuration file.
        scale (str): 'log' or 'linear' scale for covariance values.
        n_points (int): Number of covariance values to scan.
        min_cov (float): Minimum covariance value.
        max_cov (float): Maximum covariance value.
        p_order (int): Polynomial order for PCE analysis.
        fixed_params (dict, optional): Fixed parameters for the campaigns.

    Returns:
        str: Path to the results folder.
    """
    print("\n" + "=" * 70)
    print("  COVARIANCE INFLUENCE ANALYSIS")
    print("=" * 70)

    # Load config
    config = load_config(config_file)
    if config is None:
        print("Error: Could not load configuration file.")
        return None

    # Generate covariance values
    cov_values = generate_covariance_values(scale, n_points, min_cov, max_cov)
    print(f"\nScan configuration:")
    print(f"  Scale: {scale}")
    print(f"  Covariance values ({n_points}): {cov_values}")
    print(f"  PCE polynomial order: {p_order}")

    # Create results folder with descriptor and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    descriptor = f"cov_scan_{scale}_{n_points}pts_p{p_order}"
    results_folder = f"{descriptor}_{timestamp}"
    os.makedirs(results_folder, exist_ok=True)
    print(f"\nResults folder: {results_folder}")

    # Scan metadata
    scan_metadata = {
        "timestamp": timestamp,
        "config_file": config_file,
        "scale": scale,
        "n_points": n_points,
        "min_cov": min_cov,
        "max_cov": max_cov,
        "p_order": p_order,
        "covariance_values": cov_values.tolist(),
        "fixed_params": fixed_params,
    }

    # Run campaigns
    all_sobol_data = []
    param_names = None
    qois = None

    for i, cov_val in enumerate(cov_values):
        print(f"\n--- Covariance scan [{i + 1}/{n_points}]: corr = {cov_val:.6f} ---")

        try:
            results, run_qois, distributions, run_param_names = run_single_covariance_campaign(
                config, config_file, cov_val, p_order, fixed_params=fixed_params
            )

            # Store param_names and qois from first successful run
            if param_names is None:
                param_names = run_param_names
                qois = run_qois

            # Collect Sobol indices
            sobol_data = collect_sobol_indices(results, run_qois, run_param_names)
            all_sobol_data.append(sobol_data)

            # Print summary for this covariance value
            for qoi in run_qois[1:]:
                for param in run_param_names:
                    s = sobol_data[qoi][param]
                    print(f"    {qoi} / {param}: S1={s['first']:.4f}, " f"ST={s['total']:.4f}")

        except Exception as e:
            print(f"  ERROR at corr={cov_val:.6f}: {e}")
            # Append placeholder data to keep alignment with cov_values
            if qois and param_names:
                empty = {qoi: {p: {"first": 0.0, "total": 0.0} for p in param_names} for qoi in qois[1:]}
                all_sobol_data.append(empty)
            else:
                all_sobol_data.append({})

    if param_names is None or qois is None:
        print("\nError: No successful campaigns completed. Cannot save results.")
        return results_folder

    # Check for failed campaigns
    n_failed = sum(1 for d in all_sobol_data if len(d) == 0)
    if n_failed > 0:
        print(
            f"\nNote: {n_failed} out of {len(cov_values)} campaigns failed. " f"Failed entries use zero Sobol indices."
        )

    # Save results
    print(f"\nSaving results to: {results_folder}")
    save_scan_results(results_folder, cov_values, all_sobol_data, param_names, qois, scan_metadata)

    # Generate plots
    print(f"\nGenerating plots...")
    plot_sobol_vs_covariance(results_folder, cov_values, all_sobol_data, param_names, qois, scale, timestamp)

    print(f"\n{'=' * 70}")
    print(f"  COVARIANCE SCAN COMPLETE")
    print(f"  Results in: {results_folder}")
    print(f"{'=' * 70}\n")

    return results_folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse the influence of the covariance parameter on "
        "Sobol indices in correlated UQ for FESTIM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Log-scale scan with 5 points, default PCE order
  python covariance_influence_analysis.py --config config.yaml --scale log

  # Linear-scale scan with 8 points
  python covariance_influence_analysis.py --config config.yaml --scale linear --n-points 8

  # Custom range and higher PCE accuracy
  python covariance_influence_analysis.py --config config.yaml --scale log \\
      --min-cov 0.001 --max-cov 0.8 --p-order 3

  # With fixed parameters
  python covariance_influence_analysis.py --config config.yaml --scale log \\
      --fixed-length 5e-4
""",
    )

    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to YAML configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--scale",
        "-s",
        default="log",
        choices=["log", "linear"],
        help="Scale for covariance scan: log or linear (default: log)",
    )
    parser.add_argument(
        "--n-points", "-n", type=int, default=5, help="Number of covariance values to scan (default: 5)"
    )
    parser.add_argument("--min-cov", type=float, default=0.01, help="Minimum covariance value (default: 0.01)")
    parser.add_argument("--max-cov", type=float, default=0.9, help="Maximum covariance value (default: 0.9)")
    parser.add_argument("--p-order", "-p", type=int, default=2, help="Polynomial order for PCE analysis (default: 2)")
    parser.add_argument("--fixed-length", type=float, default=None, help="Fixed sample length in meters (optional)")

    args = parser.parse_args()

    # Build fixed_params from CLI
    fixed_params = {}
    if args.fixed_length is not None:
        fixed_params["length"] = args.fixed_length

    run_covariance_scan(
        config_file=args.config,
        scale=args.scale,
        n_points=args.n_points,
        min_cov=args.min_cov,
        max_cov=args.max_cov,
        p_order=args.p_order,
        fixed_params=fixed_params if fixed_params else None,
    )
