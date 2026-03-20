#!/usr/bin/env python3
"""
UQ Postprocessing Script for FESTIM EasyVVUQ Campaigns.

This script performs postprocessing on completed UQ campaigns by loading
results from either:
  - An EasyVVUQ campaign database (campaign.db / .json)
  - A folder of individual run results

It regenerates plots into a dedicated, timestamped folder and saves a
log file with all computed statistics.

Usage:
    python uq_postprocessing.py --db path/to/campaign.db
    python uq_postprocessing.py --runs-dir path/to/runs_folder --config path/to/config.yaml
    python uq_postprocessing.py --results-pickle path/to/analysis_results.pickle

Created by: Yehor Yudin
"""

import argparse
import os
import sys
import logging
import pickle
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid display connection issues
import matplotlib.pyplot as plt

# Add parent directory to path for custom imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Add uq directory to path for local imports
uq_dir = os.path.dirname(os.path.abspath(__file__))
if uq_dir not in sys.path:
    sys.path.insert(0, uq_dir)

from util.utils import load_config, add_timestamp_to_filename
from util.plotting import UQPlotter


def setup_logging(log_folder, timestamp):
    """
    Set up logging to both console and a log file in the output folder.

    Args:
        log_folder (str): Folder where the log file will be saved.
        timestamp (str): Timestamp string for the log file name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_filename = os.path.join(log_folder, f"uq_postprocessing_log_{timestamp}.txt")

    logger = logging.getLogger("uq_postprocessing")
    logger.setLevel(logging.DEBUG)

    # File handler - captures all output
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file created: {log_filename}")
    return logger


def create_output_folder(prefix="uq_postprocessing_results", timestamp=None):
    """
    Create an unambiguously named output folder with a timestamp.

    Args:
        prefix (str): Prefix for the folder name.
        timestamp (str, optional): Timestamp string. Generated if None.

    Returns:
        tuple: (folder_name, timestamp)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    folder_name = f"{prefix}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name, timestamp


def log_statistics_from_results(logger, results, qois):
    """
    Log all available statistics from the UQ analysis results.

    Args:
        logger (logging.Logger): Logger instance.
        results: EasyVVUQ analysis results object.
        qois (list[str]): List of quantity of interest names.
    """
    logger.info("=" * 70)
    logger.info("UQ STATISTICS SUMMARY")
    logger.info("=" * 70)

    for qoi in qois:
        logger.info(f"\n--- Statistics for QoI: {qoi} ---")

        # Describe basic statistics
        stat_names = ['mean', 'std']
        for stat_name in stat_names:
            try:
                values = results.describe(qoi, stat_name)
                if values is not None:
                    if hasattr(values, '__len__') and len(values) > 0:
                        logger.info(f"  {stat_name}: min={np.min(values):.6e}, max={np.max(values):.6e}, "
                                    f"avg={np.mean(values):.6e}")
                        logger.debug(f"  {stat_name} (full): {values}")
                    else:
                        logger.info(f"  {stat_name}: {values}")
            except Exception as e:
                logger.debug(f"  {stat_name}: not available ({e})")

        # Try percentiles
        for pct in ['1%', '10%', 'median', '90%', '99%']:
            try:
                values = results.describe(qoi, pct)
                if values is not None and not np.all(values == 0):
                    if hasattr(values, '__len__') and len(values) > 0:
                        logger.info(f"  {pct}: min={np.min(values):.6e}, max={np.max(values):.6e}")
                    else:
                        logger.info(f"  {pct}: {values}")
            except Exception:
                pass

        # Try Sobol indices
        try:
            sobols_first = results.sobols_first(qoi)
            if sobols_first:
                non_zero = {k: v for k, v in sobols_first.items()
                            if v is not None and not np.all(np.array(v) == 0)}
                if non_zero:
                    logger.info(f"  Sobol first-order indices:")
                    for param, vals in non_zero.items():
                        vals_arr = np.array(vals)
                        logger.info(f"    {param}: min={np.min(vals_arr):.6e}, "
                                    f"max={np.max(vals_arr):.6e}, avg={np.mean(vals_arr):.6e}")
        except Exception:
            pass

        # Try derivative-based sensitivity
        try:
            derivs = results._get_derivatives_first(qoi)
            if derivs:
                logger.info(f"  Derivative-based sensitivity indices:")
                for param, vals in derivs.items():
                    if vals is not None:
                        vals_arr = np.array(vals)
                        logger.info(f"    {param}: min={np.min(vals_arr):.6e}, "
                                    f"max={np.max(vals_arr):.6e}, avg={np.mean(vals_arr):.6e}")
        except Exception:
            pass

    logger.info("=" * 70)
    logger.info("END OF STATISTICS SUMMARY")
    logger.info("=" * 70)


def postprocess_from_db(db_path, timestamp):
    """
    Postprocess UQ results from an EasyVVUQ campaign database.

    Args:
        db_path (str): Path to the campaign database file.
        timestamp (str): Timestamp for output naming.

    Returns:
        int: 0 on success.
    """
    import easyvvuq as uq

    # Create output folder
    output_folder, timestamp = create_output_folder(
        prefix="uq_postprocessing_results", timestamp=timestamp
    )
    logger = setup_logging(output_folder, timestamp)

    logger.info(f"Loading campaign from database: {db_path}")

    # Load campaign from database
    campaign = uq.Campaign(state_file=db_path)
    logger.info(f"Campaign loaded: {campaign.campaign_dir}")

    # Get the last analysis results
    results = campaign.get_last_analysis()
    if results is None:
        logger.error("No analysis results found in the campaign database. "
                      "Please run analysis before postprocessing.")
        return 1

    # Get QoI names from results
    qoi_names = list(results.qois)
    logger.info(f"QoIs found: {qoi_names}")

    # Filter out 'x' from QoIs for plotting
    plot_qois = [q for q in qoi_names if q != 'x']

    # Read vertices
    try:
        rs = results.describe('x', 'mean')
    except Exception:
        rs = None

    if rs is None:
        logger.warning("No vertex data ('x') found. Using index-based x-axis.")
        if plot_qois:
            try:
                rs = np.arange(len(results.describe(plot_qois[0], 'mean')))
            except Exception:
                rs = np.arange(10)

    # Log all statistics
    log_statistics_from_results(logger, results, plot_qois)

    # Get runs info from campaign database
    try:
        runs_info = list(campaign.campaign_db.runs())
        logger.info(f"Number of runs in campaign: {len(runs_info)}")
    except Exception:
        runs_info = None

    # Plot results
    logger.info(f"Generating plots in folder: {output_folder}")
    uqplotter = UQPlotter()

    try:
        uqplotter.plot_stats_vs_r(results, plot_qois, output_folder, timestamp,
                                   rs=rs, runs_info=runs_info)
        logger.info("Statistical plots generated successfully.")
    except Exception as e:
        logger.warning(f"Could not generate standard plots: {e}")
        # Try correlated plots as fallback
        try:
            distributions_keys = list(results.sobols_first(plot_qois[0]).keys()) if plot_qois else []
            distributions = {k: None for k in distributions_keys}
            uqplotter.plot_stats_correlated(results, distributions, plot_qois,
                                             output_folder, timestamp, rs=rs)
            logger.info("Correlated statistical plots generated successfully.")
        except Exception as e2:
            logger.error(f"Could not generate correlated plots either: {e2}")

    logger.info(f"Postprocessing completed. Output folder: {output_folder}")
    return 0


def postprocess_from_pickle(pickle_path, timestamp):
    """
    Postprocess UQ results from a saved pickle file of analysis results.

    Args:
        pickle_path (str): Path to the pickled analysis results.
        timestamp (str): Timestamp for output naming.

    Returns:
        int: 0 on success.
    """
    # Create output folder
    output_folder, timestamp = create_output_folder(
        prefix="uq_postprocessing_results", timestamp=timestamp
    )
    logger = setup_logging(output_folder, timestamp)

    logger.info(f"Loading analysis results from pickle: {pickle_path}")

    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)

    logger.info("Analysis results loaded successfully.")

    # Get QoI names from results
    try:
        qoi_names = list(results.qois)
    except AttributeError:
        logger.error("Loaded object does not have 'qois' attribute. "
                      "Ensure the pickle file contains EasyVVUQ analysis results.")
        return 1

    logger.info(f"QoIs found: {qoi_names}")
    plot_qois = [q for q in qoi_names if q != 'x']

    # Read vertices
    try:
        rs = results.describe('x', 'mean')
    except Exception:
        rs = None

    if rs is None:
        logger.warning("No vertex data ('x') found. Using index-based x-axis.")
        if plot_qois:
            try:
                rs = np.arange(len(results.describe(plot_qois[0], 'mean')))
            except Exception:
                rs = np.arange(10)

    # Log all statistics
    log_statistics_from_results(logger, results, plot_qois)

    # Plot results
    logger.info(f"Generating plots in folder: {output_folder}")
    uqplotter = UQPlotter()

    try:
        uqplotter.plot_stats_vs_r(results, plot_qois, output_folder, timestamp,
                                   rs=rs, runs_info=None)
        logger.info("Statistical plots generated successfully.")
    except Exception as e:
        logger.warning(f"Could not generate standard plots: {e}")
        try:
            distributions_keys = list(results.sobols_first(plot_qois[0]).keys()) if plot_qois else []
            distributions = {k: None for k in distributions_keys}
            uqplotter.plot_stats_correlated(results, distributions, plot_qois,
                                             output_folder, timestamp, rs=rs)
            logger.info("Correlated statistical plots generated successfully.")
        except Exception as e2:
            logger.error(f"Could not generate correlated plots either: {e2}")

    logger.info(f"Postprocessing completed. Output folder: {output_folder}")
    return 0


def postprocess_from_runs_dir(runs_dir, config_path, timestamp):
    """
    Postprocess UQ results from a directory of individual run outputs.

    Reads CSV/TXT result files from each run subdirectory, computes
    basic statistics (mean, std, percentiles), generates plots, and
    saves a log with all statistics.

    Args:
        runs_dir (str): Path to folder containing individual run directories.
        config_path (str): Path to the YAML config file (for QoI definitions).
        timestamp (str): Timestamp for output naming.

    Returns:
        int: 0 on success.
    """
    import csv

    # Create output folder
    output_folder, timestamp = create_output_folder(
        prefix="uq_postprocessing_results", timestamp=timestamp
    )
    logger = setup_logging(output_folder, timestamp)

    logger.info(f"Loading runs from directory: {runs_dir}")
    logger.info(f"Config file: {config_path}")

    # Load config for QoI definitions
    config = load_config(config_path)
    if config is None:
        logger.error(f"Failed to load config file: {config_path}")
        return 1

    # Find run subdirectories
    run_dirs = sorted([
        os.path.join(runs_dir, d) for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    ])
    logger.info(f"Found {len(run_dirs)} run directories.")

    if not run_dirs:
        logger.error("No run directories found.")
        return 1

    # Read results from each run
    # Look for result files (CSV or TXT)
    all_data = {}
    for run_dir in run_dirs:
        result_files = []
        results_subdir = os.path.join(run_dir, "results")
        search_dir = results_subdir if os.path.isdir(results_subdir) else run_dir

        for f in os.listdir(search_dir):
            if f.endswith(('.csv', '.txt')):
                result_files.append(os.path.join(search_dir, f))

        if not result_files:
            logger.warning(f"No result files found in {run_dir}, skipping.")
            continue

        for result_file in result_files:
            basename = os.path.basename(result_file)
            if basename not in all_data:
                all_data[basename] = []

            try:
                data = np.loadtxt(result_file, delimiter=None, comments='#')
                # If whitespace parsing produced a 1D array, the file may
                # use comma delimiters instead. Retry with comma delimiter.
                if data.ndim == 1:
                    data = np.loadtxt(result_file, delimiter=',', comments='#')
                all_data[basename].append(data)
                logger.debug(f"  Loaded {result_file} shape={data.shape}")
            except Exception as e:
                # Try reading as CSV with header
                try:
                    with open(result_file, 'r') as fh:
                        reader = csv.reader(fh)
                        header = next(reader)
                        rows = [list(map(float, row)) for row in reader if row]
                    data = np.array(rows)
                    all_data[basename].append(data)
                    logger.debug(f"  Loaded {result_file} via CSV reader, shape={data.shape}")
                except Exception as e2:
                    logger.warning(f"  Could not load {result_file}: {e}, {e2}")

    if not all_data:
        logger.error("No data could be loaded from run directories.")
        return 1

    # Compute statistics for each result file type
    logger.info("=" * 70)
    logger.info("UQ STATISTICS SUMMARY (from runs directory)")
    logger.info("=" * 70)

    uqplotter = UQPlotter()

    for filename, data_list in all_data.items():
        logger.info(f"\n--- Statistics for result file: {filename} ---")
        logger.info(f"  Number of runs loaded: {len(data_list)}")

        # Stack data: shape (n_runs, n_rows, n_cols)
        try:
            stacked = np.array(data_list)
        except ValueError:
            logger.warning(f"  Runs have inconsistent shapes, skipping {filename}.")
            continue

        logger.info(f"  Data shape per run: {stacked.shape[1:]}")

        # Compute statistics along the runs axis (axis=0)
        mean_vals = np.mean(stacked, axis=0)
        std_vals = np.std(stacked, axis=0)
        median_vals = np.median(stacked, axis=0)
        pct_10 = np.percentile(stacked, 10, axis=0)
        pct_90 = np.percentile(stacked, 90, axis=0)
        pct_01 = np.percentile(stacked, 1, axis=0)
        pct_99 = np.percentile(stacked, 99, axis=0)

        # Log per-column statistics
        n_cols = mean_vals.shape[1] if mean_vals.ndim > 1 else 1
        for col_idx in range(n_cols):
            col_mean = mean_vals[:, col_idx] if mean_vals.ndim > 1 else mean_vals
            col_std = std_vals[:, col_idx] if std_vals.ndim > 1 else std_vals

            logger.info(f"  Column {col_idx}:")
            logger.info(f"    Mean:   min={np.min(col_mean):.6e}, max={np.max(col_mean):.6e}, avg={np.mean(col_mean):.6e}")
            logger.info(f"    Std:    min={np.min(col_std):.6e}, max={np.max(col_std):.6e}, avg={np.mean(col_std):.6e}")
            eps = np.finfo(float).tiny  # smallest positive float, avoids division by zero
            logger.info(f"    CoV:    avg={np.mean(col_std / (np.abs(col_mean) + eps)):.6e}")

        # Generate plots for this result file

        name_stem = os.path.splitext(filename)[0]
        for col_idx in range(1, n_cols):  # Skip first column (typically x/coordinates)
            col_mean = mean_vals[:, col_idx] if mean_vals.ndim > 1 else mean_vals
            col_std = std_vals[:, col_idx] if std_vals.ndim > 1 else std_vals
            col_p10 = pct_10[:, col_idx] if pct_10.ndim > 1 else pct_10
            col_p90 = pct_90[:, col_idx] if pct_90.ndim > 1 else pct_90
            x_vals = mean_vals[:, 0] if mean_vals.ndim > 1 else np.arange(len(col_mean))

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x_vals, col_mean, label='Mean')
            ax.fill_between(x_vals, col_mean - col_std, col_mean + col_std,
                            alpha=0.3, label=r'$\pm 1\sigma$')
            ax.fill_between(x_vals, col_p10, col_p90,
                            alpha=0.1, label='10%-90%')
            ax.set_xlabel("Coordinate")
            ax.set_ylabel(f"Column {col_idx}")
            ax.set_title(f"Statistics for {name_stem}, column {col_idx}")
            ax.legend(loc='best')
            ax.grid(True)

            plot_filename = add_timestamp_to_filename(
                f"{name_stem}_col{col_idx}_stats.pdf", timestamp
            )
            fig.savefig(os.path.join(output_folder, plot_filename))
            plt.close()
            logger.info(f"  Plot saved: {plot_filename}")

    logger.info("=" * 70)
    logger.info("END OF STATISTICS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Postprocessing completed. Output folder: {output_folder}")
    return 0


def main():
    """Main entry point for the postprocessing script."""
    parser = argparse.ArgumentParser(
        description="UQ Postprocessing: Generate plots and statistics from "
                    "EasyVVUQ campaign results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --db path/to/campaign_state.json
  %(prog)s --results-pickle path/to/analysis_results.pickle
  %(prog)s --runs-dir path/to/runs/ --config path/to/config.yaml
        """
    )

    # Input source (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '--db', type=str,
        help='Path to EasyVVUQ campaign state file (JSON or DB).'
    )
    source.add_argument(
        '--results-pickle', type=str,
        help='Path to pickled analysis results file.'
    )
    source.add_argument(
        '--runs-dir', type=str,
        help='Path to directory containing individual run subdirectories.'
    )

    parser.add_argument(
        '--config', '-c', type=str, default=None,
        help='Path to YAML config file (required with --runs-dir).'
    )

    args = parser.parse_args()

    # Generate a common timestamp for this postprocessing session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.db:
        return postprocess_from_db(args.db, timestamp)
    elif args.results_pickle:
        return postprocess_from_pickle(args.results_pickle, timestamp)
    elif args.runs_dir:
        if not args.config:
            parser.error("--config is required when using --runs-dir")
        return postprocess_from_runs_dir(args.runs_dir, args.config, timestamp)

    return 1


if __name__ == "__main__":
    sys.exit(main())
