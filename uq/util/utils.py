"""
Shared utility functions for the FESTIM-NIUQ UQ pipeline.

Provides configuration loading, file-name helpers, Python-environment
detection, execution validation, sensitivity-analysis persistence, and
heuristic absolute-tolerance estimation.
"""

import csv
import json
import os
import sys
import subprocess
import yaml
from pathlib import Path
import numpy as np

from datetime import datetime

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from joblib import dump, load


def load_config(config_file):
    """
    Load a YAML configuration file and return the parsed dictionary.

    Parameters
    ----------
    config_file : str
        Path to the YAML file.

    Returns
    -------
    dict or None
        Parsed configuration, or ``None`` if the file is missing or
        contains invalid YAML.
    """

    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        print(f" >> Configuration loaded: {config}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


def add_timestamp_to_filename(filename, timestamp=None):
    """
    Add timestamp to filename before the extension.

    Args:
        filename (str): Original filename
        timestamp (str, optional): Custom timestamp string. If None, uses current datetime.

    Returns:
        str: Filename with timestamp

    Example:
        add_timestamp_to_filename("results.hdf5") -> "results_20250718_143025.hdf5"
    """
    if timestamp is None:
        # Different timestamp formats:
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")        # 20250718_143025
        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")    # 2025-07-18_14-30-25
        # timestamp = datetime.now().strftime("%Y%m%d")               # 20250718 (date only)
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M")          # 20250718_1430 (no seconds)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    name, ext = os.path.splitext(filename)
    return f"{name}_{timestamp}{ext}"


def get_festim_python():
    """
    Get the correct Python executable for the FESTIM environment.

    Searches for a conda environment named ``festim2-env`` or
    ``festim-env`` and returns the path to its Python interpreter.
    Falls back to the current interpreter (``sys.executable``) if no
    dedicated FESTIM environment is found.

    Returns
    -------
    str
        Absolute path to a Python 3 executable.
    """
    # Method 1: Look for common FESTIM conda environment names
    for env_name in ("festim2-env", "festim-env"):
        try:
            result = subprocess.run(
                ["conda", "info", "--envs"],
                capture_output=True,
                text=True,
                check=True,
            )
            for line in result.stdout.split("\n"):
                if env_name in line:
                    env_path = line.split()[-1]
                    python_path = os.path.join(env_path, "bin", "python3")
                    if os.path.exists(python_path):
                        return python_path
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

    # Method 2: Fallback to current Python
    print("Warning: FESTIM environment not found, using current Python")
    return sys.executable


def validate_execution_setup():
    """Validate that the execution environment is properly configured."""
    runnable_script = "festim_model_run.py"
    script_path = os.path.join(os.getcwd(), runnable_script)

    # Check script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Check script is executable
    if not os.access(script_path, os.X_OK):
        print(f"Making script executable: {script_path}")
        os.chmod(script_path, 0o755)

    # Check Python executable
    python_exe = get_festim_python()
    if not os.path.exists(python_exe):
        raise FileNotFoundError(f"Python executable not found: {python_exe}")

    print(f"✓ Script validation passed: {script_path}")
    print(f"✓ Python executable: {python_exe}")
    return python_exe, script_path


def save_sa_results(results, qois, output_dir, timestamp=None, param_names=None):
    """
    Save sensitivity analysis results (Sobol indices, statistics) to YAML and CSV.

    Extracts first-order and total Sobol sensitivity indices, as well as
    mean and standard deviation statistics, from an EasyVVUQ analysis
    *results* object and writes them to timestamped files inside
    *output_dir*.

    Parameters
    ----------
    results : easyvvuq.analysis.results.AnalysisResults
        EasyVVUQ analysis results object (from ``PCEAnalysis`` or similar).
    qois : list of str
        Quantity-of-interest names to extract (e.g. ``["t=steady"]``).
    output_dir : str
        Directory where the output files will be written (created if needed).
    timestamp : str, optional
        Identifier appended to filenames.  Defaults to ``YYYYMMDD_HHMMSS``.
    param_names : list of str, optional
        Uncertain-parameter names expected in the Sobol dictionaries.
        If *None*, names are inferred from the first non-empty
        ``sobols_first`` call.

    Returns
    -------
    dict
        ``{"yaml": <path>, "csv": <path>}`` mapping to the written files.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)

    # ---- Collect data ------------------------------------------------
    sa_data = {
        "timestamp": timestamp,
        "description": "Sensitivity analysis results from FESTIM-NIUQ",
        "qois": {},
    }

    for qoi in qois:
        qoi_entry = {}

        # --- statistics -----------------------------------------------
        for stat_name in ("mean", "std"):
            try:
                values = results.describe(qoi, stat_name)
                if values is not None:
                    arr = np.asarray(values)
                    qoi_entry[stat_name] = {
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "avg": float(np.mean(arr)),
                    }
            except Exception:
                pass

        # --- first-order Sobol indices --------------------------------
        try:
            sobols = results.sobols_first(qoi)
            if sobols:
                non_zero = {
                    k: np.asarray(v) for k, v in sobols.items() if v is not None and not np.all(np.array(v) == 0)
                }
                if non_zero:
                    if param_names is None:
                        param_names = list(non_zero.keys())
                    qoi_entry["sobols_first"] = {
                        k: {"min": float(np.min(v)), "max": float(np.max(v)), "avg": float(np.mean(v))}
                        for k, v in non_zero.items()
                    }
        except Exception:
            pass

        # --- total Sobol indices --------------------------------------
        try:
            sobols_t = results.sobols_total(qoi)
            if sobols_t:
                non_zero_t = {
                    k: np.asarray(v) for k, v in sobols_t.items() if v is not None and not np.all(np.array(v) == 0)
                }
                if non_zero_t:
                    qoi_entry["sobols_total"] = {
                        k: {"min": float(np.min(v)), "max": float(np.max(v)), "avg": float(np.mean(v))}
                        for k, v in non_zero_t.items()
                    }
        except Exception:
            pass

        sa_data["qois"][qoi] = qoi_entry

    # ---- Write YAML --------------------------------------------------
    yaml_path = os.path.join(output_dir, f"sa_results_{timestamp}.yaml")
    with open(yaml_path, "w") as fh:
        yaml.dump(sa_data, fh, default_flow_style=False, sort_keys=False)
    print(f"✓ SA results (YAML) saved to: {yaml_path}")

    # ---- Write CSV (one row per QoI × param) -------------------------
    csv_path = os.path.join(output_dir, f"sa_results_{timestamp}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["qoi", "parameter", "sobol_first_avg", "sobol_total_avg", "mean_avg", "std_avg"])
        for qoi, entry in sa_data["qois"].items():
            mean_avg = entry.get("mean", {}).get("avg", "")
            std_avg = entry.get("std", {}).get("avg", "")
            sobol_first = entry.get("sobols_first", {})
            sobol_total = entry.get("sobols_total", {})
            all_params = set(list(sobol_first.keys()) + list(sobol_total.keys()))
            if not all_params:
                writer.writerow([qoi, "", "", "", mean_avg, std_avg])
            else:
                for p in sorted(all_params):
                    sf = sobol_first.get(p, {}).get("avg", "")
                    st = sobol_total.get(p, {}).get("avg", "")
                    writer.writerow([qoi, p, sf, st, mean_avg, std_avg])
    print(f"✓ SA results (CSV) saved to: {csv_path}")

    return {"yaml": yaml_path, "csv": csv_path}


def get_qoi_names(results):
    """
    Return the list of quantity-of-interest (QoI) names from an EasyVVUQ results object.

    Parameters
    ----------
    results : easyvvuq.analysis.results.AnalysisResults
        EasyVVUQ analysis results object.

    Returns
    -------
    list of str
        QoI column names present in the results.
    """
    try:
        # EasyVVUQ AnalysisResults stores QoI names in ._qois or as DataFrame columns
        if hasattr(results, "qois"):
            return list(results.qois)
        if hasattr(results, "_qois"):
            return list(results._qois)
        # Fallback: inspect the samples DataFrame
        if hasattr(results, "samples"):
            return [c for c in results.samples.columns if c not in ("run_id",)]
    except Exception:
        pass
    return []


def get_sobol_first(results, qoi):
    """
    Return the first-order Sobol indices for a given QoI.

    Parameters
    ----------
    results : easyvvuq.analysis.results.AnalysisResults
        EasyVVUQ analysis results object.
    qoi : str
        Name of the quantity of interest.

    Returns
    -------
    dict of {str: numpy.ndarray}
        Mapping of parameter name to an array of first-order Sobol indices
        (one value per spatial collocation point).  Returns an empty dict
        on failure.
    """
    try:
        sobols = results.sobols_first(qoi)
        if sobols:
            return {k: np.asarray(v) for k, v in sobols.items() if v is not None}
    except Exception:
        pass
    return {}


def get_sobol_total(results, qoi):
    """
    Return the total-order Sobol indices for a given QoI.

    Parameters
    ----------
    results : easyvvuq.analysis.results.AnalysisResults
        EasyVVUQ analysis results object.
    qoi : str
        Name of the quantity of interest.

    Returns
    -------
    dict of {str: numpy.ndarray}
        Mapping of parameter name to an array of total Sobol indices.
        Returns an empty dict on failure.
    """
    try:
        sobols = results.sobols_total(qoi)
        if sobols:
            return {k: np.asarray(v) for k, v in sobols.items() if v is not None}
    except Exception:
        pass
    return {}


def get_stat(results, qoi, stat_name):
    """
    Return a descriptive statistic for a given QoI.

    Parameters
    ----------
    results : easyvvuq.analysis.results.AnalysisResults
        EasyVVUQ analysis results object.
    qoi : str
        Name of the quantity of interest.
    stat_name : str
        Statistic to retrieve, e.g. ``"mean"``, ``"std"``, ``"1%"``,
        ``"median"``, ``"99%"``.

    Returns
    -------
    numpy.ndarray or None
        Array of the requested statistic (one value per spatial node),
        or ``None`` if not available.
    """
    try:
        values = results.describe(qoi, stat_name)
        if values is not None:
            return np.asarray(values)
    except Exception:
        pass
    return None


def integrate_statistics(results, qois=None, x_values=None):
    """
    Integrate Sobol sensitivity indices over the spatial domain.

    For spatially-resolved QoIs the first-order Sobol index is a
    function of position (e.g. *r*).  This helper integrates each
    parameter's Sobol index over *x_values* using the trapezoidal rule,
    yielding a single scalar importance measure per parameter.

    Parameters
    ----------
    results : easyvvuq.analysis.results.AnalysisResults
        EasyVVUQ analysis results object.
    qois : list of str, optional
        QoI names to process.  If *None*, all QoIs from the results are
        used (skipping the coordinate column ``"x"``).
    x_values : numpy.ndarray, optional
        Spatial coordinate array matching the length of the Sobol arrays.
        When *None*, uniform spacing is assumed (i.e. ``dx = 1``).

    Returns
    -------
    dict
        ``{qoi: {param_name: integrated_sobol_value, ...}, ...}``
    """
    if qois is None:
        qois = get_qoi_names(results)
        # Skip common non-QoI columns
        qois = [q for q in qois if q not in ("x", "run_id")]

    all_integrated = {}

    for qoi in qois:
        print(f"\n>>> Integrating Sobol indices for QoI: {qoi}")

        sobols = get_sobol_first(results, qoi)
        if not sobols:
            print(f"  No first-order Sobol indices for '{qoi}', skipping.")
            continue

        stat_integrated = {}
        for param_name, sobol_arr in sobols.items():
            if np.all(sobol_arr == 0):
                continue
            # np.trapz was renamed to np.trapezoid in NumPy 2.0
            _trapz = getattr(np, "trapezoid", None) or np.trapz
            if x_values is not None and len(x_values) == len(sobol_arr):
                integrated_value = float(_trapz(sobol_arr, x=x_values))
            else:
                integrated_value = float(_trapz(sobol_arr))
            stat_integrated[param_name] = integrated_value
            print(f"  Integrated S1({param_name}) = {integrated_value:.6e}")

        all_integrated[qoi] = stat_integrated

    return all_integrated


def compute_absolute_tolerance(default_atol, orig_params, new_params):
    """
    Compute (a heuristic for) absolute tolerance by multiplying default absolute tolerance with a factors according to parameter changes.

    Args:
        default_atol (float): Default absolute tolerance
        orig_params (dict): Original parameter values
        new_params (dict): New parameter values

    Returns:
        float: Computed absolute tolerance
    """

    if not orig_params or not new_params:
        print("Original or new parameters are empty, returning default absolute tolerance.")
        return default_atol

    # Dictionary of problem QoI sensitivities to parameters changes
    """
    The sensitivities are derived from physical considerations and dimensional analysis of the tritium transport problem.
    The sensitivity A_i for parameter x_i (out of N input parameters) indicates how much the absolute tolerance should change in response to a change in that parameter.
    The change is by the following (log) rule: atol_new = atol_orig * 10**( SUM_{i=1}^{N} (A_i * log10(x_i_new/x_i_orig)) ))
    Temparature and other parameters (...) enter the model exponentailly.
    The cahnge is by the following (exp) rule: atol_new = atol_orig * 10**( SUM_{i=1}^{N} (A_i * x_i_new/x_i_orig) ))
    """
    log_sensitivities = {
        "length": 1.0,  # 2.0,
        "G": 1.0,
        "right_bc_concentration_value": 0.5,
    }

    exp_sensitivities = {
        "T": -6.0,  # special rule has to be applied, i.e. 10**(6.0 * T_new/T_orig)
        "T_in": 0.0,  # -1.5,
    }

    # Compute the exponential factor based on parameter changes
    multiplier = 1.0
    # exp_factor = 1.0

    for key in new_params:
        if key in orig_params:
            if key in log_sensitivities:
                log_ratio = np.log10(abs(new_params[key] / orig_params[key])) if orig_params[key] != 0 else 0
                print(f" >>>> Computing tolerance for {key}: log_ratio={log_ratio}")

                log_sensitivity = log_sensitivities.get(key, 1.0)  # Default sensitivity is 1.0 if not specified
                print(f" >>>> Computing tolerance for {key}: log_sensitivity={log_sensitivity}")

                exp_factor = log_sensitivity * log_ratio
                print(f" >>>> Computing tolerance for {key}: exp_factor={exp_factor}")

            elif key in exp_sensitivities:
                frac_ratio = abs(new_params[key] / orig_params[key])
                print(f" >>>> Computing tolerance for {key}: frac_ratio={frac_ratio}")

                exp_sensitivity = exp_sensitivities.get(key, 1.0)  # Default sensitivity is 1.0 if not specified
                print(f" >>>> Computing tolerance for {key}: exp_sensitivity={exp_sensitivity}")

                exp_factor = exp_sensitivity * frac_ratio
                print(f" >>>> Computing tolerance for {key}: exp_factor={exp_factor:.3E}")

            else:
                raise NotImplemented(f"The tolerance sensitivity rule for {key} is not implemented!")

            # This is done not as a sum first because different rules can be applied to different parameters
            multiplier *= 10**exp_factor
            print(f" >>>> Computing tolerance for {key}: new multiplier={multiplier}")
        else:
            print(f"Warning: Parameter {key} not found in original parameters, skipping.")

    new_atol = float(default_atol * multiplier)
    print(f" >>>> Computing tolerance: multiplier={multiplier:.3E}")
    print(f" >>>  Computing tolerance: new_atol={new_atol:.3E}")

    return new_atol
