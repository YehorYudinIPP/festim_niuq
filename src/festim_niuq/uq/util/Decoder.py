"""
Custom decoder for scalar QoIs written by the FESTIM model.

Reads ``summary.csv`` (or ``output.csv``) produced by the forward model and
returns a dictionary of scalar quantities of interest suitable for EasyVVUQ
analysis.  This avoids the need for the profile-based ``SimpleCSV`` decoder
when only scalar QoIs are required (e.g. total tritium release and trapping).
"""
import os
import csv
import numpy as np


class ScalarCSVDecoder:
    """Decoder that reads a single-row CSV with scalar QoIs.

    The CSV is expected to have a header row with column names and a single
    data row.  Example::

        total_tritium_release,total_tritium_trapping
        1.234e+15,5.678e+14

    Provides a :class:`ScalarCSVDecoder` that reads single-row CSV files
    produced by the FESTIM model wrapper (``output.csv``).  Each column in
    the CSV corresponds to a scalar quantity of interest (QoI), for example
    ``total_tritium_release`` and ``total_tritium_trapping``.

    This decoder is intended for steady-state UQ campaigns where the QoI is
    a single scalar value per simulation, as opposed to the built-in
    ``SimpleCSV`` decoder which reads spatially-resolved profile data.

    Parameters
    ----------
    target_filename : str
        Name of the CSV file to read (relative to the run directory).
    output_columns : list of str
        Column names to extract from the CSV.

    Example
    -------
    >>> decoder = ScalarCSVDecoder(
    ...     target_filename="output.csv",
    ...     output_columns=["total_tritium_release", "total_tritium_trapping"],
    ... )

    Notes
    -----
    The CSV file is expected to have a header row followed by exactly
    one data row.  If the file is missing or malformed the decoder
    returns ``None`` values for every requested column so that the
    EasyVVUQ collation step can flag the run as failed rather than
    crashing the entire campaign.
    """

    def __init__(self, target_filename="output.csv", output_columns=None):
        self.target_filename = target_filename
        self.output_columns = output_columns or []

    # ------------------------------------------------------------------
    # EasyVVUQ decoder interface
    # ------------------------------------------------------------------

    def parse_sim_output(self, run_info=None, run_dir=None):
        """
        Parse simulation output and return a dictionary of QoI values.

        Parameters
        ----------
        run_info : dict or None
            EasyVVUQ run metadata (unused).
        run_dir : str or None
            Path to the run directory that contains ``target_filename``.

        Returns
        -------
        dict
            Mapping from column name to scalar float value.
        """
        if run_info is not None:
            run_dir = run_info.get("run_dir", run_dir or ".")
        elif run_dir is None:
            run_dir = "."

        filepath = os.path.join(run_dir, self.target_filename)

        none_result = {col: None for col in self.output_columns}

        if not os.path.isfile(filepath):
            return none_result

        with open(filepath, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if len(rows) == 0:
            return none_result

        row = rows[0]  # single-row CSV

        result = {}
        for col in self.output_columns:
            raw = row.get(col, None)
            if raw is None:
                result[col] = None
            else:
                try:
                    result[col] = float(raw)
                except (ValueError, TypeError):
                    result[col] = None

        return result

    # ------------------------------------------------------------------
    # Serialisation helpers (needed by EasyVVUQ restart)
    # ------------------------------------------------------------------
    def get_restart_dict(self):
        return {
            "target_filename": self.target_filename,
            "output_columns": self.output_columns,
        }

    @staticmethod
    def element_version():
        return "0.1"

    @staticmethod
    def element_name():
        return "ScalarCSVDecoder"

    @classmethod
    def deserialize(cls, data):
        """Reconstruct a :class:`ScalarCSVDecoder` from a restart dict."""
        return cls(
            target_filename=data["target_filename"],
            output_columns=data.get("output_columns", []),
        )


class MultiOutputDecoder:
    """Decoder that combines profile and flux outputs into one QoI dictionary.

    This decoder reads two FESTIM outputs from a run directory:
    - concentration profile file with columns ``x, t=...``
    - outer-surface flux time series file with columns
      ``time, total_hydrogen_flux_rmax``

    It returns a single dictionary that EasyVVUQ can collate/analyse, with
    profile QoIs (arrays over radius) and flux QoIs (scalars at requested
    milestone times).
    """

    def __init__(
        self,
        profile_filename="results/test/results_tritium_concentration.txt",
        flux_filename="results/test/total_hydrogen_flux_rmax.txt",
        concentration_qois=None,
        flux_qois=None,
        missing_flux_value=np.nan,
    ):
        self.profile_filename = profile_filename
        self.flux_filename = flux_filename
        self.concentration_qois = concentration_qois or []
        self.flux_qois = flux_qois or []
        self.missing_flux_value = float(missing_flux_value)
        # EasyVVUQ compatibility: some code paths inspect decoder metadata.
        self.target_filename = profile_filename
        self.output_columns = list(self.concentration_qois) + list(self.flux_qois)

    @staticmethod
    def _read_csv_with_header(filepath):
        if not os.path.isfile(filepath):
            return [], None

        with open(filepath, "r", newline="") as fh:
            header_line = fh.readline().strip()

        header_line = header_line.lstrip("#").strip()
        headers = [h.strip() for h in header_line.split(",")] if header_line else []

        data = np.genfromtxt(filepath, delimiter=",", skip_header=1)
        if data.size == 0:
            return headers, None
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return headers, data

    @staticmethod
    def _parse_flux_time_label(qoi_name):
        if not isinstance(qoi_name, str):
            return None
        if not (qoi_name.startswith("flux_t=") and qoi_name.endswith("s")):
            return None
        try:
            return float(qoi_name.split("=", 1)[1].rstrip("s"))
        except Exception:
            return None

    def parse_sim_output(self, run_info=None, run_dir=None):
        if run_info is not None:
            run_dir = run_info.get("run_dir", run_dir or ".")
        elif run_dir is None:
            run_dir = "."

        result = {}

        # --- concentration profile QoIs ---
        profile_path = os.path.join(run_dir, self.profile_filename)
        profile_headers, profile_data = self._read_csv_with_header(profile_path)
        profile_index = {name: i for i, name in enumerate(profile_headers)}

        for qoi in self.concentration_qois:
            if profile_data is None:
                result[qoi] = None
                continue
            idx = profile_index.get(qoi)
            if idx is None or idx >= profile_data.shape[1]:
                result[qoi] = None
            else:
                # EasyVVUQ external-run import serializes outputs to JSON, so keep
                # concentration profiles as plain Python lists rather than ndarrays.
                result[qoi] = np.asarray(profile_data[:, idx], dtype=float).tolist()

        # --- flux QoIs (scalar per requested milestone time) ---
        flux_path = os.path.join(run_dir, self.flux_filename)
        flux_headers, flux_data = self._read_csv_with_header(flux_path)
        flux_index = {name: i for i, name in enumerate(flux_headers)}

        if flux_data is None:
            for qoi in self.flux_qois:
                result[qoi] = self.missing_flux_value
            return result

        # Expected format: time,total_hydrogen_flux_rmax
        t_idx = flux_index.get("time", 0)
        f_idx = flux_index.get("total_hydrogen_flux_rmax", 1 if flux_data.shape[1] > 1 else 0)

        if t_idx >= flux_data.shape[1] or f_idx >= flux_data.shape[1]:
            for qoi in self.flux_qois:
                result[qoi] = self.missing_flux_value
            return result

        t_vals = np.asarray(flux_data[:, t_idx], dtype=float).reshape(-1)
        f_vals = np.asarray(flux_data[:, f_idx], dtype=float).reshape(-1)

        for qoi in self.flux_qois:
            t_target = self._parse_flux_time_label(qoi)
            if t_target is None or t_vals.size == 0:
                result[qoi] = self.missing_flux_value
                continue
            idx = int(np.argmin(np.abs(t_vals - t_target)))
            result[qoi] = float(f_vals[idx]) if idx < f_vals.size else self.missing_flux_value

        return result

    def get_restart_dict(self):
        return {
            "profile_filename": self.profile_filename,
            "flux_filename": self.flux_filename,
            "concentration_qois": self.concentration_qois,
            "flux_qois": self.flux_qois,
            "missing_flux_value": self.missing_flux_value,
        }

    @staticmethod
    def element_version():
        return "0.1"

    @staticmethod
    def element_name():
        return "MultiOutputDecoder"

    @classmethod
    def deserialize(cls, data):
        return cls(
            profile_filename=data.get("profile_filename", "results/test/results_tritium_concentration.txt"),
            flux_filename=data.get("flux_filename", "results/test/total_hydrogen_flux_rmax.txt"),
            concentration_qois=data.get("concentration_qois", []),
            flux_qois=data.get("flux_qois", []),
            missing_flux_value=data.get("missing_flux_value", np.nan),
        )
