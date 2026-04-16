"""
Custom decoder for scalar QoIs written by the FESTIM model.

Reads ``summary.csv`` (or ``output.csv``) produced by the forward model and
returns a dictionary of scalar quantities of interest suitable for EasyVVUQ
analysis.  This avoids the need for the profile-based ``SimpleCSV`` decoder
when only scalar QoIs are required (e.g. total tritium release and trapping).
"""
import os
import csv


class ScalarCSVDecoder:
  """
  Decoder that reads a single-row CSV with scalar QoIs.

  The CSV is expected to have a header row with column names and a single
  data row.  Example:

          total_tritium_release,total_tritium_trapping
          1.234e+15,5.678e+14
          
  Custom CSV Decoder for EasyVVUQ.

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
        if run_info is None:
            run_dir = "."
        else:
            run_dir = run_info.get("run_dir", ".")

        filepath = os.path.join(run_dir, self.target_filename)

        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"ScalarCSVDecoder: output file not found: {filepath}"
            )

        with open(filepath, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        if len(rows) == 0:
            raise ValueError(
                f"ScalarCSVDecoder: no data rows in {filepath}"
            )

        row = rows[0]  # single-row CSV

        result = {}
        for col in self.output_columns:
            raw = row.get(col, None)
            if raw is None:
                raise KeyError(
                    f"ScalarCSVDecoder: column '{col}' not found in "
                    f"{filepath}. Available: {list(row.keys())}"
                )
            result[col] = float(raw)

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
