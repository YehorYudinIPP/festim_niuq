"""
Custom CSV Decoder for EasyVVUQ.

Provides a :class:`ScalarCSVDecoder` that reads single-row CSV files
produced by the FESTIM model wrapper (``output.csv``).  Each column in
the CSV corresponds to a scalar quantity of interest (QoI), for example
``total_tritium_release`` and ``total_tritium_trapping``.

This decoder is intended for steady-state UQ campaigns where the QoI is
a single scalar value per simulation, as opposed to the built-in
``SimpleCSV`` decoder which reads spatially-resolved profile data.

Example
-------
>>> decoder = ScalarCSVDecoder(
...     target_filename="output.csv",
...     output_columns=["total_tritium_release", "total_tritium_trapping"],
... )
"""

import csv
import os


class ScalarCSVDecoder:
    """
    Decoder for single-row CSV files with scalar QoIs.

    Parameters
    ----------
    target_filename : str
        Name of the CSV file to read (relative to the run directory).
    output_columns : list of str
        Column names to extract from the CSV.

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
            Mapping of column name -> float value for each requested QoI.
            Missing columns are returned as ``None``.
        """
        filepath = os.path.join(run_dir or ".", self.target_filename)

        if not os.path.isfile(filepath):
            return {col: None for col in self.output_columns}

        try:
            with open(filepath, "r", newline="") as fh:
                reader = csv.DictReader(fh)
                row = next(reader)
            return {
                col: float(row[col]) if col in row else None
                for col in self.output_columns
            }
        except (StopIteration, KeyError, ValueError):
            return {col: None for col in self.output_columns}

    # ------------------------------------------------------------------
    # EasyVVUQ restart / serialisation interface
    # ------------------------------------------------------------------

    def get_restart_dict(self):
        """Return a dictionary sufficient to reconstruct this decoder."""
        return {
            "target_filename": self.target_filename,
            "output_columns": self.output_columns,
        }

    @classmethod
    def deserialize(cls, data):
        """Reconstruct a :class:`ScalarCSVDecoder` from a restart dict."""
        return cls(
            target_filename=data["target_filename"],
            output_columns=data.get("output_columns", []),
        )
