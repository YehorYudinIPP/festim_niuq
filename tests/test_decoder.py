"""
Tests for the ScalarCSVDecoder class from uq/util/Decoder.py.
"""

import os
import pytest

from uq.util.Decoder import ScalarCSVDecoder


class TestScalarCSVDecoder:
    """Tests for parsing single-row CSV files with scalar QoIs."""

    @pytest.fixture
    def output_csv(self, tmp_path):
        """Write a minimal output.csv file and return the directory."""
        csv_content = "total_tritium_release,total_tritium_trapping\n1.23e18,4.56e17\n"
        (tmp_path / "output.csv").write_text(csv_content)
        return tmp_path

    def test_parse_returns_correct_values(self, output_csv):
        decoder = ScalarCSVDecoder(
            target_filename="output.csv",
            output_columns=["total_tritium_release", "total_tritium_trapping"],
        )
        result = decoder.parse_sim_output(run_dir=str(output_csv))
        assert pytest.approx(result["total_tritium_release"], rel=1e-6) == 1.23e18
        assert pytest.approx(result["total_tritium_trapping"], rel=1e-6) == 4.56e17

    def test_missing_file_returns_none(self, tmp_path):
        decoder = ScalarCSVDecoder(
            target_filename="output.csv",
            output_columns=["total_tritium_release"],
        )
        result = decoder.parse_sim_output(run_dir=str(tmp_path))
        assert result["total_tritium_release"] is None

    def test_missing_column_returns_none(self, output_csv):
        decoder = ScalarCSVDecoder(
            target_filename="output.csv",
            output_columns=["nonexistent_column"],
        )
        result = decoder.parse_sim_output(run_dir=str(output_csv))
        assert result["nonexistent_column"] is None

    def test_empty_csv_returns_none(self, tmp_path):
        (tmp_path / "output.csv").write_text("")
        decoder = ScalarCSVDecoder(
            target_filename="output.csv",
            output_columns=["total_tritium_release"],
        )
        result = decoder.parse_sim_output(run_dir=str(tmp_path))
        assert result["total_tritium_release"] is None

    def test_header_only_csv_returns_none(self, tmp_path):
        (tmp_path / "output.csv").write_text("total_tritium_release\n")
        decoder = ScalarCSVDecoder(
            target_filename="output.csv",
            output_columns=["total_tritium_release"],
        )
        result = decoder.parse_sim_output(run_dir=str(tmp_path))
        assert result["total_tritium_release"] is None

    def test_get_restart_dict(self):
        decoder = ScalarCSVDecoder(
            target_filename="out.csv",
            output_columns=["col_a", "col_b"],
        )
        rd = decoder.get_restart_dict()
        assert rd["target_filename"] == "out.csv"
        assert rd["output_columns"] == ["col_a", "col_b"]

    def test_deserialize_roundtrip(self):
        decoder = ScalarCSVDecoder(
            target_filename="out.csv",
            output_columns=["col_a"],
        )
        rd = decoder.get_restart_dict()
        decoder2 = ScalarCSVDecoder.deserialize(rd)
        assert decoder2.target_filename == decoder.target_filename
        assert decoder2.output_columns == decoder.output_columns
