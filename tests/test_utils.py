"""
Tests for utility functions in uq/util/utils.py.
"""

import csv
import os
import tempfile
import pytest
import yaml
import numpy as np
from unittest.mock import MagicMock

from uq.util.utils import (
    load_config,
    add_timestamp_to_filename,
    compute_absolute_tolerance,
    save_sa_results,
    get_qoi_names,
    get_sobol_first,
    get_sobol_total,
    get_stat,
    integrate_statistics,
)


class TestLoadConfig:
    """Tests for load_config."""

    def test_load_valid_yaml(self, tmp_path):
        config = {"model": {"name": "test"}, "value": 42}
        config_file = tmp_path / "test.yaml"
        with open(str(config_file), "w") as f:
            yaml.dump(config, f)

        result = load_config(str(config_file))
        assert result is not None
        assert result["model"]["name"] == "test"
        assert result["value"] == 42

    def test_load_missing_file_returns_none(self, tmp_path):
        result = load_config(str(tmp_path / "nonexistent.yaml"))
        assert result is None

    def test_load_invalid_yaml_returns_none(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("key: [unbalanced\n")
        result = load_config(str(bad_yaml))
        assert result is None


class TestAddTimestampToFilename:
    """Tests for add_timestamp_to_filename."""

    def test_adds_custom_timestamp(self):
        result = add_timestamp_to_filename("results.csv", timestamp="20250101_120000")
        assert result == "results_20250101_120000.csv"

    def test_preserves_extension(self):
        result = add_timestamp_to_filename("data.hdf5", timestamp="ts")
        assert result == "data_ts.hdf5"

    def test_handles_no_extension(self):
        result = add_timestamp_to_filename("output", timestamp="ts")
        assert result == "output_ts"

    def test_auto_timestamp_has_expected_format(self):
        result = add_timestamp_to_filename("test.txt")
        # Should match pattern: test_YYYYMMDD_HHMMSS.txt
        assert result.startswith("test_")
        assert result.endswith(".txt")
        # Timestamp part should be 15 chars: YYYYMMDD_HHMMSS
        name_part = result[len("test_") : -len(".txt")]
        assert len(name_part) == 15

    def test_handles_path_with_directory(self):
        result = add_timestamp_to_filename("/path/to/file.csv", timestamp="ts")
        assert result == "/path/to/file_ts.csv"


class TestComputeAbsoluteTolerance:
    """Tests for compute_absolute_tolerance."""

    def test_default_when_empty_params(self):
        result = compute_absolute_tolerance(1e-10, {}, {})
        assert result == 1e-10

    def test_default_when_none_params(self):
        result = compute_absolute_tolerance(1e-10, None, {})
        assert result == 1e-10

    def test_no_change_when_same_params(self):
        orig = {"length": 1.0, "G": 1.0}
        new = {"length": 1.0, "G": 1.0}
        result = compute_absolute_tolerance(1e-10, orig, new)
        assert np.isclose(result, 1e-10, rtol=1e-6)

    def test_tolerance_increases_with_larger_length(self):
        orig = {"length": 1.0}
        new = {"length": 10.0}
        result = compute_absolute_tolerance(1e-10, orig, new)
        # length sensitivity is 1.0, so 10^(1.0 * log10(10)) = 10
        assert result > 1e-10

    def test_tolerance_changes_with_temperature(self):
        orig = {"T": 300.0}
        new = {"T": 600.0}
        result = compute_absolute_tolerance(1e-10, orig, new)
        # T has exp sensitivity of -6.0, so multiplier = 10^(-6.0 * 600/300) = 10^(-12)
        assert result != 1e-10

    def test_missing_param_in_orig_is_skipped(self):
        orig = {"length": 1.0}
        new = {"length": 1.0, "missing_param": 5.0}
        # missing_param is not in orig, should be skipped, but "missing_param" is not
        # in log_sensitivities or exp_sensitivities either, so it would raise NotImplemented
        # Actually it checks `if key in orig_params` first, so it's skipped
        result = compute_absolute_tolerance(1e-10, orig, new)
        assert np.isclose(result, 1e-10, rtol=1e-6)


# ---------------------------------------------------------------------------
# Helper: mock EasyVVUQ AnalysisResults
# ---------------------------------------------------------------------------


def _make_mock_results(qoi_names=None, sobols_first=None, sobols_total=None, stats=None):
    """Create a mock object mimicking an EasyVVUQ AnalysisResults."""
    mock = MagicMock()
    mock.qois = qoi_names or ["x", "t=steady"]

    def _sobols_first(qoi):
        if sobols_first and qoi in sobols_first:
            return sobols_first[qoi]
        return {}

    def _sobols_total(qoi):
        if sobols_total and qoi in sobols_total:
            return sobols_total[qoi]
        return {}

    def _describe(qoi, stat):
        if stats and qoi in stats and stat in stats[qoi]:
            return stats[qoi][stat]
        return None

    mock.sobols_first = _sobols_first
    mock.sobols_total = _sobols_total
    mock.describe = _describe
    return mock


# ---------------------------------------------------------------------------
# Tests for save_sa_results
# ---------------------------------------------------------------------------


class TestSaveSaResults:
    """Tests for save_sa_results."""

    def test_creates_yaml_and_csv(self, tmp_path):
        results = _make_mock_results(
            sobols_first={"t=steady": {"D_0": np.array([0.5, 0.3]), "kappa": np.array([0.2, 0.4])}},
            stats={"t=steady": {"mean": np.array([1.0, 2.0]), "std": np.array([0.1, 0.2])}},
        )
        paths = save_sa_results(results, ["t=steady"], str(tmp_path), timestamp="test")
        assert os.path.isfile(paths["yaml"])
        assert os.path.isfile(paths["csv"])

    def test_yaml_contains_qoi_entry(self, tmp_path):
        results = _make_mock_results(
            sobols_first={"t=steady": {"D_0": np.array([0.6])}},
            stats={"t=steady": {"mean": np.array([1.0]), "std": np.array([0.1])}},
        )
        save_sa_results(results, ["t=steady"], str(tmp_path), timestamp="ytest")
        with open(os.path.join(str(tmp_path), "sa_results_ytest.yaml")) as fh:
            data = yaml.safe_load(fh)
        assert "t=steady" in data["qois"]
        assert "sobols_first" in data["qois"]["t=steady"]

    def test_csv_has_header_and_rows(self, tmp_path):
        results = _make_mock_results(
            sobols_first={"t=steady": {"D_0": np.array([0.7]), "G": np.array([0.1])}},
            stats={"t=steady": {"mean": np.array([1.0])}},
        )
        save_sa_results(results, ["t=steady"], str(tmp_path), timestamp="csvtest")
        csv_path = os.path.join(str(tmp_path), "sa_results_csvtest.csv")
        with open(csv_path) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 2  # one per param (D_0, G)
        assert rows[0]["qoi"] == "t=steady"

    def test_no_sobol_writes_single_row(self, tmp_path):
        results = _make_mock_results(
            stats={"t=steady": {"mean": np.array([1.0])}},
        )
        save_sa_results(results, ["t=steady"], str(tmp_path), timestamp="nosob")
        csv_path = os.path.join(str(tmp_path), "sa_results_nosob.csv")
        with open(csv_path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Tests for get_* accessor helpers
# ---------------------------------------------------------------------------


class TestGetQoiNames:
    """Tests for get_qoi_names."""

    def test_returns_qoi_list(self):
        results = _make_mock_results(qoi_names=["x", "t=steady", "tritium_concentration"])
        assert get_qoi_names(results) == ["x", "t=steady", "tritium_concentration"]

    def test_returns_empty_on_failure(self):
        mock = MagicMock(spec=[])  # no attributes at all
        assert get_qoi_names(mock) == []


class TestGetSobolFirst:
    """Tests for get_sobol_first."""

    def test_returns_sobol_dict(self):
        arr = np.array([0.3, 0.4, 0.5])
        results = _make_mock_results(sobols_first={"qoi1": {"D_0": arr}})
        out = get_sobol_first(results, "qoi1")
        assert "D_0" in out
        np.testing.assert_array_equal(out["D_0"], arr)

    def test_returns_empty_when_missing(self):
        results = _make_mock_results()
        assert get_sobol_first(results, "nonexistent") == {}


class TestGetSobolTotal:
    """Tests for get_sobol_total."""

    def test_returns_sobol_dict(self):
        arr = np.array([0.8, 0.9])
        results = _make_mock_results(sobols_total={"qoi1": {"kappa": arr}})
        out = get_sobol_total(results, "qoi1")
        assert "kappa" in out
        np.testing.assert_array_equal(out["kappa"], arr)

    def test_returns_empty_when_missing(self):
        results = _make_mock_results()
        assert get_sobol_total(results, "none") == {}


class TestGetStat:
    """Tests for get_stat."""

    def test_returns_stat_array(self):
        arr = np.array([10.0, 20.0])
        results = _make_mock_results(stats={"qoi1": {"mean": arr}})
        out = get_stat(results, "qoi1", "mean")
        np.testing.assert_array_equal(out, arr)

    def test_returns_none_when_missing(self):
        results = _make_mock_results()
        assert get_stat(results, "qoi1", "std") is None


# ---------------------------------------------------------------------------
# Tests for integrate_statistics
# ---------------------------------------------------------------------------


class TestIntegrateStatistics:
    """Tests for integrate_statistics."""

    def test_integrates_over_uniform_spacing(self):
        # Constant Sobol = 0.5 over 5 points with dx=1 → integral ≈ 0.5*4 = 2.0
        results = _make_mock_results(
            qoi_names=["x", "qoi1"],
            sobols_first={"qoi1": {"D_0": np.array([0.5, 0.5, 0.5, 0.5, 0.5])}},
        )
        out = integrate_statistics(results, qois=["qoi1"])
        assert "qoi1" in out
        assert np.isclose(out["qoi1"]["D_0"], 2.0)

    def test_integrates_with_x_values(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        sobol = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        results = _make_mock_results(
            qoi_names=["x", "qoi1"],
            sobols_first={"qoi1": {"D_0": sobol}},
        )
        out = integrate_statistics(results, qois=["qoi1"], x_values=x)
        assert np.isclose(out["qoi1"]["D_0"], 4.0)

    def test_skips_all_zero_sobol(self):
        results = _make_mock_results(
            qoi_names=["x", "qoi1"],
            sobols_first={"qoi1": {"D_0": np.zeros(5)}},
        )
        out = integrate_statistics(results, qois=["qoi1"])
        assert out["qoi1"] == {}

    def test_returns_empty_when_no_sobols(self):
        results = _make_mock_results(qoi_names=["x", "qoi1"])
        out = integrate_statistics(results, qois=["qoi1"])
        assert out == {}
