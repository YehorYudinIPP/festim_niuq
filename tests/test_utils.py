"""
Tests for utility functions in uq/util/utils.py.
"""
import os
import tempfile
import pytest
import yaml
import numpy as np

from uq.util.utils import (
    load_config,
    add_timestamp_to_filename,
    compute_absolute_tolerance,
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
        name_part = result[len("test_"):-len(".txt")]
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
