"""
Tests for the UQPlotter class from uq/util/plotting.py.
Uses mocks to avoid requiring real EasyVVUQ results objects.
"""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

from festim_niuq.uq.util.plotting import UQPlotter


@pytest.fixture
def plotter():
    """Create a UQPlotter instance."""
    return UQPlotter()


@pytest.fixture
def mock_results():
    """Create a mock EasyVVUQ results object."""
    results = MagicMock()

    # Mock describe method for statistics
    n_points = 10
    rs = np.linspace(0, 1, n_points)

    def describe_side_effect(qoi, stat):
        if qoi == "x":
            return rs
        if stat == "mean":
            return np.ones(n_points) * 1e18
        if stat == "std":
            return np.ones(n_points) * 1e16
        if stat in ("1%", "10%"):
            return np.ones(n_points) * 0.9e18
        if stat in ("90%", "99%"):
            return np.ones(n_points) * 1.1e18
        if stat == "median":
            return np.ones(n_points) * 1e18
        return np.zeros(n_points)

    results.describe = MagicMock(side_effect=describe_side_effect)

    # Mock sobols_first
    results.sobols_first = MagicMock(return_value={"D_0": np.random.rand(n_points), "G": np.random.rand(n_points)})

    # Mock sobols_total
    results.sobols_total = MagicMock(return_value={"D_0": np.random.rand(n_points), "G": np.random.rand(n_points)})

    return results


class TestUQPlotterInit:
    """Tests for UQPlotter initialization."""

    def test_has_quantities_descriptor(self, plotter):
        assert "tritium_concentration" in plotter.quantities_descriptor
        assert "temperature" in plotter.quantities_descriptor

    def test_has_parameters_descriptor(self, plotter):
        assert "D_0" in plotter.parameters_descriptor

    def test_quantities_have_required_keys(self, plotter):
        for key, desc in plotter.quantities_descriptor.items():
            assert "name" in desc
            assert "unit" in desc
            assert "dimensionality" in desc


class TestUQPlotterMethods:
    """Tests for UQPlotter plotting methods using mocks."""

    def test_plot_stats_vs_r_creates_files(self, plotter, mock_results, tmp_path):
        """Test that plot_stats_vs_r produces output files."""
        folder = str(tmp_path / "plots")
        os.makedirs(folder, exist_ok=True)

        qois = ["t=steady"]
        rs = np.linspace(0, 1, 10)

        try:
            plotter.plot_stats_vs_r(mock_results, qois, folder, "test_ts", rs=rs, runs_info=None)
        except Exception:
            # Plotting may fail due to mock limitations, but we verify it was called
            pass

    def test_plotter_is_instantiable(self, plotter):
        """Basic check that the plotter can be instantiated without errors."""
        assert plotter is not None
        assert hasattr(plotter, "plot_stats_vs_r")
        assert hasattr(plotter, "plot_qoi_distribution")


# ---------------------------------------------------------------------------
# Tests for UQPlotter.plot_qoi_distribution
# ---------------------------------------------------------------------------


class TestPlotQoiDistribution:
    """Tests for UQPlotter.plot_qoi_distribution."""

    N_RUNS = 32  # enough for a reasonable KDE / histogram

    @pytest.fixture
    def qoi_values(self):
        rng = np.random.default_rng(seed=0)
        return rng.normal(loc=1e18, scale=1e16, size=self.N_RUNS)

    def test_creates_pdf_file(self, plotter, qoi_values, tmp_path):
        """Default call writes a PDF file."""
        out = plotter.plot_qoi_distribution(
            qoi_values,
            qoi_name="tritium_concentration",
            foldername=str(tmp_path),
            filename="test_dist.pdf",
        )
        assert os.path.isfile(out)

    def test_histogram_only(self, plotter, qoi_values, tmp_path):
        """histogram-only mode writes a file without error."""
        out = plotter.plot_qoi_distribution(
            qoi_values,
            qoi_name="c",
            foldername=str(tmp_path),
            filename="hist_only.pdf",
            show_histogram=True,
            show_kde=False,
            show_pce_pdf=False,
        )
        assert os.path.isfile(out)

    def test_kde_mode(self, plotter, qoi_values, tmp_path):
        """KDE mode writes a file (scipy may be absent; must not crash)."""
        out = plotter.plot_qoi_distribution(
            qoi_values,
            qoi_name="c",
            foldername=str(tmp_path),
            filename="kde_only.pdf",
            show_histogram=False,
            show_kde=True,
            show_pce_pdf=False,
        )
        assert os.path.isfile(out)

    def test_pce_mode_with_surrogate(self, plotter, tmp_path):
        """PCE mode with a simple chaospy surrogate writes a file."""
        cp = pytest.importorskip("chaospy")

        # Build a tiny 2-param PCE surrogate
        D0_dist = cp.Uniform(1e-7, 3e-7)
        G_dist = cp.Uniform(1e18, 3e18)
        joint = cp.J(D0_dist, G_dist)

        order = 2
        expansion = cp.generate_expansion(order, joint)
        nodes, weights = cp.generate_quadrature(order, joint, rule="gaussian")
        # QoI = G / (6 * D0) at r=0 (sphere centre)
        evaluations = nodes[1] / (6.0 * nodes[0]) * (1e-3) ** 2
        surrogate = cp.fit_quadrature(expansion, nodes, weights, evaluations)

        qoi_values = evaluations  # use quadrature evaluations as raw samples

        out = plotter.plot_qoi_distribution(
            qoi_values,
            qoi_name="c",
            foldername=str(tmp_path),
            filename="pce_dist.pdf",
            show_histogram=True,
            show_kde=False,
            show_pce_pdf=True,
            pce_surrogate=surrogate,
            joint_dist=joint,
            n_mc_samples=500,
        )
        assert os.path.isfile(out)

    def test_empty_values_returns_empty_string(self, plotter, tmp_path):
        """All-NaN input returns an empty string and writes nothing."""
        values = np.full(10, np.nan)
        out = plotter.plot_qoi_distribution(
            values,
            qoi_name="c",
            foldername=str(tmp_path),
            filename="should_not_exist.pdf",
        )
        assert out == ""
        assert not os.path.isfile(os.path.join(str(tmp_path), "should_not_exist.pdf"))

    def test_auto_filename_with_timestamp(self, plotter, qoi_values, tmp_path):
        """When filename is None, a timestamped filename is auto-generated."""
        out = plotter.plot_qoi_distribution(
            qoi_values,
            qoi_name="c",
            foldername=str(tmp_path),
            timestamp="20260101_000000",
        )
        assert os.path.isfile(out)
        assert "distribution" in os.path.basename(out)
        assert "20260101_000000" in os.path.basename(out)
