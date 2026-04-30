"""
Pytest unit tests for the analytical verification solutions in gfederici1991.py.

Tests cover:
- CarlsJaeger1959: convergence to the known steady-state profile.
- CarlsJaeger1959: ValueError for non-positive time.
- Crank1975: no NaN/Inf at r=0 (regression for the singularity fix).
- Crank1975: correctness of the c_0 scaling.
- Crank1975: ValueError for non-positive time.

All tests are pure Python/NumPy — no FESTIM or FEniCSx required.
"""

import sys
import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Make the verification module importable when tests/ is on sys.path
# (pytest.ini_options sets testpaths = ["tests"], so running pytest from the
# repo root should resolve the import automatically; the next two lines are a
# belt-and-suspenders fallback for direct execution).
# ---------------------------------------------------------------------------
_tests_dir = os.path.dirname(os.path.abspath(__file__))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)

from verification.gfederici1991 import CarlsJaeger1959, Crank1975  # noqa: E402


# ===========================================================================
# CarlsJaeger1959
# ===========================================================================


class TestCarlsJaeger1959:
    """Tests for the Carslaw & Jaeger 1959 diffusion-with-source solution."""

    # Default physical parameters
    D = 1.0
    G = 1.0
    a = 1.0
    m = 64

    def _steady_state(self, r):
        """Exact steady-state profile c_ss(r) = G/(6D) * (a² - r²)."""
        return self.G / (6.0 * self.D) * (self.a**2 - r**2)

    # -----------------------------------------------------------------------
    def test_raises_for_nonpositive_time(self):
        """ValueError must be raised when t <= 0."""
        with pytest.raises(ValueError, match="greater than 0"):
            CarlsJaeger1959(t=0.0, D=self.D, G=self.G, a=self.a)
        with pytest.raises(ValueError, match="greater than 0"):
            CarlsJaeger1959(t=-1.0, D=self.D, G=self.G, a=self.a)

    # -----------------------------------------------------------------------
    def test_returns_array_of_correct_length(self):
        """Output array must have *m* elements."""
        result = CarlsJaeger1959(t=1.0, D=self.D, G=self.G, a=self.a, m=self.m)
        assert len(result) == self.m

    # -----------------------------------------------------------------------
    def test_steady_state_convergence(self):
        """At large t the solution must be close to the steady-state profile."""
        # Use a very large time to ensure the transient terms have decayed.
        t_large = 50.0  # many diffusion timescales: τ = a²/(π²D) ≈ 0.1
        result = CarlsJaeger1959(t=t_large, D=self.D, G=self.G, a=self.a, m=self.m, n=64)
        r = np.linspace(0, self.a, self.m)
        expected = self._steady_state(r)
        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-6)

    # -----------------------------------------------------------------------
    def test_no_nan_or_inf(self):
        """No NaN or Inf should appear anywhere in the output, including r=0."""
        result = CarlsJaeger1959(t=0.1, D=self.D, G=self.G, a=self.a, m=self.m)
        assert np.all(np.isfinite(result)), "Output contains NaN or Inf values."

    # -----------------------------------------------------------------------
    def test_concentration_at_boundary_zero(self):
        """Concentration at the outer boundary r=a must be zero (Dirichlet BC)."""
        result = CarlsJaeger1959(t=1.0, D=self.D, G=self.G, a=self.a, m=self.m)
        # Last element corresponds to r = a
        assert abs(result[-1]) < 1e-10, f"Boundary value not zero: {result[-1]}"

    # -----------------------------------------------------------------------
    def test_scaling_with_source_term(self):
        """Concentration should scale linearly with G (for fixed D, a, t)."""
        t = 10.0
        c1 = CarlsJaeger1959(t=t, D=self.D, G=1.0, a=self.a, m=self.m)
        c2 = CarlsJaeger1959(t=t, D=self.D, G=2.0, a=self.a, m=self.m)
        np.testing.assert_allclose(c2, 2.0 * c1, rtol=1e-12)

    # -----------------------------------------------------------------------
    def test_scaling_with_diffusivity(self):
        """At steady state, concentration should scale as 1/D."""
        t_large = 50.0
        c1 = CarlsJaeger1959(t=t_large, D=1.0, G=self.G, a=self.a, m=self.m, n=64)
        c2 = CarlsJaeger1959(t=t_large, D=2.0, G=self.G, a=self.a, m=self.m, n=64)
        # atol handles r=a where both values are zero (relative comparison is 0/0)
        np.testing.assert_allclose(c2, 0.5 * c1, rtol=1e-4, atol=1e-30)


# ===========================================================================
# Crank1975
# ===========================================================================


class TestCrank1975:
    """Tests for the Crank 1975 release-from-preloaded-sphere solution."""

    D = 1.0
    c_0 = 1.0
    a = 1.0
    m = 64

    # -----------------------------------------------------------------------
    def test_raises_for_nonpositive_time(self):
        """ValueError must be raised when t <= 0."""
        with pytest.raises(ValueError, match="greater than 0"):
            Crank1975(t=0.0, D=self.D, c_0=self.c_0, a=self.a)
        with pytest.raises(ValueError, match="greater than 0"):
            Crank1975(t=-5.0, D=self.D, c_0=self.c_0, a=self.a)

    # -----------------------------------------------------------------------
    def test_returns_array_of_correct_length(self):
        """Output array must have *m* elements."""
        result = Crank1975(t=0.1, D=self.D, c_0=self.c_0, a=self.a, m=self.m)
        assert len(result) == self.m

    # -----------------------------------------------------------------------
    def test_no_nan_or_inf_including_r0(self):
        """
        Regression test for the r=0 singularity fix.

        The Fourier-series formula contains sin(k π r / a) / r, which is
        formally 0/0 at r=0.  After the fix, the value at r=0 must be a
        finite real number.
        """
        result = Crank1975(t=0.1, D=self.D, c_0=self.c_0, a=self.a, m=self.m)
        assert np.all(np.isfinite(result)), (
            f"Output contains NaN or Inf.  r=0 value: {result[0]}"
        )

    # -----------------------------------------------------------------------
    def test_concentration_at_boundary_zero(self):
        """Concentration at r=a must be zero (homogeneous Dirichlet BC)."""
        result = Crank1975(t=0.1, D=self.D, c_0=self.c_0, a=self.a, m=self.m)
        assert abs(result[-1]) < 1e-8, f"Boundary value not zero: {result[-1]}"

    # -----------------------------------------------------------------------
    def test_decay_over_time(self):
        """Average concentration must decrease monotonically for t1 < t2."""
        t1, t2, t3 = 0.01, 0.1, 0.5
        r1 = Crank1975(t=t1, D=self.D, c_0=self.c_0, a=self.a, m=self.m)
        r2 = Crank1975(t=t2, D=self.D, c_0=self.c_0, a=self.a, m=self.m)
        r3 = Crank1975(t=t3, D=self.D, c_0=self.c_0, a=self.a, m=self.m)
        assert np.mean(r1) > np.mean(r2) > np.mean(r3), (
            "Concentration should decay over time for the release problem."
        )

    # -----------------------------------------------------------------------
    def test_scaling_with_c0(self):
        """Concentration must scale linearly with the initial value c_0."""
        t = 0.1
        res1 = Crank1975(t=t, D=self.D, c_0=1.0, a=self.a, m=self.m)
        res3 = Crank1975(t=t, D=self.D, c_0=3.0, a=self.a, m=self.m)
        np.testing.assert_allclose(res3, 3.0 * res1, rtol=1e-12)

    # -----------------------------------------------------------------------
    def test_r0_value_matches_limit(self):
        """
        Value at r=0 should match the L'Hôpital limit to reasonable precision.

        The implementation sets c(0, t) via L'Hôpital's rule:
            lim_{r→0} sin(kπr/a)/r = kπ/a

        so the r=0 value for the actual implementation equals:
            c_0 * (-2a/π) * sum_{k=1}^{n} (-1)^k * exp(-k²π²Dt/a²) * (kπ/a)
          = c_0 * 2 * sum_{k=1}^{n} (-1)^{k+1} * k * exp(-k²π²Dt/a²)
        """
        import math

        t = 0.1
        n = 32
        result = Crank1975(t=t, D=self.D, c_0=self.c_0, a=self.a, m=self.m, n=n)

        # Compute expected limit independently (must include the k factor)
        expected_r0 = 0.0
        for k in range(1, n + 1):
            expected_r0 += (-1) ** (k + 1) * 2.0 * k * math.exp(-k**2 * np.pi**2 * self.D * t / self.a**2)
        expected_r0 *= self.c_0

        assert abs(result[0] - expected_r0) < 1e-8 * max(abs(expected_r0), 1.0), (
            f"r=0 value {result[0]:.6e} deviates from expected limit {expected_r0:.6e}"
        )
