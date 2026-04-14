"""
Scientific verification tests for FESTIM-NIUQ.

These tests verify the scientific correctness of the UQ pipeline by comparing
results against known analytical solutions and reference data. They require
FESTIM and FEniCSx to be installed and are intended to be run in a full
environment (not in CI with mocks).

Verification strategy:
    1. Analytical benchmark tests: compare FESTIM solutions against
       Carslaw & Jaeger (1959) analytical solutions for diffusion in a sphere
    2. Method-of-manufactured-solutions (MMS): verify convergence rates
    3. UQ convergence: verify that PCE Sobol indices converge with polynomial order
    4. Surrogate accuracy: verify PCE surrogate matches direct sampling statistics

To run these tests (requires FESTIM + FEniCSx):
    pytest tests/test_scientific.py -v --runslow

Mark all tests with @pytest.mark.scientific so they can be skipped in CI.
"""
import pytest
import numpy as np


# Custom marker for scientific tests that need FESTIM
scientific = pytest.mark.skipif(
    True,  # Set to False when running with FESTIM installed
    reason="Scientific tests require FESTIM and FEniCSx installation"
)


@scientific
class TestAnalyticalBenchmarkCJ1959:
    """
    Verification against Carslaw & Jaeger (1959) analytical solutions.

    Reference: Carslaw, H.S. & Jaeger, J.C., "Conduction of Heat in Solids",
    Oxford University Press, 1959.

    The test case is diffusion in a sphere with:
    - Constant volumetric source
    - Dirichlet outer boundary condition
    - Initially zero concentration

    The analytical steady-state solution for concentration in a sphere of
    radius R with source G, diffusivity D, and zero concentration at r=R is:

        c(r) = G/(6D) * (R^2 - r^2)

    TODO: Implement when FESTIM is available in the test environment.
    """

    def test_steady_state_concentration_profile(self):
        """
        Compare FESTIM steady-state solution with analytical solution.

        Steps:
            1. Run FESTIM model with config/config.uq_test_cj1959.yaml
            2. Extract steady-state concentration profile
            3. Compare with c(r) = G/(6D) * (R^2 - r^2)
            4. Assert L2 error norm < tolerance
        """
        pytest.skip("Requires FESTIM installation")
        # Implementation outline:
        # from festim_model.Model import Model
        # from uq.util.utils import load_config
        #
        # config = load_config("uq/config/config.uq_test_cj1959.yaml")
        # model = Model()
        # model.run(config)
        # c_numerical = model.get_concentration_profile()
        # r = model.get_mesh_coordinates()
        #
        # G = config["source_terms"]["concentration"]["value"]["mean"]
        # D = config["materials"][0]["D_0"]["mean"]
        # R = config["geometry"]["domains"][0]["length"]
        # c_analytical = G / (6 * D) * (R**2 - r**2)
        #
        # l2_error = np.sqrt(np.mean((c_numerical - c_analytical)**2))
        # assert l2_error / np.max(c_analytical) < 1e-3

    def test_mesh_convergence(self):
        """
        Verify spatial convergence rate by refining the mesh.

        Steps:
            1. Run simulations with n_elements = [64, 128, 256, 512]
            2. Compute L2 error vs analytical for each
            3. Fit convergence rate
            4. Assert rate >= 1.8 (near second-order for linear FE)
        """
        pytest.skip("Requires FESTIM installation")


@scientific
class TestUQConvergence:
    """
    Verify convergence of the UQ pipeline with increasing polynomial order.

    For a smooth response surface, PCE should converge exponentially with
    polynomial order. We verify:
    1. Mean converges to a stable value
    2. Variance converges to a stable value
    3. Sobol indices converge to stable values
    """

    def test_pce_mean_convergence(self):
        """
        Run PCE with p=1,2,3 and verify mean converges.

        The relative change in mean between p=2 and p=3 should be small
        (< 1% for smooth problems).
        """
        pytest.skip("Requires FESTIM installation")

    def test_pce_sobol_convergence(self):
        """
        Run PCE with p=1,2,3 and verify Sobol indices converge.

        The first-order Sobol indices should stabilise by p=3 for
        this problem (6 parameters, smooth Arrhenius-type responses).
        """
        pytest.skip("Requires FESTIM installation")

    def test_sobol_indices_sum_to_one(self):
        """
        Verify that total Sobol indices sum to approximately 1.

        For independent parameters, sum of first-order Sobol indices <= 1,
        and total Sobol indices should account for all variance.
        """
        pytest.skip("Requires FESTIM installation")


@scientific
class TestSurrogateAccuracy:
    """
    Verify PCE surrogate accuracy against direct model evaluations.

    The PCE surrogate should accurately predict the model output at
    random test points not used in the training set.
    """

    def test_surrogate_prediction_at_mean(self):
        """
        Evaluate the PCE surrogate at the mean parameter values and
        compare with a direct FESTIM evaluation.
        """
        pytest.skip("Requires FESTIM installation")

    def test_leave_one_out_cross_validation(self):
        """
        Compute leave-one-out cross-validation error for the PCE surrogate.
        The LOOCV error should be small (< 5% relative).
        """
        pytest.skip("Requires FESTIM installation")
