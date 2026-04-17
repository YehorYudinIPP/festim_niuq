#!/usr/bin/env python3
"""
Bayesian Inverse UQ: PCE Surrogate as Likelihood for Posterior Estimation
=========================================================================

This script implements a Bayesian inversion pipeline for the FESTIM tritium
transport model using a Polynomial Chaos Expansion (PCE) surrogate as the
forward model inside an MCMC sampler.

**Pipeline overview**:

1. Build a PCE surrogate via an EasyVVUQ forward campaign with three uncertain
   energy parameters: ``E_D`` (activation energy), ``E_k`` (trapping energy),
   and ``E_kr`` (surface reaction energy).
2. Generate synthetic observations at known "true" parameter values + Gaussian
   noise.
3. Run ``emcee`` ensemble MCMC with the PCE surrogate as the log-likelihood
   evaluator.
4. Produce posterior diagnostics: trace plots, corner plots, posterior
   predictive checks, and summary statistics.

Usage::

    python bayesian_inverse_uq.py --config config/config_bayesian_ss.yaml \\
        --p-order 3 --noise-level 0.05 --n-walkers 32 --n-steps 5000

The script can also be imported and used programmatically via its main
classes and functions.
"""

import argparse
import os
import sys
import logging
import pickle
from datetime import datetime

import numpy as np

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt

# Local imports
from .util.utils import load_config, add_timestamp_to_filename
from .util.Encoder import AdvancedYAMLEncoder
from .util.Decoder import ScalarCSVDecoder
from .util.plotting import UQPlotter

import chaospy as cp
import easyvvuq as uq
from easyvvuq.actions import (
    Encode, Decode, ExecuteLocal, Actions, CreateRunDirectory,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("bayesian_inverse_uq")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(_handler)


# ============================================================================
# 1. Parameter & prior definitions
# ============================================================================
PARAM_NAMES = ["E_D", "E_k", "E_kr"]
QOI_COLS = ["total_tritium_release", "total_tritium_trapping"]


def read_prior_bounds(config, delta=0.2):
    """Read mean parameter values from config and compute uniform prior bounds.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    delta : float
        Relative half-width for the uniform prior around each mean.

    Returns
    -------
    dict
        ``{param_name: (lower, upper)}``
    dict
        ``{param_name: mean_value}``

    Raises
    ------
    ValueError
        If any required parameter (E_D, E_k, E_kr) has mean value of 0.0 or
        is missing from the configuration, which would indicate a
        misconfigured YAML file.
    """
    # Material index
    mat_idx = 0
    materials = config.get("materials", [{}])
    if isinstance(materials, list) and len(materials) > mat_idx:
        mat = materials[mat_idx]
    else:
        mat = materials

    def _mean(node):
        if isinstance(node, dict):
            return float(node.get("mean", 0.0))
        return float(node)

    means = {
        "E_D": _mean(mat.get("E_D", {})),
        "E_k": _mean(mat.get("E_k", {})),
        "E_kr": _mean(
            config.get("boundary_conditions", {})
                  .get("concentration", {})
                  .get("right", {})
                  .get("E_kr", {})
        ),
    }

    # Validate that all parameters are present and non-zero
    for name, mu in means.items():
        if mu == 0.0:
            raise ValueError(
                f"Parameter '{name}' has mean value 0.0 or is missing from "
                f"the config. Check the YAML file for a valid '{name}' entry."
            )

    bounds = {}
    for name, mu in means.items():
        lo = mu * (1.0 - delta)
        hi = mu * (1.0 + delta)
        bounds[name] = (lo, hi)

    logger.info("Prior bounds: %s", bounds)
    return bounds, means


def build_vary(bounds):
    """Build the EasyVVUQ ``vary`` dict of chaospy uniform distributions.

    Parameters
    ----------
    bounds : dict
        ``{param_name: (lower, upper)}``

    Returns
    -------
    dict
        ``{param_name: cp.Uniform(lo, hi)}``
    """
    vary = {}
    for name, (lo, hi) in bounds.items():
        vary[name] = cp.Uniform(lo, hi)
    return vary


# ============================================================================
# 2. EasyVVUQ forward campaign
# ============================================================================
def _prepare_execution_command():
    """Return an ExecuteLocal action for the FESTIM model runner."""
    from util.utils import validate_execution_setup
    python_exe, script_path = validate_execution_setup()
    cmd = f"{python_exe} {script_path} --config config.yaml"
    return ExecuteLocal(cmd)


def run_pce_campaign(config, config_file, bounds, p_order=3):
    """Set up, execute, and analyse a PCE forward campaign.

    Parameters
    ----------
    config : dict
        Parsed YAML configuration.
    config_file : str
        Path to the YAML config template.
    bounds : dict
        Prior bounds for each parameter.
    p_order : int
        Polynomial order for the PCE expansion.

    Returns
    -------
    results
        EasyVVUQ ``PCEAnalysis`` results object.
    campaign
        The campaign object (useful for accessing raw data).
    sampler
        The PCE sampler (needed for analysis).
    """
    vary = build_vary(bounds)

    # ---- Parameters ----
    parameters = {
        "E_D":  {"type": "float", "default": bounds["E_D"][0]},
        "E_k":  {"type": "float", "default": bounds["E_k"][0]},
        "E_kr": {"type": "float", "default": bounds["E_kr"][0]},
    }

    # ---- Encoder ----
    encoder = AdvancedYAMLEncoder(
        template_fname=config_file,
        target_filename="config.yaml",
        parameter_map={
            "E_D": "materials.E_D.mean",
            "E_k": "materials.E_k.mean",
            "E_kr": "boundary_conditions.concentration.right.E_kr.mean",
        },
        type_conversions={
            "E_D": float,
            "E_k": float,
            "E_kr": float,
        },
    )

    # ---- Decoder ----
    decoder = ScalarCSVDecoder(
        target_filename="output.csv",
        output_columns=QOI_COLS,
    )

    # ---- Execution ----
    execute = _prepare_execution_command()

    actions = Actions(
        CreateRunDirectory("run_dir"),
        Encode(encoder),
        execute,
        Decode(decoder),
    )

    # ---- Campaign ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign = uq.Campaign(
        name=f"bayesian_pce_{ts}_",
        params=parameters,
        actions=actions,
    )

    sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=p_order)
    campaign.set_sampler(sampler)

    n_runs = sampler.n_samples
    logger.info("PCE campaign: p_order=%d, n_runs=%d", p_order, n_runs)

    # ---- Execute ----
    campaign.execute().collate()

    # ---- Analyse ----
    analysis = uq.analysis.PCEAnalysis(sampler=sampler, qoi_cols=QOI_COLS)
    campaign.apply_analysis(analysis)
    results = campaign.get_last_analysis()

    logger.info("PCE campaign completed")
    for qoi in QOI_COLS:
        logger.info("  %s: mean=%.4e, std=%.4e",
                     qoi,
                     float(results.describe(qoi, "mean")),
                     float(results.describe(qoi, "std")))

    return results, campaign, sampler


# ============================================================================
# 3. Surrogate extraction
# ============================================================================
class PCESurrogate:
    """Lightweight wrapper around PCE analysis results.

    Evaluates the PCE polynomial surrogate for the two scalar QoIs at
    arbitrary parameter values.

    Parameters
    ----------
    results : EasyVVUQ PCEAnalysis results
    bounds : dict
        Prior bounds ``{name: (lo, hi)}`` — used for clipping.
    """

    def __init__(self, results, bounds):
        self.results = results
        self.bounds = bounds
        # Extract the chaospy polynomials
        self._polys = {}
        for qoi in QOI_COLS:
            try:
                self._polys[qoi] = results.raw_data["fit"][qoi]
            except (KeyError, TypeError):
                logger.warning(
                    "Could not extract PCE polynomial for '%s'; "
                    "falling back to results.surrogate()", qoi)
                self._polys = None
                break

    def __call__(self, theta):
        """Evaluate surrogate.

        Parameters
        ----------
        theta : array-like, shape (3,)
            ``[E_D, E_k, E_kr]``

        Returns
        -------
        numpy.ndarray, shape (2,)
            ``[total_tritium_release, total_tritium_trapping]``
        """
        E_D, E_k, E_kr = theta

        if self._polys is not None:
            out = np.array([
                float(self._polys[qoi](E_D, E_k, E_kr))
                for qoi in QOI_COLS
            ])
        else:
            # Fallback: evaluate via results.surrogate() (dict-based API)
            sample_dict = {"E_D": E_D, "E_k": E_k, "E_kr": E_kr}
            surrogate_fn = self.results.surrogate()
            result = surrogate_fn(sample_dict)
            out = np.array([float(result[qoi]) for qoi in QOI_COLS])

        return out


# ============================================================================
# 4. Synthetic evidence
# ============================================================================
def generate_synthetic_observations(surrogate, true_params, noise_level=0.05,
                                     seed=42):
    """Generate synthetic observations at known true parameter values.

    Parameters
    ----------
    surrogate : callable
        The PCE surrogate ``f(theta) -> [release, trapping]``.
    true_params : array-like, shape (3,)
        ``[E_D_true, E_k_true, E_kr_true]``
    noise_level : float
        Relative standard deviation of the Gaussian noise added to true QoIs.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    d_obs : numpy.ndarray, shape (2,)
        Noisy synthetic observations.
    sigma_obs : numpy.ndarray, shape (2,)
        Standard deviations of the observation noise.
    y_true : numpy.ndarray, shape (2,)
        Noise-free surrogate prediction at true parameters.
    """
    rng = np.random.default_rng(seed)
    y_true = surrogate(true_params)
    sigma_obs = noise_level * np.abs(y_true)
    noise = rng.normal(0.0, sigma_obs)
    d_obs = y_true + noise

    logger.info("True QoIs:   %s", y_true)
    logger.info("Noise sigma: %s", sigma_obs)
    logger.info("Observed:    %s", d_obs)

    return d_obs, sigma_obs, y_true


# ============================================================================
# 5. Bayesian inversion via MCMC (emcee)
# ============================================================================
def log_prior(theta, bounds):
    """Flat (uniform) log-prior within *bounds*, -inf outside."""
    for val, (lo, hi) in zip(theta, bounds.values()):
        if val < lo or val > hi:
            return -np.inf
    return 0.0


def log_likelihood(theta, surrogate, d_obs, inv_sigma2):
    """Gaussian log-likelihood using the PCE surrogate as the forward model.

    Parameters
    ----------
    theta : array-like, shape (3,)
    surrogate : callable
    d_obs : array-like, shape (2,)
    inv_sigma2 : array-like, shape (2,)
        ``1 / sigma_obs**2`` for each QoI.
    """
    y_pred = surrogate(theta)
    residual = d_obs - y_pred
    return -0.5 * np.sum(residual ** 2 * inv_sigma2)


def log_posterior(theta, bounds, surrogate, d_obs, inv_sigma2):
    """Log-posterior = log-prior + log-likelihood."""
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, surrogate, d_obs, inv_sigma2)
    return lp + ll


def run_mcmc(surrogate, bounds, d_obs, sigma_obs, means,
             n_walkers=32, n_steps=5000, n_burnin=1000, seed=42):
    """Run emcee ensemble MCMC.

    Parameters
    ----------
    surrogate : callable
    bounds : dict
    d_obs : array, shape (2,)
    sigma_obs : array, shape (2,)
    means : dict
        Mean parameter values (used to initialise walkers).
    n_walkers : int
    n_steps : int
    n_burnin : int
    seed : int

    Returns
    -------
    sampler : emcee.EnsembleSampler
    flat_samples : numpy.ndarray, shape (n_flat, 3)
    """
    import emcee

    ndim = len(PARAM_NAMES)
    inv_sigma2 = 1.0 / sigma_obs ** 2

    # Initialise walkers in a small ball around the means
    rng = np.random.default_rng(seed)
    p0_center = np.array([means[n] for n in PARAM_NAMES])
    spread = 0.01  # 1 % perturbation around center
    p0 = p0_center * (1.0 + spread * rng.standard_normal((n_walkers, ndim)))

    # Clip to stay within bounds
    for i, name in enumerate(PARAM_NAMES):
        lo, hi = bounds[name]
        p0[:, i] = np.clip(p0[:, i], lo, hi)

    logger.info("Starting emcee: ndim=%d, nwalkers=%d, nsteps=%d",
                ndim, n_walkers, n_steps)

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_posterior,
        args=(bounds, surrogate, d_obs, inv_sigma2),
    )
    sampler.run_mcmc(p0, n_steps, progress=True)

    # Discard burn-in and thin
    flat_samples = sampler.get_chain(discard=n_burnin, flat=True)

    logger.info("MCMC done. Effective samples: %d", len(flat_samples))

    return sampler, flat_samples


# ============================================================================
# 6. Diagnostics & visualisation
# ============================================================================
def posterior_summary(flat_samples, true_params, output_dir):
    """Compute and log posterior summary statistics.

    Parameters
    ----------
    flat_samples : ndarray, shape (N, 3)
    true_params : array-like, shape (3,)
    output_dir : str

    Returns
    -------
    dict
        Summary statistics.
    """
    summary = {}
    lines = []
    lines.append("=" * 60)
    lines.append("POSTERIOR SUMMARY")
    lines.append("=" * 60)
    for i, name in enumerate(PARAM_NAMES):
        chain = flat_samples[:, i]
        q16, q50, q84 = np.percentile(chain, [16, 50, 84])
        mu = np.mean(chain)
        std = np.std(chain)
        summary[name] = {
            "mean": mu, "std": std,
            "median": q50,
            "ci_16": q16, "ci_84": q84,
            "true": true_params[i],
        }
        lines.append(
            f"  {name:6s}: mean={mu:.6f}  std={std:.6f}  "
            f"median={q50:.6f}  [16%-84%]=[{q16:.6f}, {q84:.6f}]  "
            f"true={true_params[i]:.6f}"
        )
    lines.append("=" * 60)
    text = "\n".join(lines)
    logger.info("\n%s", text)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "posterior_summary.txt"), "w") as f:
        f.write(text + "\n")

    return summary


def plot_trace(sampler, true_params, output_dir, n_burnin=1000):
    """Plot MCMC trace for each parameter."""
    chain = sampler.get_chain()  # (n_steps, n_walkers, ndim)
    n_steps, n_walkers, ndim = chain.shape

    fig, axes = plt.subplots(ndim, 1, figsize=(10, 2.5 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        for w in range(min(n_walkers, 20)):  # plot subset of walkers
            ax.plot(chain[:, w, i], alpha=0.3, linewidth=0.5)
        ax.axhline(true_params[i], color="red", linestyle="--",
                    label=f"true={true_params[i]:.4f}")
        ax.axvline(n_burnin, color="grey", linestyle=":", label="burn-in")
        ax.set_ylabel(name)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Step")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "trace_plot.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Trace plot saved to %s", path)


def plot_corner(flat_samples, true_params, output_dir):
    """Corner / pair plot of the posterior."""
    try:
        import corner as corner_lib
    except ImportError:
        logger.warning("corner package not installed; skipping corner plot")
        return

    fig = corner_lib.corner(
        flat_samples,
        labels=PARAM_NAMES,
        truths=true_params,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "corner_plot.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Corner plot saved to %s", path)


def plot_marginals(flat_samples, true_params, bounds, output_dir):
    """Plot marginal posterior histograms with true values."""
    ndim = len(PARAM_NAMES)
    fig, axes = plt.subplots(1, ndim, figsize=(5 * ndim, 4))
    if ndim == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, PARAM_NAMES)):
        ax.hist(flat_samples[:, i], bins=60, density=True,
                alpha=0.7, color="steelblue", edgecolor="white")
        ax.axvline(true_params[i], color="red", linestyle="--",
                    linewidth=2, label=f"true = {true_params[i]:.4f}")
        lo, hi = bounds[name]
        ax.axvspan(lo, hi, color="grey", alpha=0.1)
        ax.set_xlim(lo, hi)
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.set_title(f"Posterior: {name}")

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "marginal_posteriors.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Marginal posteriors saved to %s", path)


def plot_posterior_predictive(flat_samples, surrogate, d_obs, sigma_obs,
                               output_dir, n_draws=500, seed=123):
    """Posterior predictive check: predict QoIs from posterior samples."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flat_samples), size=min(n_draws, len(flat_samples)),
                     replace=False)

    predictions = np.array([surrogate(flat_samples[j]) for j in idx])

    fig, axes = plt.subplots(1, len(QOI_COLS), figsize=(6 * len(QOI_COLS), 4))
    if len(QOI_COLS) == 1:
        axes = [axes]

    for i, (ax, qoi) in enumerate(zip(axes, QOI_COLS)):
        ax.hist(predictions[:, i], bins=50, density=True,
                alpha=0.7, color="steelblue", edgecolor="white",
                label="Posterior predictive")
        ax.axvline(d_obs[i], color="red", linewidth=2, linestyle="--",
                    label=f"Observed = {d_obs[i]:.3e}")
        ax.axvspan(d_obs[i] - 2 * sigma_obs[i],
                   d_obs[i] + 2 * sigma_obs[i],
                   color="red", alpha=0.1,
                   label="±2σ obs")
        ax.set_xlabel(qoi)
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior predictive: {qoi}")
        ax.legend(fontsize=8)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "posterior_predictive.pdf")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Posterior predictive plot saved to %s", path)


def compute_diagnostics(sampler, n_burnin=1000):
    """Compute MCMC diagnostics: acceptance fraction and autocorrelation."""
    acc = np.mean(sampler.acceptance_fraction)
    logger.info("Mean acceptance fraction: %.3f", acc)

    try:
        tau = sampler.get_autocorr_time(quiet=True)
        logger.info("Autocorrelation times: %s", tau)
    except Exception:
        tau = None
        logger.warning("Could not estimate autocorrelation time")

    return {"acceptance_fraction": acc, "autocorrelation_time": tau}


# ============================================================================
# 7. Standalone Bayesian inversion (no EasyVVUQ campaign needed)
# ============================================================================
def run_standalone_bayesian(config, config_file, args):
    """Run the full Bayesian inversion pipeline.

    This is the main entry point that either:
      (a) Runs the EasyVVUQ PCE campaign and then MCMC, or
      (b) Loads a previously-saved PCE surrogate and runs MCMC only.

    Parameters
    ----------
    config : dict
    config_file : str
    args : argparse.Namespace
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("bayesian_results", f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # ---- Prior bounds ----
    bounds, means = read_prior_bounds(config, delta=args.delta)
    true_params = np.array([means[n] for n in PARAM_NAMES])

    # ---- PCE surrogate ----
    surrogate_pickle = args.surrogate_pickle

    if surrogate_pickle and os.path.isfile(surrogate_pickle):
        logger.info("Loading pre-built surrogate from %s", surrogate_pickle)
        with open(surrogate_pickle, "rb") as f:
            saved = pickle.load(f)
        surrogate = saved["surrogate"]
    else:
        logger.info("Running PCE forward campaign (p_order=%d)...", args.p_order)
        results, campaign, sampler_pce = run_pce_campaign(
            config, config_file, bounds, p_order=args.p_order)

        surrogate = PCESurrogate(results, bounds)

        # Save surrogate for reuse
        pkl_path = os.path.join(output_dir, "pce_surrogate.pickle")
        with open(pkl_path, "wb") as f:
            pickle.dump({"surrogate": surrogate, "bounds": bounds,
                         "means": means, "results": results}, f)
        logger.info("PCE surrogate saved to %s", pkl_path)

    # ---- Synthetic observations ----
    d_obs, sigma_obs, y_true = generate_synthetic_observations(
        surrogate, true_params,
        noise_level=args.noise_level,
        seed=args.seed,
    )

    # Save synthetic data
    np.savez(
        os.path.join(output_dir, "synthetic_data.npz"),
        d_obs=d_obs, sigma_obs=sigma_obs, y_true=y_true,
        true_params=true_params, param_names=PARAM_NAMES,
    )

    # ---- MCMC ----
    emcee_sampler, flat_samples = run_mcmc(
        surrogate, bounds, d_obs, sigma_obs, means,
        n_walkers=args.n_walkers,
        n_steps=args.n_steps,
        n_burnin=args.n_burnin,
        seed=args.seed,
    )

    # ---- Diagnostics ----
    diag = compute_diagnostics(emcee_sampler, n_burnin=args.n_burnin)
    summary = posterior_summary(flat_samples, true_params, output_dir)

    # ---- Plots ----
    plot_trace(emcee_sampler, true_params, output_dir, n_burnin=args.n_burnin)
    plot_corner(flat_samples, true_params, output_dir)
    plot_marginals(flat_samples, true_params, bounds, output_dir)
    plot_posterior_predictive(flat_samples, surrogate, d_obs, sigma_obs,
                              output_dir)

    # ---- Save full results ----
    results_path = os.path.join(output_dir, "mcmc_results.pickle")
    with open(results_path, "wb") as f:
        pickle.dump({
            "flat_samples": flat_samples,
            "d_obs": d_obs,
            "sigma_obs": sigma_obs,
            "y_true": y_true,
            "true_params": true_params,
            "bounds": bounds,
            "means": means,
            "summary": summary,
            "diagnostics": diag,
        }, f)
    logger.info("Full MCMC results saved to %s", results_path)
    logger.info("All outputs in: %s", output_dir)

    return output_dir


# ============================================================================
# 8. CLI
# ============================================================================
def parse_args(argv=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bayesian Inverse UQ: PCE surrogate + emcee MCMC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config_bayesian_ss.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--p-order", type=int, default=3,
        help="Polynomial order for PCE surrogate",
    )
    parser.add_argument(
        "--delta", type=float, default=0.2,
        help="Relative half-width for uniform prior around means",
    )
    parser.add_argument(
        "--noise-level", type=float, default=0.05,
        help="Relative noise level for synthetic observations",
    )
    parser.add_argument(
        "--n-walkers", type=int, default=32,
        help="Number of emcee walkers",
    )
    parser.add_argument(
        "--n-steps", type=int, default=5000,
        help="Number of MCMC steps per walker",
    )
    parser.add_argument(
        "--n-burnin", type=int, default=1000,
        help="Number of burn-in steps to discard",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--surrogate-pickle", type=str, default=None,
        help="Path to a pre-built PCE surrogate pickle (skip campaign)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Main entry point."""
    args = parse_args(argv)

    config = load_config(args.config)
    if config is None:
        logger.error("Failed to load config from %s", args.config)
        sys.exit(1)

    output_dir = run_standalone_bayesian(config, args.config, args)
    print(f"\n✓ Bayesian inverse UQ complete. Results in: {output_dir}\n")


if __name__ == "__main__":
    main()
