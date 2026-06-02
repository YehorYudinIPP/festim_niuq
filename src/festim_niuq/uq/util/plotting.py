"""
This module contains functions for plotting uncertainty quantification (UQ) results from the FESTIM model.
It includes functions to plot uncertainty as a function of radius, time, and Sobol indices,
as well as functions to visualize statistics of the results.
It is designed to work with EasyVVUQ and FESTIM libraries, providing a way to visualize the results of UQ campaigns.

Created by: Yehor Yudin
Date: July 2025
"""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to avoid display connection issues
import matplotlib.pyplot as plt
import numpy as np

import itertools
import json
import logging
import os
import re

from .utils import add_timestamp_to_filename

logger = logging.getLogger(__name__)


class UQPlotter:
    """
    A class to encapsulate UQ plotting functionalities.
    """

    def __init__(self):

        self.quantities_descriptor = {
            "tritium_inventory": {
                "name": "Tritium Inventory",
                "unit": "T",
                "dimensionality": "0d",
                "description": "Total tritium inventory in the sample",
            },
            "tritium_concentration": {
                "name": "Tritium Concentration",
                "unit": "m^-3",
                "dimensionality": "1d",
                "description": "Tritium concentration in the volume",
            },
            "temperature": {
                "name": "Temperature",
                "unit": "K",
                "dimensionality": "1d",
                "description": "Temperature distribution at a point",
            },
        }

        self.parameters_descriptor = {
            "D_0": {
                "name": "Diffusion Coefficient",
                "unit": "m^{2}/s",
                "dimensionality": "0d",
                "description": "Base diffusion coefficient",
            },
            "E_D": {
                "name": "Activation Energy",
                "unit": "J/mol",
                "dimensionality": "0d",
                "description": "Activation energy for diffusion",
            },
            "G": {
                "name": "Tritium Generation Rate",
                "unit": "[m^-{3} s^-{1}]",
                "dimensionality": "0d",
                "description": "Volumetric Tritium Generation Rate",
            },
            "kappa": {
                "name": "Heat Transfer Coefficient",
                "unit": "W/(m^{2} K)",
                "dimensionality": "0d",
                "description": "Heat transfer coefficient",
            },
            "h_coev": {
                "name": "Boundary Heat Transfer Coefficient",
                "unit": "W/(m^{2} K)",
                "dimensionality": "0d",
                "description": "Heat transfer coefficient at the boundary",
            },
            "Q": {
                "name": "Heat Source Term",
                "unit": "W/m^{3}",
                "dimensionality": "0d",
                "description": "Volumetric heat source term",
            },
            "E_kr": {
                "name": "Surface Recombination Energy",
                "unit": "J/mol",
                "dimensionality": "0d",
                "description": "Surface recombination energy",
            },
            "k_r0": {
                "name": "Surface recombination rate constant",
                "unit": "m/s",
                "dimensionality": "0d",
                "description": "Surface recombination rate constant",
            },
        }

        self.scale_descriptor = {
            "plot": "Linear scale",
            "semilogy": "Logarithmic scale (Y-axis)",
            # Add more scale types if needed
        }

        self.quantity = "tritium_concentration"  # Default quantity to plot

    @staticmethod
    def _format_label_with_unit(name: str, unit: str = "") -> str:
        if not unit:
            return f"{name}"
        return f"{name} [${unit}$]"

    def plot_unc_vs_r(
        self, r, y, sy, y10, y90, y01=None, y99=None, qoi_name: str = "", foldername: str = "", filename: str = "", runs_info=None
    ):
        """
        Plot uncertainty in the results as a function of radius (spatial coordinates).

        Parameters:
        - r: array of radius / 1D coordinate values
        - y: array of mean values at each radius
        - sy: array of standard deviation values at each radius
        - y10: array of 10% quantile values at each radius
        - y90: array of 90% quantile values at each radius
        - y01: array of 1% quantile values at each radius (optional)
        - y99: array of 99% quantile values at each radius (optional)
        - qoi_name: name of the quantity of interest (QoI) for labeling
        - foldername: folder to save the plot
        - filename: name of the file to save the plot
        - runs_info: information about individual runs, if available (optional); should be a list of dictionaries, with 'id' and 'results'
        """

        # making an array of plot for different axis scales
        plot_types = ["plot", "semilogy"]  # Add more plot types if needed
        n_plots = len(plot_types)

        fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 8, 6))

        # Iterative over type of axis scales
        for i, plot_func_name in enumerate(plot_types):

            # Chosing the plotting function based on the plot_func_name (axis scale)
            plot_func = getattr(axs[i], plot_func_name, None)
            if plot_func is None:
                raise ValueError(f"Plot function '{plot_func_name}' is not supported. Use 'plot' or 'semilogy'.")

            # Plotting the mean values with error bars
            plot_func(r, y, label=f"<y> at {qoi_name}")

            # Plotting the standard deviation as a shaded area
            axs[i].fill_between(r, y - sy, y + sy, alpha=0.3, label="+/- STD")

            # Plotting the 1% and 99% quantiles as a wide shaded area.
            if y01 is not None and y99 is not None:
                y01_arr = np.asarray(y01)
                y99_arr = np.asarray(y99)
                if y01_arr.shape == y99_arr.shape and np.all(np.isfinite(y01_arr)) and np.all(np.isfinite(y99_arr)):
                    axs[i].fill_between(r, y01_arr, y99_arr, alpha=0.12, label="1%-99%")

            # Plotting the 10% and 90% quantiles as a shaded area
            axs[i].fill_between(r, y10, y90, alpha=0.1, label="10% - 90%")

            # Plotting individual trajectories for each run if runs_info is provided

            if runs_info is not None:
                logger.debug(
                    f" > Plotting individual trajectories for each run in {qoi_name} at '{self.scale_descriptor[plot_func_name]}' scale"
                )

                # Iterating over individual runs
                for run_id, run_info in runs_info:
                    # print(f" >> Plotting run {run_id} for {qoi_name}")
                    # print(f" >> Run {run_id} info: {run_info}")

                    # Checking if individual run has non-empty results
                    if "result" in run_info:

                        # Deserializing the result to a dictionary
                        result_str = run_info["result"]  # This is a string

                        result_dict = json.loads(result_str)  # This SHOULD BE a dictionary
                        # print(f" >> Run {run_id} result_dict type: {type(result_dict)}, content: {result_dict}")

                        # Plotting the individual trajectory for the current run
                        plot_func(
                            result_dict["x"],
                            result_dict[qoi_name],
                            alpha=0.5,
                            color="gray",
                            label=f"Individual runs trajectories" if run_id == 1 else None,
                        )

                    else:
                        print(f"Run {run_id} does not have 'result' key, skipping individual trajectory plotting.")
                logger.debug(" > Individual trajectories plotted for each run.")
            else:
                print("No runs_info provided, skipping individual trajectories plotting.")

            # Setting the title and labels for the plot
            axs[i].set_title(f"Uncertainty at {qoi_name} as a function of radius, in '{plot_func_name}' scale")
            length_unit = self.quantities_descriptor.get("x", {}).get("unit", "m")
            axs[i].set_xlabel(f"Radius, [{length_unit}]")
            qty_name = self.quantities_descriptor.get(self.quantity, {}).get("name", self.quantity)
            qty_unit = self.quantities_descriptor.get(self.quantity, {}).get("unit", "")
            axs[i].set_ylabel(f"{self._format_label_with_unit(qty_name, qty_unit)} at {qoi_name}")

            axs[i].legend(loc="best")
            axs[i].grid(True)

        # Save the figure with a bespoke filename
        fig.savefig(f"{foldername}/bespoke_{filename}")

        plt.close()  # Close the plot to avoid display issues in some environments
        return 0

    def plot_unc_qoi(self, stats_dict_s: dict, qoi_name: str, foldername: str = "", filename: str = "", r_ind: int = 0):
        """
        Plot uncertainty in the specific scalar QoIs.
        Parameters:
        - stats_dict_s: list of dictionaries with statistics for each QoI
        - qoi_name: name of the quantity of interest (QoI) for labeling
        - foldername: folder to save the plot
        - filename: name of the file to save the plot
        - r_ind: index of the radius to plot (default is 0)
        """

        # Specific to bespoke plot for a list of QoIs
        fig, ax = plt.subplots()

        # Boxplotting the mean and std at a single radius

        # ax.plot(qoi_name, y[r_ind], 'o', label=f"<y> at r={0} and {qoi_name}")

        # ax.errorbar(qoi_name, y[r_ind], yerr=sy[r_ind], fmt='o', label=f"+/- STD at r_ind={r_ind} and {qoi_name}")

        ax.bxp(
            stats_dict_s,
            patch_artist=True,
            showmeans=True,
            shownotches=True,
            # meanline=True,  # Show mean line
            label=f"QoIs at r.ind {r_ind}",
            # label=f"Mean, 95% CI, 10% - 90%, min - max",
        )

        # ax.fill_betweenx([y10[r_ind], y90[r_ind]], qoi - 0.01, qoi + 0.01, alpha=0.1, label=f"10% - 90% at r={0} and {qoi_name}")

        ax.set_ylabel(f"Concentration [m^-3] at radius r index [{r_ind}]")  # Assuming all QoIs have the same units
        ax.set_xlabel(f"Different times of a simulation")
        ax.set_title(f"Uncertainty in QoIs: mean, median, 95% CI, 10% - 90%, min - max")
        ax.legend(loc="best")
        ax.grid(axis="y")

        # fig_qoi.suptitle("Uncertainty in QoIs at selected radius")

        fig.savefig(f"{foldername}/{filename}")

        return 0

    def plot_sobols_first_vs_r(self, r, sobols_first, qoi_name_s, foldername="", filename_base=""):
        """
        Plot first-order Sobol indices as a function of radius.
        Parameters:
        - sobols_first: dictionary of first-order Sobol indices
        - qoi_name_s: list of QoI names to plot
        - foldername: folder to save the plot
        - filename_base: base name of the file to save the plot
        """

        fig, ax = plt.subplots()

        # for qoi_name in qoi_name_s:

        # Get Sobol dictionary for each QoI
        for qoi_name in qoi_name_s:

            sobols_first_qoi = sobols_first.get(qoi_name, None) if sobols_first else None

            if sobols_first_qoi is None:
                print(f"Warning: No first-order Sobol data found for QoI '{qoi_name}'")
                continue

            # Iterate over parameters for first-order Sobol indices
            for param_name, sobol_values in sobols_first_qoi.items():
                if sobol_values is not None:
                    ax.plot(
                        r,
                        sobol_values,
                        label=f"{self.parameters_descriptor.get(param_name, {'name': param_name}).get('name', param_name)} [{self.parameters_descriptor.get(param_name, {'unit': ''}).get('unit', '')}]",
                    )
                    # fig.savefig(f"{foldername}/{filename_base}_{qoi_name}_{param_name}.pdf")

            ax.set_xlabel(f"Radius [m]")
            ax.set_ylabel(f"Sobol index [fraction of unity]")

            ax.set_title(
                f"First-order Sobol indices vs Radius for {self.quantities_descriptor.get(self.quantity, {'name': self.quantity}).get('name', self.quantity)} at {qoi_name}"
            )
            ax.legend(loc="best")

            ax.grid()

            fig.savefig(f"{foldername}/bespoke_{filename_base}_{qoi_name}_first_order_sobols.pdf")
            plt.close()

        return 0

    def _parse_time_from_qoi(self, qoi_name):
        """Extract time value from QoI labels like 't=1.00e-01s'."""
        if not isinstance(qoi_name, str):
            return None
        m = re.match(r"^t=([0-9eE+\-.]+)s$", qoi_name)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def _group_qois_for_sobol_heatmap(self, qoi_name_s):
        """Group QoIs into columns for heatmap blocks.

        If all QoIs are transient labels ('t=...s'), they are treated as one
        output quantity (single column). Otherwise each QoI is a separate column.
        """
        qoi_name_s = list(qoi_name_s or [])
        if qoi_name_s and all(self._parse_time_from_qoi(q) is not None for q in qoi_name_s):
            return {"tritium_concentration": qoi_name_s}
        return {str(q): [q] for q in qoi_name_s}

    def plot_sobols_first_heatmap_blocks(self, results, r, qoi_name_s, foldername="", filename_base="sobols_first_heatmap"):
        """Plot Sobol index colormaps on (radius,time) blocks.

        Rows correspond to uncertain parameters, columns correspond to uncertain
        output QoI groups.
        """
        groups = self._group_qois_for_sobol_heatmap(qoi_name_s)
        if not groups:
            return 0

        # Collect uncertain parameter names from available Sobol outputs.
        param_names = []
        for q in qoi_name_s:
            try:
                s1 = results.sobols_first(q)
            except Exception:
                s1 = None
            if s1:
                for p in s1.keys():
                    if p not in param_names:
                        param_names.append(p)

        if not param_names:
            print("No first-order Sobol data available for heatmap blocks.")
            return 0

        r = np.asarray(r, dtype=float)
        if r.ndim != 1 or r.size == 0:
            raise ValueError("Radius array 'r' must be a non-empty 1D array for Sobol heatmaps.")

        n_rows = len(param_names)
        n_cols = len(groups)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 3.6 * n_rows), squeeze=False)

        for col_idx, (qoi_group_name, qoi_group_members) in enumerate(groups.items()):
            # Build a sorted (time, qoi, sobol_dict) sequence for this group.
            entries = []
            for q in qoi_group_members:
                t_val = self._parse_time_from_qoi(q)
                if t_val is None:
                    continue
                try:
                    s1 = results.sobols_first(q)
                except Exception:
                    s1 = None
                if s1:
                    entries.append((t_val, q, s1))

            entries.sort(key=lambda item: item[0])

            if not entries:
                for row_idx, p_name in enumerate(param_names):
                    ax = axes[row_idx, col_idx]
                    ax.text(0.5, 0.5, "No transient Sobol data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_axis_off()
                continue

            t_vals = np.asarray([item[0] for item in entries], dtype=float)

            for row_idx, p_name in enumerate(param_names):
                ax = axes[row_idx, col_idx]
                z = np.full((len(t_vals), len(r)), np.nan, dtype=float)

                for i_t, (_, _, sobol_dict) in enumerate(entries):
                    sobol_values = sobol_dict.get(p_name, None)
                    if sobol_values is None:
                        continue
                    sobol_values = np.asarray(sobol_values, dtype=float).reshape(-1)
                    if sobol_values.size == len(r):
                        z[i_t, :] = sobol_values
                    elif sobol_values.size > 1:
                        z[i_t, :] = np.interp(
                            np.linspace(0.0, 1.0, len(r)),
                            np.linspace(0.0, 1.0, sobol_values.size),
                            sobol_values,
                        )

                z_plot = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
                pcm = ax.pcolormesh(r, t_vals, z_plot, shading="auto", vmin=0.0, vmax=1.0, cmap="viridis")
                cbar = fig.colorbar(pcm, ax=ax)
                cbar.set_label("Sobol index")

                if row_idx == 0:
                    ax.set_title(f"QoI: {qoi_group_name}")
                if row_idx == n_rows - 1:
                    ax.set_xlabel("Radius [m]")

                param_label = self.parameters_descriptor.get(p_name, {}).get("name", p_name)
                ax.set_ylabel(f"Time [s]\n{param_label}")
                ax.grid(False)

        fig.suptitle("First-order Sobol Indices on (radius, time) blocks", fontsize=14)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])

        if filename_base.endswith(".pdf") or filename_base.endswith(".png"):
            out_name = filename_base
        else:
            out_name = f"{filename_base}.pdf"
        fig.savefig(f"{foldername}/{out_name}")
        plt.close(fig)
        return 0

    def plot_sobols_seconds_vs_r(self, r, sobols_second, qoi_name_s, foldername="", filename_base=""):
        """
        Plot second-order Sobol indices as a function of radius.
        Parameters:
        - sobols_second: dictionary of second-order Sobol indices (?)
        - foldername: folder to save the plot
        - filename: name of the file to save the plot
        """

        fig, ax = plt.subplots()

        # for param_name_1, param_name_2 in itertools.product(param_name_s, repeat=2):

        # Get Sobol dictionary for each QoI
        for qoi_name in qoi_name_s:

            # sobols_second_qoi = sobols_second.get(qoi_name_1, None).get(qoi_name_2, None) if sobols_second and qoi_name_1 != qoi_name_2 else None # TODO: get rid of the repetition of pairs

            sobol_data = sobols_second.get(qoi_name, None)
            if sobol_data is None:
                print(f"Warning: No second-order Sobol data found for QoI '{qoi_name}'")
                continue

            # Iterate over pairs of parameters for second-order Sobol indices
            for param_name_1 in sobol_data:
                for param_name_2 in sobol_data[param_name_1]:
                    # Avoid plotting the same pair twice or diagonal elements
                    if param_name_1 > param_name_2:

                        sobols_second_qoi = sobol_data[param_name_1][param_name_2]
                        if sobols_second_qoi is not None:

                            ax.plot(
                                r,
                                sobols_second_qoi,
                                label=f"{param_name_1} & {param_name_2}",
                            )
                            # fig.savefig(f"{foldername}/{filename_base}_{qoi_name_1}_{qoi_name_2}.pdf")

            ax.set_xlabel(f"Radius [m]")
            ax.set_ylabel(f"Sobol index [fraction of unity]")

            ax.set_title(f"Second-order Sobol indices vs Radius at {qoi_name}")
            ax.legend(loc="best")

            ax.grid()

            fig.savefig(f"{foldername}/{filename_base}_{qoi_name}_second_order_sobols.pdf")
            plt.close()

        return 0

    def plot_stats_vs_r(
        self,
        results,
        qois: list[str],
        plot_folder_name: str,
        plot_timestamp: str,
        rs=None,
        runs_info=None,
        show_distribution=False,
        dist_x_index=0,
        dist_mode="point",
        pce_surrogate=None,
        joint_dist=None,
    ):
        """
        Plot statistics of the results as a function of radius (spatial coordinates).
        Parameters:
        - results: analysis results object from EasyVVUQ
        - qois: list of quantities of interest (QoIs) names to plot, for labeling
        - plot_folder_name: folder to save the plots
        - plot_timestamp: timestamp to append to the filenames
        - rs: array of radius values (optional, if not provided, will be generated)
        - runs_info: information about individual runs, if available (optional); should be a list of dictionaries, with 'id' and 'results'
        - show_distribution: if True, also generate a marginal distribution plot per QoI
        - dist_x_index: spatial index used when reducing profiles to scalars for distribution plots
        - dist_mode: reduction mode for scalar extraction ('point', 'mean', 'trapz')
        - pce_surrogate: fitted chaospy PCE polynomial (used for PCE marginal PDF)
        - joint_dist: chaospy joint distribution of uncertain inputs (used with pce_surrogate)
        """

        # Specific for common boxplot for QoIs
        stats_dict_s = []
        r_ind_qoi = 0  # Select the max radius values: r=0.0 should be physical centre of domain

        # Rund over QoIs in analysis results object
        for qoi in qois:
            # Generate filenames with timestamp
            file_type = "pdf"  # Assuming we want to save as PDF
            moments_vsr_filename = add_timestamp_to_filename(f"{qoi}_moments_vs_r.{file_type}", plot_timestamp)
            sobols_treemap_filename = add_timestamp_to_filename(f"{qoi}_sobols_treemap.{file_type}", plot_timestamp)
            sobols_filename = add_timestamp_to_filename(f"{qoi}_sobols_first_vs_r.{file_type}", plot_timestamp)

            # Read out the arrays of stats from the results object
            y = results.describe(qoi, "mean")
            ymed = results.describe(qoi, "median")
            sy = results.describe(qoi, "std")
            y01 = results.describe(qoi, "1%")
            y10 = results.describe(qoi, "10%")
            y90 = results.describe(qoi, "90%")
            y99 = results.describe(qoi, "99%")
            # ymin = results.describe(qoi, 'min')
            # ymax = results.describe(qoi, 'max')

            # print(f" >>> Finished reading statistics for QoI with EasyVVUQ: {qoi}")

            # Filling in the values for the list of dicts for a common boxplot
            stats_dict_s.append(
                {
                    "mean": [y[r_ind_qoi]],
                    "med": [ymed[r_ind_qoi]],
                    "q1": [y10[r_ind_qoi]],
                    "q3": [y90[r_ind_qoi]],
                    "cilo": [y[r_ind_qoi] - 1.95 * sy[r_ind_qoi]],
                    "cihi": [y[r_ind_qoi] + 1.95 * sy[r_ind_qoi]],
                    "whislo": [y01[r_ind_qoi]],
                    "whishi": [y99[r_ind_qoi]],
                    "fliers": [],  # [ymin[r_ind_qoi], ymax[r_ind_qoi]],
                    "label": f"{qoi}",
                }
            )

            # Define a simple range for x-axis
            # rs = np.linspace(0., 1., len(y))  # Should be done outside of scope of current function

            # Read out individual trajectories of single runs from .raw_data
            logger.debug(
                f" >>> Reading individual trajectories for QoI with EasyVVUQ: results.raw_data = \n{results.raw_data}"
            )
            # TODO might be needed to read from campaign.db

            # Default plotting of the moments
            # print(f" >> Plotting moments for QoI with EasyVVUQ: {qoi}")
            qty_name = self.quantities_descriptor.get(qoi, {'name': qoi}).get('name', qoi)
            qty_unit = self.quantities_descriptor.get(qoi, {'unit': ''}).get('unit', '')
            results.plot_moments(
                qoi=qoi,
                ylabel=f"{self._format_label_with_unit(qty_name, qty_unit)}, {qoi}",
                xlabel="Radius, #vertices",
                filename=f"{plot_folder_name}/{moments_vsr_filename}",
            )
            print(f" >>> Finished plotting moments for QoI with EasyVVUQ: {qoi}")

            # Plotting Sobol indices as a treemap
            # TODO: figure out how to plot treemaps at arbitrary locations
            # results.plot_sobols_treemap(
            #     qoi=qoi,
            #     filename=f"{plot_folder_name}/{sobols_treemap_filename}",
            # )

            # Bespoke plotting of uncertainty in QoI (vs. radius)
            # print(f" >> Plotting moments for QoI via bespoke function: {qoi}")
            self.plot_unc_vs_r(
                rs,
                y,
                sy,
                y10,
                y90,
                y01,
                y99,
                qoi_name=qoi,
                foldername=plot_folder_name,
                filename=moments_vsr_filename,
                runs_info=runs_info,
            )

            # Plotting Sobol indices as a function of radius
            # print(f" >> Plotting first Sobol indices for QoI via EasyVVUQ: {qoi}")
            results.plot_sobols_first(
                qoi=qoi,
                withdots=False,  # Show dots for each Sobol index
                xlabel=f"Radius, #vertices",
                ylabel=f"Sobol Index (first) for {self.quantities_descriptor.get(self.quantity, {'name': self.quantity}).get('name', self.quantity)} at {qoi}",
                filename=f"{plot_folder_name}/{sobols_filename}",  # Save with bespoke prefix
            )

            # Bespoke plotting of first-order Sobol indices vs radius
            self.plot_sobols_first_vs_r(
                rs, results.sobols_first(), qois, foldername=plot_folder_name, filename_base="sobols_first_vs_r"
            )

            print(
                f"Plots (for spatially resolved functions) saved: {moments_vsr_filename}, {sobols_treemap_filename}, {sobols_filename}"
            )

            # ---- optional: marginal distribution plot for this QoI ----
            if show_distribution:
                try:
                    from .utils import extract_scalar_qoi

                    qoi_scalars = extract_scalar_qoi(
                        results, qoi, mode=dist_mode, x_values=rs, x_index=dist_x_index
                    )
                    dist_filename = add_timestamp_to_filename(f"{qoi}_distribution.pdf", plot_timestamp)
                    self.plot_qoi_distribution(
                        qoi_scalars,
                        qoi_name=qoi,
                        foldername=plot_folder_name,
                        filename=dist_filename,
                        pce_surrogate=pce_surrogate,
                        joint_dist=joint_dist,
                        timestamp=plot_timestamp,
                    )
                    print(f" >>> Distribution plot saved for QoI: {qoi}")
                except Exception as exc:
                    logger.warning(f"Distribution plot failed for '{qoi}': {exc}")

        # Save plot common for QoIs: specific for bespoke QoI uncertainty plotting
        #  - bespoke plotting of uncertainty in QoI (at selected radius)
        # print(f" >> Plotting uncertainties for QoI via bespoke functionality: {qoi}")
        file_type = "pdf"  # Assuming we want to save as PDF
        self.plot_unc_qoi(
            stats_dict_s,
            qoi_name=qoi,
            foldername=plot_folder_name,
            filename=add_timestamp_to_filename(f"qoi_uncertainty_vs_r.{file_type}", plot_timestamp),
            r_ind=r_ind_qoi,
        )

        # TODO add total Sobol indices as well

        # Read second-order Sobol indices from the UQ results object
        sobols_second = results.sobols_second()
        logger.debug(f" >> Second-order Sobol indices for QoI with EasyVVUQ: {qoi} : \n {sobols_second}")
        self.plot_sobols_seconds_vs_r(
            rs, sobols_second, qois, foldername=plot_folder_name, filename_base="sobols_second_vs_r"
        )

        # Sobol colormap blocks in (radius,time): rows=uncertain parameters, columns=QoI groups.
        self.plot_sobols_first_heatmap_blocks(
            results,
            rs,
            qois,
            foldername=plot_folder_name,
            filename_base=add_timestamp_to_filename("sobols_first_blocks_r_vs_t.pdf", plot_timestamp),
        )

        return 0

    def plot_unc_vs_t(self, r_at_r, t_s, y_at_r, sy_at_r, y10_at_r, y90_at_r, y01_at_r=None, y99_at_r=None, foldername="", filename=""):
        """
        Plot uncertainty in the results as a function of time.
        """
        fig, ax = plt.subplots()
        # print(f"Shapes of the lists: y_s: {len(y_s)}, sy_s: {len(sy_s)}, y10_s: {len(y10_s)}, y90_s: {len(y90_s)}")

        ax.plot(t_s, y_at_r, label=f"<y> at r={r_at_r}")
        ax.fill_between(
            t_s, np.array(y_at_r) - np.array(sy_at_r), np.array(y_at_r) + np.array(sy_at_r), alpha=0.3, label="+/- STD"
        )
        if y01_at_r is not None and y99_at_r is not None:
            y01_arr = np.asarray(y01_at_r)
            y99_arr = np.asarray(y99_at_r)
            if y01_arr.shape == y99_arr.shape and np.all(np.isfinite(y01_arr)) and np.all(np.isfinite(y99_arr)):
                ax.fill_between(t_s, y01_arr, y99_arr, alpha=0.12, label="1%-99%")
        ax.fill_between(t_s, y10_at_r, y90_at_r, alpha=0.1, label="10% - 90%")

        ax.set_title(f"Uncertainty as a function of time at r={r_at_r}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Concentration [m^-3] at {r_at_r}")
        ax.legend(loc="best")
        ax.grid(True)

        fig.savefig(f"{foldername}/{filename}")

        plt.close()

        return 0

    def plot_sobols_vs_t(self, r_s, t_s, s1_s, distributions, foldername="", filename="", r_ind=0):
        """
        Plot Sobol indices as a function of time.
        """
        fig, ax = plt.subplots()
        # print(s1_s[-1])

        for i, param_name in enumerate(distributions.keys()):
            # Extract r_ind-th element from each Sobol array to get time series at fixed radius
            s1_at_r = [s1_timestep[r_ind] for s1_timestep in s1_s[i]]
            ax.plot(t_s, s1_at_r, label=f"Sobol Index (first) for {param_name}")

        ax.set_title(f"Sobol Indices as a function of time at r={r_s}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Sobol Index (first), fraction of unity")
        ax.legend()

        fig.savefig(f"{foldername}/{filename}")

        plt.close()

        return 0

    def plot_stats_vs_t(self, results, distributions, qois, plot_folder_name, plot_timestamp, rs=None):
        """
        Plot statistics of the results as a function of time.
        """

        # Select - uncertainty at the depth of the specimen (r=0.)
        r_ind_selected = [0, -1]  # Select the first and last radius index (or any other index)

        # Read the results for all times and align data for plotting against time
        y_s = []
        sy_s = []
        y01_s = []
        y99_s = []
        y10_s = []
        y90_s = []
        r_s = []

        s1_s = [[] for _ in range(len(distributions))]  # Assuming one first Sobol index per distribution

        # op1) Extract time from QoI names
        # t_s = [float(qoi.split('=')[1].strip()) for qoi in qois]
        # op2) read from results
        t_s = []

        # Run over QoIs in analysis results object and read statistics
        for qoi in qois:
            # Every element in qois list is a single time step
            # Read every time step from results in a list-of-lists [n_timesteps x n_elements]
            t_s.append(
                float(qoi.split("=")[1].strip()[:-1])
            )  # Extract time from QoI names, strp 's' at the end and '='
            y_s.append(results.describe(qoi, "mean"))
            sy_s.append(results.describe(qoi, "std"))
            try:
                y01_s.append(results.describe(qoi, "1%"))
            except Exception:
                y01_s.append(np.full_like(np.asarray(y_s[-1]), np.nan, dtype=float))
            y10_s.append(results.describe(qoi, "10%"))
            y90_s.append(results.describe(qoi, "90%"))
            try:
                y99_s.append(results.describe(qoi, "99%"))
            except Exception:
                y99_s.append(np.full_like(np.asarray(y_s[-1]), np.nan, dtype=float))
            s1 = results.sobols_first(qoi)  # returing a dict {input_param: (list of) Sobol index values}
            for i, param_name in enumerate(distributions.keys()):
                # Assuming each distribution is a valid QoI descriptor
                s1_s[i].append(s1[param_name])  # Assuming 'first' is a valid QoI descriptor
            # r_s.append(np.linspace(0., 1., len(y_s[-1])))
            r_s.append(rs)  # assuming we read the readius values from outside, and they are the same for all QoIs

        # Run over selected radius indices
        for r_ind in r_ind_selected:
            # Generate filenames with timestamp for time series plots
            file_type = "pdf"  # Assuming we want to save as PDF
            moments_vst_filename = add_timestamp_to_filename(f"moments_vs_t_at_{r_ind}.{file_type}", plot_timestamp)
            sobols_vst_filename = add_timestamp_to_filename(f"sobols_first_vs_t_at_{r_ind}.{file_type}", plot_timestamp)

            # Extract r_ind-th element from each time step (array) to get time series at fixed radius
            y_at_r = [y_timestep[r_ind] for y_timestep in y_s]
            sy_at_r = [sy_timestep[r_ind] for sy_timestep in sy_s]
            y01_at_r = [y01_timestep[r_ind] for y01_timestep in y01_s]
            y10_at_r = [y10_timestep[r_ind] for y10_timestep in y10_s]
            y90_at_r = [y90_timestep[r_ind] for y90_timestep in y90_s]
            y99_at_r = [y99_timestep[r_ind] for y99_timestep in y99_s]
            r_at_r = [r_timestep[r_ind] for r_timestep in r_s]  # Assuming r_s is a list of lists with radius values
            # TODO check if r_at_r changes with time, or is constant

            # Plotting of moments as a function of time
            self.plot_unc_vs_t(
                r_at_r[0],
                t_s,
                y_at_r,
                sy_at_r,
                y10_at_r,
                y90_at_r,
                y01_at_r,
                y99_at_r,
                foldername=plot_folder_name,
                filename=moments_vst_filename,
            )

            # Plotting Sobol indices as a function of time
            self.plot_sobols_vs_t(
                r_at_r[0],
                t_s,
                s1_s,
                distributions,
                foldername=plot_folder_name,
                filename=sobols_vst_filename,
                r_ind=r_ind,
            )

            print(f"Plots (for time series) saved: {moments_vst_filename}, {sobols_vst_filename}")

        return 0

    def plot_derivatives_vs_r(self, r, derivatives_first, qoi_name_s, param_names, foldername="", filename_base=""):
        """
        Plot first-order derivative-based sensitivity indices as a function of radius.
        This is the primary sensitivity measure from FDAnalysis (finite-difference approach).

        Parameters:
        - r: array of radius / 1D coordinate values
        - derivatives_first: dictionary of derivative-based indices from FDAnalysis,
          structured as {qoi: {param: array_of_values}}
        - qoi_name_s: list of QoI names to plot
        - param_names: list of uncertain parameter names
        - foldername: folder to save the plot
        - filename_base: base name of the file to save the plot
        """

        for qoi_name in qoi_name_s:

            deriv_qoi = derivatives_first.get(qoi_name, None) if derivatives_first else None

            if deriv_qoi is None:
                print(f"Warning: No derivative data found for QoI '{qoi_name}'")
                continue

            fig, ax = plt.subplots(figsize=(8, 6))

            for param_name in param_names:
                deriv_values = deriv_qoi.get(param_name, None)
                if deriv_values is not None:
                    label = self.parameters_descriptor.get(param_name, {}).get("name", param_name)
                    unit = self.parameters_descriptor.get(param_name, {}).get("unit", "")
                    ax.plot(r, deriv_values, label=f"{label} [{unit}]")

            ax.set_xlabel("Radius [m]")
            ax.set_ylabel("Derivative-based sensitivity index")
            qty_name = self.quantities_descriptor.get(self.quantity, {}).get("name", self.quantity)
            ax.set_title(f"Derivative-based sensitivity vs Radius\n{qty_name} at {qoi_name}")
            ax.legend(loc="best")
            ax.grid(True)

            fig.savefig(f"{foldername}/{filename_base}_{qoi_name}_derivatives.pdf")
            plt.close()

        return 0

    def plot_derivatives_vs_t(self, r_value, t_s, deriv_s, param_names, foldername="", filename="", r_ind=0):
        """
        Plot derivative-based sensitivity indices as a function of time at a fixed radius.

        Parameters:
        - r_value: radius value for labeling
        - t_s: list of time values
        - deriv_s: list of lists, shape [n_params][n_timesteps], each entry is an array over r
        - param_names: list of uncertain parameter names
        - foldername: folder to save the plot
        - filename: name of the file to save the plot
        - r_ind: index of the radius to extract
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, param_name in enumerate(param_names):
            deriv_at_r = [d_timestep[r_ind] for d_timestep in deriv_s[i]]
            label = self.parameters_descriptor.get(param_name, {}).get("name", param_name)
            ax.plot(t_s, deriv_at_r, label=f"{label}")

        ax.set_title(f"Derivative-based sensitivity vs time at r={r_value}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Derivative-based sensitivity index")
        ax.legend(loc="best")
        ax.grid(True)

        fig.savefig(f"{foldername}/{filename}")
        plt.close()

        return 0

    def plot_unc_correlated_vs_r(self, r, y, sy, qoi_name, foldername="", filename=""):
        """
        Plot uncertainty for correlated FD analysis results as a function of radius.
        Uses mean +/- k*std Gaussian confidence intervals instead of quantiles,
        since FDAnalysis does not compute sample-based percentiles.

        Parameters:
        - r: array of radius / 1D coordinate values
        - y: array of mean values
        - sy: array of standard deviation values
        - qoi_name: name of the quantity of interest (QoI) for labeling
        - foldername: folder to save the plot
        - filename: name of the file to save the plot
        """

        plot_types = ["plot", "semilogy"]
        n_plots = len(plot_types)

        fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 8, 6))

        for i, plot_func_name in enumerate(plot_types):

            plot_func = getattr(axs[i], plot_func_name, None)
            if plot_func is None:
                raise ValueError(f"Plot function '{plot_func_name}' is not supported.")

            # Plot mean
            plot_func(r, y, label=f"Mean at {qoi_name}")

            # Plot +/- 1 std as shaded area (approx. 68% CI for Gaussian)
            axs[i].fill_between(r, y - sy, y + sy, alpha=0.3, label=r"$\pm 1\sigma$ (68% CI)")

            # Plot +/- 2 std as lighter shaded area (approx. 95% CI for Gaussian)
            axs[i].fill_between(r, y - 2 * sy, y + 2 * sy, alpha=0.1, label=r"$\pm 2\sigma$ (95% CI)")

            qty_name = self.quantities_descriptor.get(self.quantity, {}).get("name", self.quantity)
            qty_unit = self.quantities_descriptor.get(self.quantity, {}).get("unit", "")
            axs[i].set_title(f"Uncertainty (correlated FD) at {qoi_name}\n'{plot_func_name}' scale")
            axs[i].set_xlabel("Radius [m]")
            ylabel = self._format_label_with_unit(qty_name, qty_unit)
            axs[i].set_ylabel(ylabel)
            axs[i].legend(loc="best")
            axs[i].grid(True)

        fig.tight_layout()
        fig.savefig(f"{foldername}/correlated_{filename}")
        plt.close()

        return 0

    def plot_unc_correlated_vs_t(self, r_value, t_s, y_at_r, sy_at_r, foldername="", filename=""):
        """
        Plot uncertainty for correlated FD analysis results as a function of time at fixed radius.
        Uses mean +/- k*std Gaussian confidence intervals.

        Parameters:
        - r_value: radius value for labeling
        - t_s: list of time values
        - y_at_r: list of mean values at fixed radius
        - sy_at_r: list of std values at fixed radius
        - foldername: folder to save the plot
        - filename: name of the file to save the plot
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        y_arr = np.array(y_at_r)
        sy_arr = np.array(sy_at_r)

        ax.plot(t_s, y_arr, label=f"Mean at r={r_value}")
        ax.fill_between(t_s, y_arr - sy_arr, y_arr + sy_arr, alpha=0.3, label=r"$\pm 1\sigma$ (68% CI)")
        ax.fill_between(t_s, y_arr - 2 * sy_arr, y_arr + 2 * sy_arr, alpha=0.1, label=r"$\pm 2\sigma$ (95% CI)")

        ax.set_title(f"Uncertainty (correlated FD) vs time at r={r_value}")
        ax.set_xlabel("Time [s]")
        qty_name = self.quantities_descriptor.get(self.quantity, {}).get("name", self.quantity)
        qty_unit = self.quantities_descriptor.get(self.quantity, {}).get("unit", "")
        ylabel = self._format_label_with_unit(qty_name, qty_unit)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best")
        ax.grid(True)

        fig.savefig(f"{foldername}/{filename}")
        plt.close()

        return 0

    def plot_stats_correlated(
        self,
        results,
        distributions,
        qois,
        plot_folder_name,
        plot_timestamp,
        rs=None,
        show_distribution=False,
        dist_x_index=0,
        dist_mode="point",
    ):
        """
        Plot statistics from a correlated FD (finite-difference) UQ analysis.

        This is a bespoke plotting function for FDAnalysis results, which differ from
        PCEAnalysis results in that:
        - Percentiles (10%, 90%, etc.) are NOT computed by FDAnalysis — they are
          left as zero arrays and should not be used for plotting.
        - Sobol indices are NOT computed — they are zero arrays.
        - The primary sensitivity measure is `derivatives_first`.
        - Mean, variance, and std are properly computed from the simulation outputs.

        This function plots:
        1. Uncertainty bands (mean +/- k*std) vs radius for each QoI (spatial)
        2. Uncertainty bands vs time at selected radii (temporal)
        3. Derivative-based sensitivity indices vs radius (spatial sensitivity)
        4. Derivative-based sensitivity indices vs time (temporal sensitivity)

        Parameters:
        - results: FDAnalysis results object from EasyVVUQ
        - distributions: dictionary of uncertain parameter distributions
        - qois: list of QoI names (time-step labels), excluding 'x'
        - plot_folder_name: folder to save the plots
        - plot_timestamp: timestamp to append to filenames
        - rs: array of radius values (optional)
        - show_distribution: if True, also generate a marginal distribution plot per QoI
        - dist_x_index: spatial index used when reducing profiles to scalars for distribution plots
        - dist_mode: reduction mode for scalar extraction ('point', 'mean', 'trapz')

        Note:
            For future improvements, consider:
            - Using Monte Carlo sampling from the joint distribution (cp.MvNormal)
              to obtain proper quantiles via EasyVVUQ's QMCSampler.
            - Implementing Latin Hypercube Sampling (LHS) with correlation via
              Iman-Conover method for non-Gaussian marginals.
            - Using Nataf transform for correlated non-Gaussian inputs.
            - Applying Polynomial Chaos Expansion with Rosenblatt/Nataf transforms
              to handle correlated inputs while preserving spectral convergence.
            - Visualising correlation contribution using decomposed variance
              (independent vs correlated contributions).
        """

        param_names = list(distributions.keys())
        file_type = "pdf"

        # ---- Part 1: Spatial plots (vs radius) for each QoI ----

        for qoi in qois:
            # Read properly computed statistics
            y = results.describe(qoi, "mean")
            sy = results.describe(qoi, "std")

            # Generate filename
            moments_filename = add_timestamp_to_filename(f"{qoi}_moments_vs_r.{file_type}", plot_timestamp)

            # Bespoke uncertainty plot (mean +/- std bands)
            self.plot_unc_correlated_vs_r(
                rs, y, sy, qoi_name=qoi, foldername=plot_folder_name, filename=moments_filename
            )

            # EasyVVUQ built-in moments plot (uses mean and std only, safe for FDAnalysis)
            results.plot_moments(
                qoi=qoi,
                ylabel=f"{self.quantities_descriptor.get(self.quantity, {}).get('name', self.quantity)} at {qoi}",
                xlabel="Radius, #vertices",
                xvalues=rs,
                filename=f"{plot_folder_name}/{moments_filename}",
            )

            print(f" >>> Spatial plots saved for QoI: {qoi}")

            # ---- optional: marginal distribution plot for this QoI ----
            if show_distribution:
                try:
                    from .utils import extract_scalar_qoi

                    qoi_scalars = extract_scalar_qoi(
                        results, qoi, mode=dist_mode, x_values=rs, x_index=dist_x_index
                    )
                    dist_filename = add_timestamp_to_filename(f"{qoi}_distribution.pdf", plot_timestamp)
                    self.plot_qoi_distribution(
                        qoi_scalars,
                        qoi_name=qoi,
                        foldername=plot_folder_name,
                        filename=dist_filename,
                        timestamp=plot_timestamp,
                    )
                    print(f" >>> Distribution plot saved for QoI: {qoi}")
                except Exception as exc:
                    logger.warning(f"Distribution plot failed for '{qoi}': {exc}")

        # ---- Part 2: Derivative-based sensitivity (vs radius) ----

        derivatives_first = results.derivatives_first()
        self.plot_derivatives_vs_r(
            rs,
            derivatives_first,
            qois,
            param_names,
            foldername=plot_folder_name,
            filename_base=add_timestamp_to_filename("bespoke_sensitivity", plot_timestamp),
        )

        # ---- Part 3: Temporal plots (vs time) at selected radii ----

        r_ind_selected = [0, -1]

        # Collect time-resolved statistics
        y_s = []
        sy_s = []
        r_s = []
        t_s = []
        deriv_s = [[] for _ in range(len(param_names))]

        for qoi in qois:
            t_s.append(float(qoi.split("=")[1].strip()[:-1]))
            y_s.append(results.describe(qoi, "mean"))
            sy_s.append(results.describe(qoi, "std"))
            r_s.append(rs)

            # Read derivative-based sensitivity for each parameter at this time step
            for i, param_name in enumerate(param_names):
                deriv_s[i].append(results.derivatives_first(qoi, param_name))

        for r_ind in r_ind_selected:
            moments_vst_filename = add_timestamp_to_filename(
                f"corr_moments_vs_t_at_{r_ind}.{file_type}", plot_timestamp
            )
            derivs_vst_filename = add_timestamp_to_filename(f"corr_derivs_vs_t_at_{r_ind}.{file_type}", plot_timestamp)

            y_at_r = [y_timestep[r_ind] for y_timestep in y_s]
            sy_at_r = [sy_timestep[r_ind] for sy_timestep in sy_s]
            r_at_r = [r_timestep[r_ind] for r_timestep in r_s]

            # Uncertainty vs time
            self.plot_unc_correlated_vs_t(
                r_at_r[0], t_s, y_at_r, sy_at_r, foldername=plot_folder_name, filename=moments_vst_filename
            )

            # Derivative-based sensitivity vs time
            self.plot_derivatives_vs_t(
                r_at_r[0],
                t_s,
                deriv_s,
                param_names,
                foldername=plot_folder_name,
                filename=derivs_vst_filename,
                r_ind=r_ind,
            )

            print(f" >>> Temporal plots saved at r_ind={r_ind}")

        return 0

    # ------------------------------------------------------------------
    # QoI distribution plots
    # ------------------------------------------------------------------

    def plot_qoi_distribution(
        self,
        qoi_values,
        qoi_name,
        foldername="",
        filename=None,
        prior_dist=None,
        pce_surrogate=None,
        joint_dist=None,
        show_histogram=True,
        show_kde=True,
        show_pce_pdf=True,
        n_mc_samples=10_000,
        timestamp=None,
    ):
        """
        Plot the marginal distribution of a scalar QoI.

        Three optional layers can be overlaid on the same axis:

        * **Histogram** — normalised to a probability density, built from the
          raw quadrature/sample evaluations in *qoi_values*.
        * **KDE** — a kernel-density estimate computed with
          ``scipy.stats.gaussian_kde`` (requires *scipy*; silently skipped if
          not installed).
        * **PCE marginal PDF** — *n_mc_samples* points are drawn from
          *joint_dist* (a ``chaospy`` joint distribution), the fitted
          *pce_surrogate* polynomial is evaluated at each, and the resulting
          empirical density is plotted as a smooth curve.  Requires both
          *pce_surrogate* and *joint_dist*.

        Additionally, vertical reference lines mark the mean, ±1 σ, and the
        10th / 90th percentiles.  When *prior_dist* is provided (a
        ``chaospy`` or ``scipy.stats`` frozen distribution), its PDF is drawn
        on a twin axis as a dashed line.

        Parameters
        ----------
        qoi_values : array-like
            1-D array of scalar QoI evaluations (one per model run /
            quadrature node).
        qoi_name : str
            Label used in axis titles and the output filename.
        foldername : str, optional
            Directory where the plot file is saved.  Current directory if
            empty.
        filename : str, optional
            Output filename (including extension).  Defaults to
            ``"{qoi_name}_distribution_{timestamp}.pdf"``.
        prior_dist : chaospy.Dist or scipy.stats rv_frozen, optional
            Prior distribution of the *output* QoI to overlay as a reference
            PDF.  Typically the push-forward of the joint input prior through
            the PCE surrogate.
        pce_surrogate : chaospy polynomial, optional
            Fitted PCE polynomial evaluated over *joint_dist* to produce a
            high-fidelity empirical output distribution.  Must be paired with
            *joint_dist*.
        joint_dist : chaospy.Dist, optional
            Joint distribution of the uncertain *input* parameters used to
            draw Monte-Carlo samples for the PCE marginal PDF.
        show_histogram : bool, optional
            Whether to plot the raw-sample histogram.  Default ``True``.
        show_kde : bool, optional
            Whether to overlay a KDE curve.  Default ``True``.
        show_pce_pdf : bool, optional
            Whether to draw the PCE-derived marginal PDF.  Default ``True``.
        n_mc_samples : int, optional
            Number of Monte-Carlo samples drawn from *joint_dist* for the PCE
            marginal PDF.  Default 10 000.
        timestamp : str, optional
            Timestamp appended to the auto-generated filename.

        Returns
        -------
        str
            Path to the saved figure.
        """
        qoi_values = np.asarray(qoi_values, dtype=float)
        qoi_values = qoi_values[np.isfinite(qoi_values)]

        if len(qoi_values) == 0:
            logger.warning(f"plot_qoi_distribution: no finite values for QoI '{qoi_name}', skipping.")
            return ""

        # ---- statistics ------------------------------------------------
        mean_val = float(np.mean(qoi_values))
        std_val = float(np.std(qoi_values))
        p10 = float(np.percentile(qoi_values, 10))
        p90 = float(np.percentile(qoi_values, 90))

        # ---- figure setup -----------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))

        x_min = min(qoi_values.min(), mean_val - 3.5 * std_val) if std_val > 0 else qoi_values.min() - 1
        x_max = max(qoi_values.max(), mean_val + 3.5 * std_val) if std_val > 0 else qoi_values.max() + 1
        x_grid = np.linspace(x_min, x_max, 500)

        # ---- PCE marginal PDF (highest fidelity) -------------------------
        if show_pce_pdf and pce_surrogate is not None and joint_dist is not None:
            try:
                mc_inputs = joint_dist.sample(n_mc_samples, rule="random")
                mc_outputs = np.asarray(pce_surrogate(*mc_inputs), dtype=float)
                mc_outputs = mc_outputs[np.isfinite(mc_outputs)]
                if len(mc_outputs) > 1:
                    try:
                        from scipy.stats import gaussian_kde as _kde

                        pce_kde = _kde(mc_outputs)
                        ax.plot(
                            x_grid,
                            pce_kde(x_grid),
                            color="steelblue",
                            lw=2,
                            label=f"PCE marginal PDF (N={n_mc_samples})",
                        )
                    except ImportError:
                        # Fall back to normalised histogram for PCE samples
                        ax.hist(
                            mc_outputs,
                            bins=60,
                            density=True,
                            alpha=0.25,
                            color="steelblue",
                            label=f"PCE marginal (N={n_mc_samples})",
                        )
            except Exception as exc:
                logger.warning(f"PCE marginal PDF could not be computed for '{qoi_name}': {exc}")

        # ---- raw-sample histogram ----------------------------------------
        if show_histogram:
            n_bins = max(5, min(int(np.sqrt(len(qoi_values))), 30))
            ax.hist(
                qoi_values,
                bins=n_bins,
                density=True,
                alpha=0.35,
                color="orange",
                edgecolor="darkorange",
                label=f"Samples histogram (N={len(qoi_values)})",
            )

        # ---- KDE of raw samples ------------------------------------------
        if show_kde and len(qoi_values) >= 2:
            try:
                from scipy.stats import gaussian_kde as _kde

                raw_kde = _kde(qoi_values)
                ax.plot(x_grid, raw_kde(x_grid), color="tomato", lw=1.5, ls="--", label="KDE (raw samples)")
            except ImportError:
                logger.warning(
                    "scipy is not installed; KDE curve skipped. "
                    "Install scipy to enable: pip install scipy"
                )
            except Exception as exc:
                logger.warning(f"KDE could not be computed for '{qoi_name}': {exc}")

        # ---- prior / reference distribution overlay ----------------------
        twin_ax = None
        if prior_dist is not None:
            try:
                # chaospy distribution
                if hasattr(prior_dist, "pdf"):
                    prior_pdf = np.asarray(prior_dist.pdf(x_grid), dtype=float)
                    prior_pdf = np.where(np.isfinite(prior_pdf), prior_pdf, 0.0)
                    twin_ax = ax.twinx()
                    twin_ax.plot(
                        x_grid, prior_pdf, color="purple", lw=1.5, ls=":", label="Prior PDF (reference)"
                    )
                    twin_ax.set_ylabel("Prior PDF", color="purple")
                    twin_ax.tick_params(axis="y", labelcolor="purple")
            except Exception as exc:
                logger.warning(f"Prior PDF could not be plotted for '{qoi_name}': {exc}")

        # ---- reference lines (mean, ±1σ, percentiles) --------------------
        ax.axvline(mean_val, color="black", lw=1.5, ls="-", label=f"Mean = {mean_val:.3g}")
        ax.axvline(mean_val + std_val, color="grey", lw=1.0, ls="--", label=f"+1σ = {mean_val + std_val:.3g}")
        ax.axvline(mean_val - std_val, color="grey", lw=1.0, ls="--", label=f"−1σ = {mean_val - std_val:.3g}")
        ax.axvline(p10, color="darkorange", lw=1.0, ls="-.", label=f"10th pct = {p10:.3g}")
        ax.axvline(p90, color="darkorange", lw=1.0, ls="-.", label=f"90th pct = {p90:.3g}")

        # ---- labels & legend --------------------------------------------
        qty_info = self.quantities_descriptor.get(qoi_name, {})
        qty_label = qty_info.get("name", qoi_name)
        qty_unit = qty_info.get("unit", "")
        xlabel = f"{qty_label} [{qty_unit}]" if qty_unit else qty_label
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Probability density")
        ax.set_title(f"Marginal distribution of '{qoi_name}'")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Combine legends when twin axis is used
        if twin_ax is not None:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = twin_ax.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

        fig.tight_layout()

        # ---- save -------------------------------------------------------
        if filename is None:
            raw_name = f"{qoi_name}_distribution"
            if timestamp:
                raw_name = add_timestamp_to_filename(raw_name + ".pdf", timestamp)
            else:
                raw_name = raw_name + ".pdf"
            filename = raw_name

        save_path = os.path.join(foldername, filename) if foldername else filename
        fig.savefig(save_path)
        plt.close(fig)
        logger.info(f"Distribution plot saved: {save_path}")
        return save_path

    def plot_scan_results(self, scan_results, foldername, timestamp):
        """
        Plot results from parameter scan.
        """
        # Placeholder for actual plotting code
        print("Plotting scan results... (functionality to be implemented)")

        # Example: Save scan results to a YAML file
        results = {
            "description": "Parameter scan results",
            "data": {
                # Example data structure
                "scan_values": [1, 2, 3],
                "results": [0.1, 0.2, 0.3],
            },
        }

        filename = add_timestamp_to_filename("scan_results.yaml")
        # serialize_yaml(results, filename) # TODO - implement, or copy

        print(f"✓ Scan results saved to: {filename}")
        return filename
