"""
This module contains functions for plotting uncertainty quantification (UQ) results from the FESTIM model.
It includes functions to plot uncertainty as a function of radius, time, and Sobol indices,
as well as functions to visualize statistics of the results.
It is designed to work with EasyVVUQ and FESTIM libraries, providing a way to visualize the results of UQ campaigns.   

Created by: Yehor Yudin
Date: July 2025
"""

import matplotlib.pyplot as plt
import numpy as np

import itertools
import json

from .utils import add_timestamp_to_filename

def plot_unc_vs_r(r, y, sy, y10, y90, qoi_name:str, foldername:str="", filename:str="", runs_info=None):
    """
    Plot uncertainty in the results as a function of radius (spatial coordinates).

    Parameters:
    - r: array of radius / 1D coordinate values
    - y: array of mean values at each radius
    - sy: array of standard deviation values at each radius
    - y10: array of 10% quantile values at each radius
    - y90: array of 90% quantile values at each radius
    - qoi_name: name of the quantity of interest (QoI) for labeling
    - foldername: folder to save the plot
    - filename: name of the file to save the plot
    - runs_info: information about individual runs, if available (optional); should be a list of dictionaries, with 'id' and 'results'
    """

    # making an array of plot for different axis scales
    plot_types = ['plot', 'semilogy']  # Add more plot types if needed
    n_plots = len(plot_types)

    fig, axs = plt.subplots(1, n_plots, figsize=(n_plots * 8, 6))

    # Iterative over type of axis scales
    for i, plot_func_name in enumerate(plot_types):

        # Chosing the plotting function based on the plot_func_name (axis scale)
        plot_func = getattr(axs[i], plot_func_name, None)
        if plot_func is None:
            raise ValueError(f"Plot function '{plot_func_name}' is not supported. Use 'plot' or 'semilogy'.")

        # Plotting the mean values with error bars
        plot_func(r, y, label=f'<y> at {qoi_name}')

        # Plotting the standard deviation as a shaded area
        axs[i].fill_between(r, y - sy, y + sy, alpha=0.3, label='+/- STD')

        # Plotting the 10% and 90% quantiles as a shaded area
        axs[i].fill_between(r, y10, y90, alpha=0.1, label='10% - 90%')

        # Plotting individual trajectories for each run if runs_info is provided

        if runs_info is not None:
            print(f" > Plotting individual trajectories for each run in {qoi_name} at '{plot_func_name}' scale")  ###DEBUG

            # Iterating over individual runs
            for run_id, run_info in runs_info:
                print(f" >> Plotting run {run_id} for {qoi_name}")  ###DEBUGs
                #print(f" >> Run {run_id} info: {run_info}")  ###DEBUG

                # Checking if individual run has non-empty results
                if 'result' in run_info:

                    # Deserializing the result to a dictionary
                    result_str = run_info['result'] # This is a string

                    result_dict = json.loads(result_str) # This SHOULD BE a dictionary
                    #print(f" >> Run {run_id} result_dict type: {type(result_dict)}, content: {result_dict}")  ###DEBUG

                    # Plotting the individual trajectory for the current run
                    plot_func(
                        result_dict["x"], 
                        result_dict[qoi_name],
                        alpha=0.5,
                        color='gray',
                        label=f"Individual runs trajectories" if run_id == 1 else None
                        )
                    
                else:
                    print(f"Run {run_id} does not have 'result' key, skipping individual trajectory plotting.")
            print(" > Individual trajectories plotted for each run.")  ###DEBUG
        else:
            print("No runs_info provided, skipping individual trajectories plotting.")

        # Setting the title and labels for the plot
        axs[i].set_title(f"Uncertainty at {qoi_name} as a function of radius, in '{plot_func_name}' scale")
        axs[i].set_xlabel("Radius, [m]") # TODO pass and display proper units for the length
        axs[i].set_ylabel(f"Concentration [m^-3] at {qoi_name}") #TODO read full name of the QoI from results


        axs[i].legend(loc='best')
        axs[i].grid(True)

    # Save the figure with a bespoke filename
    fig.savefig(f"{foldername}/bespoke_{filename}")

    plt.close()  # Close the plot to avoid display issues in some environments
    return 0

def plot_unc_qoi(stats_dict_s:dict, qoi_name:str, foldername:str="", filename:str="", r_ind:int=0):
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

    #Boxplotting the mean and std at a single radius

    #ax.plot(qoi_name, y[r_ind], 'o', label=f"<y> at r={0} and {qoi_name}")

    #ax.errorbar(qoi_name, y[r_ind], yerr=sy[r_ind], fmt='o', label=f"+/- STD at r_ind={r_ind} and {qoi_name}")

    ax.bxp(
        stats_dict_s,
        patch_artist=True,
        showmeans=True,
        shownotches=True,
        #meanline=True,  # Show mean line
        label=f"QoIs at r.ind {r_ind}",
        #label=f"Mean, 95% CI, 10% - 90%, min - max",
    )

    #ax.fill_betweenx([y10[r_ind], y90[r_ind]], qoi - 0.01, qoi + 0.01, alpha=0.1, label=f"10% - 90% at r={0} and {qoi_name}")

    ax.set_ylabel(f"Concentration [m^-3] at radius r index [{r_ind}]")  # Assuming all QoIs have the same units
    ax.set_xlabel(f"Different times of a simulation")
    ax.set_title(f"Uncertainty in QoIs: mean, median, 95% CI, 10% - 90%, min - max")
    ax.legend(loc='best')
    ax.grid(axis='y')

    #fig_qoi.suptitle("Uncertainty in QoIs at selected radius")
    
    fig.savefig(f"{foldername}/{filename}")

    return 0

def plot_sobols_seconds_vs_r(r, sobols_second, qoi_name_s, foldername="", filename_base=""):
    """
    Plot second-order Sobol indices as a function of radius.
    Parameters:
    - sobols_second: dictionary of second-order Sobol indices (?)
    - foldername: folder to save the plot
    - filename: name of the file to save the plot
    """

    fig, ax = plt.subplots()
    
    #for qoi_name_1, qoi_name_2 in itertools.product(qoi_name_s, repeat=2): # TODO: should be param_name_s

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

                        ax.plot(r, 
                                sobols_second_qoi, 
                                label=f"{param_name_1} & {param_name_2}",
                                )
                        # fig.savefig(f"{foldername}/{filename_base}_{qoi_name_1}_{qoi_name_2}.pdf")

        ax.set_xlabel(f"Radius [m]")
        ax.set_ylabel(f"Sobol index [fraction of unity]")

        ax.set_title(f"Second-order Sobol indices vs Radius at {qoi_name}")
        ax.legend(loc='best')

        ax.grid()

        fig.savefig(f"{foldername}/{filename_base}_{qoi_name}_second_order_sobols.pdf")
        plt.close()

    return 0

def plot_stats_vs_r(results, qois:list[str], plot_folder_name:str, plot_timestamp:str, rs=None, runs_info=None):
    """
    Plot statistics of the results as a function of radius (spatial coordinates).
    Parameters:
    - results: analysis results object from EasyVVUQ
    - qois: list of quantities of interest (QoIs) names to plot, for labeling
    - plot_folder_name: folder to save the plots
    - plot_timestamp: timestamp to append to the filenames
    - rs: array of radius values (optional, if not provided, will be generated)
    - runs_info: information about individual runs, if available (optional); should be a list of dictionaries, with 'id' and 'results'
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
        y = results.describe(qoi, 'mean')
        ymed = results.describe(qoi, 'median')
        sy = results.describe(qoi, 'std')
        y01 = results.describe(qoi, '1%')
        y10 = results.describe(qoi, '10%')
        y90 = results.describe(qoi, '90%')
        y99 = results.describe(qoi, '99%')
        # ymin = results.describe(qoi, 'min')
        # ymax = results.describe(qoi, 'max')
        
        #print(f" >>> Finished reading statistics for QoI with EasyVVUQ: {qoi}") ### DEBUG

        # Filling in the values for the list of dicts for a common boxplot
        stats_dict_s.append({
            'mean': [y[r_ind_qoi]],
            'med': [ymed[r_ind_qoi]],
            'q1': [y10[r_ind_qoi]],
            'q3': [y90[r_ind_qoi]],
            'cilo': [y[r_ind_qoi] - 1.95* sy[r_ind_qoi]],
            'cihi': [y[r_ind_qoi] + 1.95* sy[r_ind_qoi]],
            'whislo': [y01[r_ind_qoi]],
            'whishi': [y99[r_ind_qoi]],
            'fliers': [],  # [ymin[r_ind_qoi], ymax[r_ind_qoi]],
            'label': f"{qoi}",
        })

        # Define a simple range for x-axis
        #rs = np.linspace(0., 1., len(y))  # Should be done outside of scope of current function

        # Read out individual trajectories of single runs from .raw_data
        print(f" >>> Reading individual trajectories for QoI with EasyVVUQ: results.raw_data = \n{results.raw_data}") ### DEBUG
        #TODO might be needed to read from campaign.db

        # Default plotting of the moments
        #print(f" >> Plotting moments for QoI with EasyVVUQ: {qoi}") ###DEBUG
        results.plot_moments(
            qoi=qoi,
            ylabel=f"Concentration [m^-3], {qoi}",
            xlabel=f"Radius, #vertices",
            filename=f"{plot_folder_name}/{moments_vsr_filename}",
        )
        print(f" >>> Finished plotting moments for QoI with EasyVVUQ: {qoi}")

        # Plotting Sobol indices as a treemap
        #TODO: figure out how to plot treemaps at arbitrary locations
        # results.plot_sobols_treemap(
        #     qoi=qoi,
        #     filename=f"{plot_folder_name}/{sobols_treemap_filename}",
        # )

        # Bespoke plotting of uncertainty in QoI (vs. radius)
        #print(f" >> Plotting moments for QoI via bespoke function: {qoi}") ###DEBUG
        plot_unc_vs_r(rs, y, sy, y10, y90, qoi_name=qoi, foldername=plot_folder_name, filename=moments_vsr_filename, runs_info=runs_info)

        # Plotting Sobol indices as a function of radius
        #print(f" >> Plotting first Sobol indices for QoI via EasyVVUQ: {qoi}") ###DEBUG
        results.plot_sobols_first(
            qoi=qoi,
            withdots=False,  # Show dots for each Sobol index
            xlabel=f"Radius, #vertices",
            ylabel=f"Sobol Index (first) at {qoi}",
            filename=f"{plot_folder_name}/{sobols_filename}",  # Save with bespoke prefix
        )

        print(f"Plots (for spatially resolved functions) saved: {moments_vsr_filename}, {sobols_treemap_filename}, {sobols_filename}")
        #TODO compare those in absolute values - fix the y axis limits?

    # Save plot common for QoIs: specific for bespoke QoI uncertainty plotting
    #  - bespoke plotting of uncertainty in QoI (at selected radius)
    #print(f" >> Plotting uncertainties for QoI via bespoke functionality: {qoi}") ###DEBUG
    file_type = "pdf"  # Assuming we want to save as PDF
    plot_unc_qoi(stats_dict_s, qoi_name=qoi, foldername=plot_folder_name, filename=add_timestamp_to_filename(f"qoi_uncertainty_vs_r.{file_type}", plot_timestamp),r_ind=r_ind_qoi)

    #TODO add total Sobol indices as well

    # Read second-order Sobol indices from the UQ results object
    sobols_second = results.sobols_second()
    print(f" >> Second-order Sobol indices for QoI with EasyVVUQ: {qoi} : \n {sobols_second}") ###DEBUG
    plot_sobols_seconds_vs_r(rs, sobols_second, qois, foldername=plot_folder_name, filename_base="sobols_second_vs_r")

    return 0

def plot_unc_vs_t(r_at_r, t_s, y_at_r, sy_at_r, y10_at_r, y90_at_r, foldername="", filename=""):
    """
    Plot uncertainty in the results as a function of time.
    """
    fig, ax = plt.subplots()
    # print(f"Shapes of the lists: y_s: {len(y_s)}, sy_s: {len(sy_s)}, y10_s: {len(y10_s)}, y90_s: {len(y90_s)}") ###DEBUG

    ax.plot(t_s, y_at_r, label=f'<y> at r={r_at_r}')
    ax.fill_between(t_s, np.array(y_at_r) - np.array(sy_at_r), np.array(y_at_r) + np.array(sy_at_r), alpha=0.3, label='+/- STD')
    ax.fill_between(t_s, y10_at_r, y90_at_r, alpha=0.1, label='10% - 90%')

    ax.set_title(f"Uncertainty as a function of time at r={r_at_r}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Concentration [m^-3] at {r_at_r}")
    ax.legend(loc='best')
    ax.grid(True)

    fig.savefig(f"{foldername}/{filename}")

    plt.close()

    return 0

def plot_sobols_vs_t(r_s, t_s, s1_s, distributions, foldername="", filename="", r_ind=0):
    """
    Plot Sobol indices as a function of time.
    """
    fig, ax = plt.subplots()
    # print(s1_s[-1]) ### DEBUG

    for i, param_name in enumerate(distributions.keys()):
        # Extract r_ind-th element from each Sobol array to get time series at fixed radius
        s1_at_r = [s1_timestep[r_ind] for s1_timestep in s1_s[i]]
        ax.plot(t_s, s1_at_r, label=f'Sobol Index (first) for {param_name}')

    ax.set_title(f"Sobol Indices as a function of time at r={r_s}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"Sobol Index (first), fraction of unity")
    ax.legend()

    fig.savefig(f"{foldername}/{filename}")

    plt.close()

    return 0

def plot_stats_vs_t(results, distributions, qois, plot_folder_name, plot_timestamp, rs=None):
    """
    Plot statistics of the results as a function of time.
    """

    # Select - uncertainty at the depth of the specimen (r=0.)
    r_ind_selected = [0, -1] # Select the first and last radius index (or any other index)

    # Read the results for all times and align data for plotting against time
    y_s = []
    sy_s = []
    y10_s = []
    y90_s = []
    r_s = []

    s1_s = [[] for _ in range(len(distributions))]  # Assuming one first Sobol index per distribution

    # op1) Extract time from QoI names
    #t_s = [float(qoi.split('=')[1].strip()) for qoi in qois]
    # op2) read from results
    t_s = []

    # Run over QoIs in analysis results object and read statistics
    for qoi in qois:
        # Every element in qois list is a single time step
        # Read every time step from results in a list-of-lists [n_timesteps x n_elements]
        t_s.append(float(qoi.split('=')[1].strip()[:-1]))  # Extract time from QoI names, strp 's' at the end and '='
        y_s.append(results.describe(qoi, 'mean'))
        sy_s.append(results.describe(qoi, 'std'))
        y10_s.append(results.describe(qoi, '10%'))
        y90_s.append(results.describe(qoi, '90%'))
        s1 = results.sobols_first(qoi) # returing a dict {input_param: (list of) Sobol index values}
        for i,param_name in enumerate(distributions.keys()):
            # Assuming each distribution is a valid QoI descriptor
            s1_s[i].append(s1[param_name])  # Assuming 'first' is a valid QoI descriptor
        #r_s.append(np.linspace(0., 1., len(y_s[-1])))  
        r_s.append(rs) # assuming we read the readius values from outside, and they are the same for all QoIs

    # Run over selected radius indices
    for r_ind in r_ind_selected:
        # Generate filenames with timestamp for time series plots
        file_type = "pdf"  # Assuming we want to save as PDF
        moments_vst_filename = add_timestamp_to_filename(f"moments_vs_t_at_{r_ind}.{file_type}", plot_timestamp)
        sobols_vst_filename = add_timestamp_to_filename(f"sobols_first_vs_t_at_{r_ind}.{file_type}", plot_timestamp)

        # Extract r_ind-th element from each time step (array) to get time series at fixed radius
        y_at_r = [y_timestep[r_ind] for y_timestep in y_s]
        sy_at_r = [sy_timestep[r_ind] for sy_timestep in sy_s]
        y10_at_r = [y10_timestep[r_ind] for y10_timestep in y10_s]
        y90_at_r = [y90_timestep[r_ind] for y90_timestep in y90_s]
        r_at_r = [r_timestep[r_ind] for r_timestep in r_s]  # Assuming r_s is a list of lists with radius values
        #TODO check if r_at_r changes with time, or is constant

        # Plotting of moments as a function of time
        plot_unc_vs_t(r_at_r[0], t_s, y_at_r, sy_at_r, y10_at_r, y90_at_r, foldername=plot_folder_name, filename=moments_vst_filename)

        # Plotting Sobol indices as a function of time
        plot_sobols_vs_t(r_at_r[0], t_s, s1_s, distributions, foldername=plot_folder_name, filename=sobols_vst_filename, r_ind=r_ind)

        print(f"Plots (for time series) saved: {moments_vst_filename}, {sobols_vst_filename}")

    return 0
