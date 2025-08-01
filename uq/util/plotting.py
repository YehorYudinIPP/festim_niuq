
import matplotlib.pyplot as plt
import numpy as np

from .utils import add_timestamp_to_filename

def plot_unc_vs_r(r, y, sy, y10, y90, qoi_name, foldername="", filename=""):
    """
    Plot uncertainty in the results as a function of radius (spatial coordinates).
    """
    fig, ax = plt.subplots()

    # ax.plot(r, y, label=f'<y> at {qoi_name}')
    #TODO plot in semilogy
    ax.semilogy(r, y, label=f'<y> at {qoi_name} (lin-log)')

    ax.fill_between(r, y - sy, y + sy, alpha=0.3, label='+/- STD')
    ax.fill_between(r, y10, y90, alpha=0.1, label='10% - 90%')

    #TODO plot individual trajectories!

    ax.set_title(f"Uncertainty at {qoi_name} as a function of radius")
    ax.set_xlabel("#Radius, [m]") # TODO pass and display proper units for the length
    ax.set_ylabel(f"Concentration [m^-3] at {qoi_name}") #TODO read full name of the QoI from results

    ax.legend(loc='best')
    ax.grid(True)

    fig.savefig(f"{foldername}/bespoke_{filename}")

    plt.close()  # Close the plot to avoid display issues in some environments
    return 0

def plot_unc_qoi(stats_dict_s, qoi_name, foldername="", filename="", r_ind=0):
    """
    Plot uncertainty in the specific scalar QoIs.
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

def plot_stats_vs_r(results, qois, plot_folder_name, plot_timestamp, rs=None):
    """
    Plot statistics of the results as a function of radius (spatial coordinates).
    """

    # Specific for common boxplot for QoIs
    stats_dict_s = []
    r_ind_qoi = 0  # Select the max radius values: r=0.0 should be physical centre of domain
    
    # Rund over QoIs in analysis results object
    for qoi in qois:
        # Generate filenames with timestamp
        moments_vsr_filename = add_timestamp_to_filename(f"{qoi}_moments_vs_r.png", plot_timestamp)
        sobols_treemap_filename = add_timestamp_to_filename(f"{qoi}_sobols_treemap.png", plot_timestamp)
        sobols_filename = add_timestamp_to_filename(f"{qoi}_sobols_first_vs_r.png", plot_timestamp)
        
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
        plot_unc_vs_r(rs, y, sy, y10, y90, qoi_name=qoi, foldername=plot_folder_name, filename=moments_vsr_filename)

        # Plotting Sobol indices as a function of radius
        #print(f" >> Plotting first Sobol indices for QoI via EasyVVUQ: {qoi}") ###DEBUG
        results.plot_sobols_first(
            qoi=qoi,
            withdots=False,  # Show dots for each Sobol index
            xlabel=f"Radius, #vertices",
            ylabel=f"Sobol Index (first) at {qoi}",
            filename=f"{plot_folder_name}/{sobols_filename}",  # Save with bespoke prefix
        )

        #TODO add total Sobol indices as well, probaly higher order separately

        print(f"Plots (for spatially resolved functions) saved: {moments_vsr_filename}, {sobols_treemap_filename}, {sobols_filename}")
        #TODO compare those in absolute values - fix the y axis limits?

    # Save plot common for QoIs: specific for bespoke QoI uncertainty plotting
    #  - bespoke plotting of uncertainty in QoI (at selected radius)
    #print(f" >> Plotting uncertainties for QoI via bespoke functionality: {qoi}") ###DEBUG
    plot_unc_qoi(stats_dict_s, qoi_name=qoi, foldername=plot_folder_name, filename=add_timestamp_to_filename("qoi_uncertainty_vs_r.png", plot_timestamp),r_ind=r_ind_qoi)

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
        moments_vst_filename = add_timestamp_to_filename(f"moments_vs_t_at_{r_ind}.png", plot_timestamp)
        sobols_vst_filename = add_timestamp_to_filename(f"sobols_first_vs_t_at_{r_ind}.png", plot_timestamp)

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
