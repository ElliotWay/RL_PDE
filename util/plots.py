import os
import re
import numpy as np
import pandas as pd
import scipy.stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import util.colors as colors

def generate_polynomial(order, grid_sizes, comparison_error):
    """
    Generate a curve to represent the order of error approximation in e.g. a convergence plot.

    Parameters
    ---------
    order : int
        The order of approximation.
    grid_sizes : [int]
        The x values on which to plot the polynomial curve.
    comparison_error : [float]
        Real error values for an error plot. The polynomial curve will be adjusted to be near the
        comparison error. comparison_error should have the same size as grid_sizes.
        The polynomial will be cropped so that it diverges by at most 2 orders of magnitude below
        the comparison error.

    Returns
    -------
    grid_sizes : [float]
        x values of the polynomial curve. The same as the grid_sizes passed in unless cropped to
        limit distance from comparison_error.
    polynomial_values : [float]
        Values for the polynomial curve.
    label : str
        Standard O-notation label for this order of approximation.
    """         
    values = (1.0 / grid_sizes) ** order

    # Scale so it intersects around the middle of the comparison error.
    #error_coord = np.sqrt((max(comparison_error) * min(comparison_error)))
    #size_coord = np.sqrt((max(grid_sizes) * min(grid_sizes)))
    error_coord = comparison_error[0]
    size_coord = grid_sizes[0]
    values = error_coord * (size_coord ** order) * values

    if np.log10(min(comparison_error)) - np.log10(values[-1]) > 2:
        cutoff_index = np.argmax(np.log10(min(comparison_error)) - np.log10(values) > 2)
        values = values[:cutoff_index]
        out_grid_sizes = np.array(grid_sizes[:cutoff_index])
    else:
        out_grid_sizes = np.array(grid_sizes)

    label = r"$\mathcal{{O}}(\Delta x^{{{}}})$".format(order)

    return out_grid_sizes, values, label
 
def convergence_plot(grid_sizes, errors, log_dir, name="convergence.png", labels=None,
        kwargs_list=None, title=None):
    """
    Create a convergence plot of grid size vs. L2 error.

    One or more sets of errors can be plotted. With one, the line will be black and have no label.
    With more than one, the colors will follow the default matplotlib color cycle and may
    optionally have a labeled legend.

    Multiple sets of error can use either one list of grid sizes or multiple. With one, the grid
    sizes will be broadcast to each set of errors. With multiple, each set of errors will use the
    corresponding grid sizes.

    Parameters
    ----------
    grid_sizes : [int] or [[int]]
        The sizes of grids for which the L2 error was computed. A single list will be applied to
        each list of errors (if there are more than one); multiple lists can be used if different
        lists of errors have different grid sizes.
    errors : [float] or [[float]]
        The L2 error for each grid size. Can plot one or more set of errors.
    log_dir : str
        Path of the directory to save the convergence plot to.
    name : str
        Name of the file to save into log_dir. 'convergence.png' by default.
    label : [str]
        Labels for each set of errors, if there are more than one.
    kwargs_list : [dict]
        Kwargs passed to plot(), e.g. color and linestyle, for each set of errors.
    title : str
        Title to give to the plot. No title by default.
    """
    try:
        _ = iter(grid_sizes[0])
    except TypeError:
        multiple_size_lists = False
    else:
        multiple_size_lists = True

    try:
        _ = iter(errors[0])
    except TypeError:
        multiple_error_lists = False
    else:
        multiple_error_lists = True

    if multiple_size_lists and not multiple_error_lists:
        raise ValueError("Can't use {} lists of grid sizes for only one set of L2 errors.".format(
                                                                        len(grid_sizes)))

    if multiple_error_lists:
        for index, error_list in enumerate(errors):
            if multiple_size_lists:
                sizes = grid_sizes[index]
            else:
                sizes = grid_sizes
            if kwargs_list is not None:
                kwargs = kwargs_list[index]
            else:
                kwargs = {}

            if labels is not None:
                label = labels[index]
                plt.plot(sizes, error_list, label=label, **kwargs)
            else:
                plt.plot(sizes, error_list, **kwargs)
                
    else:
        plt.plot(grid_sizes, errors, color='k', marker='.')

    ax = plt.gca()
    ax.set_xlabel("grid size")
    #ax.set_xticks(grid_sizes) # Use the actual grid sizes as ticks instead of powers of 10.
    ax.set_xscale('log')
    ax.set_ylabel("L2 error")
    ax.set_yscale('log')

    if multiple_error_lists and labels is not None:
        ax.legend(loc="lower left")
    if title is not None:
        ax.set_title(title)

    filename = os.path.join(log_dir, name)
    plt.savefig(filename)
    print('Saved plot to ' + filename + '.')
    plt.close()

def action_plot(x_vals, action_vals, x_label, labels, log_dir, name="actions.png", title=None,
        vector_parts=None, action_parts=None, kwargs_list=None):
    """
    Create a plot of actions.

    Intended to compare actions from different configurations, such as different agents.
    Can handle x as the horizontal axis or time.

    The number of action parts must be the same for each vector component.

    Parameters
    ----------
    x_vals : [float] OR [[float]]
        The locations (or times) for each action. Either a single list for every configuration, or
        a list of different lists for each configuration.
    action_vals : [[[[float]]]]
        The actions at each x for each configuration.
        Axes are [configuration, vector, action_part, x].
    x_label : str
        Label for the horizontal dimension.
    labels : [str]
        The label to apply to each configuration.
    log_dir : str
        Path of the directory to save the convergence plot to.
    name : str
        Name of the file to save into log_dir. 'error_over_x.png' by default.
    title : str
        Title to give to the plot. No title by default.
    vector_parts : str
        Name of each part of the vector. [u1, u2, ...] by default.
    action_parts : str
        Name of each action part. [w1, w2, ...] by default.
    kwargs_list : [dict]
        Kwargs passed to plot(), e.g. color and linestyle, for each action configuration.
    """

    try:
        iterator = iter(x_vals[0])
    except TypeError:
        # broadcast_to creates a view that looks like the array repeated multiple times,
        # but uses the same space as the original array.
        x_vals = np.broadcast_to(x_vals, (len(action_vals), len(x_vals)))

    vector_dimensions = len(action_vals[0])
    action_dimensions = len(action_vals[0][0])

    if vector_parts is None:
        vector_parts = [f"$u_{i}$" for i in range(vector_dimensions)]
    if action_parts is None:
        action_parts = [f"$w_{i}$" for i in range(action_dimensions)]
    if kwargs_list is None:
        kwargs_list = [{}] * len(action_vals)

    vertical_size = 5 * vector_dimensions
    horizontal_size = 4 * action_dimensions
    fig, axes = plt.subplots(vector_dimensions, action_dimensions, sharex=True, sharey=True,
                             figsize=(horizontal_size, vertical_size), squeeze=False)

    # x values are [config, x].
    # Actions are [config, vector, action_part, x].
    # Subplot axes are [vector, action_part].
    for x, action, label, kwargs in zip(x_vals, action_vals, labels, kwargs_list):
        for vector_actions, vector_axes in zip(action, axes):
            for action_part, ax in zip(vector_actions, vector_axes):
                ax.plot(x, action_part, label=str(label), **kwargs)

    # Only put the x label on the bottom row of plots
    # and the action part title on the top row.
    for action_index in range(action_dimensions):
        axes[-1][action_index].set_xlabel(x_label)
        axes[0][action_index].set_title(action_parts[action_index])

    # And the vector part label on the left column row.
    for vector_index in range(vector_dimensions):
        axes[vector_index][0].set_ylabel(vector_parts[vector_index])

    # And the legend in only the top right plot.
    axes[-1][-1].legend()

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()

    filename = os.path.join(log_dir, name)
    plt.savefig(filename)
    print("Saved plot to " + filename + ".")
    plt.close(fig)

def error_plot(x_vals, error_vals, labels, log_dir, name="error_over_x.png", title=None,
        vector_parts=None):
    """
    Create a plot of x location vs error.

    Intended to compare between the error of many configurations, e.g. the different sizes in a
    convergence plot, or using different agents.

    The y axis (the error) will use log scaling.

    The data must be 1 dimensional, though it may be useful to plot slices of higher dimensional
    data.

    For 10 or fewer configurations, the default matplotlib color cycle is used. For more than 10
    configurations, each line is assigned a color from a spectrum based on its label, and a
    colorbar is added to the plot.
    The labels are expected to be numeric and from a logarithmic range in the >10 case.

    Parameters
    ----------
    x_vals : [float] or [[float]]
        The x location for the errors in every list of errors, or a list of x locations
        corresponding to each list of errors.
    error_vals : [[[float]]]
        The error(s) at each point for each configuration.
        Axes are [configuration, vector, location].
    labels : [str] or [number]
        The label to apply to each configuration. Must be numerical for >10 configurations;
        otherwise strings can be used.
    log_dir : str
        Path of the directory to save the convergence plot to.
    name : str
        Name of the file to save into log_dir. 'error_over_x.png' by default.
    title : str
        Title to give to the plot. No title by default.
    vector_parts : str
        Name of each part of the vector. [u1, u2, ...] by default.
    """

    try:
        iterator = iter(x_vals[0])
    except TypeError:
        # broadcast_to creates a view that looks like the array repeated multiple times,
        # but uses the same space as the original array.
        x_vals = np.broadcast_to(x_vals, (len(error_vals), len(x_vals)))

    vec_len = len(error_vals[0])
    fig, ax = plt.subplots(nrows=vec_len, ncols=1, figsize=[6.4, 4.8 * vec_len], dpi=100,
            squeeze=False)

    MAX_COLORS = 10
    if len(labels) <= MAX_COLORS:
        for x, y, label in zip(x_vals, error_vals, labels):
            for i in range(vec_len):
                ax[i][0].plot(x, y[i], ls='-', label=str(label))
    else:
        color_map = matplotlib.cm.get_cmap('viridis')
        normalize = matplotlib.colors.LogNorm(vmin=min(labels), vmax=max(labels))
        for x, y, label in zip(x_vals, error_vals, labels):
            for i in range(vec_len):
                ax[i][0].plot(x, y[i], ls='-', color=color_map(normalize(label)))
        scalar_mappable = matplotlib.cm.ScalarMappable(norm=normalize, cmap=color_map)
        scalar_mappable.set_array(labels)

    for i in range(vec_len):
        ax[i][0].set_xlabel("$x$")
        if vector_parts is None:
            ax[i][0].set_ylabel(f"u{i} |error|")
        else:
            # The $s render e.g. rho correctly.
            ax[i][0].set_ylabel(f"${vector_parts[i]}$ |error|")

        ax[i][0].set_yscale('log')
        ax[i][0].set_ymargin(0.0)

        if len(labels) <= MAX_COLORS:
            ax[i][0].legend()
        else:
            fig.colorbar(scalar_mappable, ax=ax[i][0], label="nx")
    if title is not None:
        fig.suptitle(title)

    #extreme_cutoff = 3.0
    #max_not_extreme = max([np.max(y[y < extreme_cutoff]) for y in error_vals])
    #ymax = max_not_extreme*1.05 if max_not_extreme > 0.0 else 0.01
    #ax.set_ylim((None, ymax))

    filename = os.path.join(log_dir, name)
    plt.savefig(filename)
    print("Saved plot to " + filename + ".")
    plt.close(fig)

#TODO, possibly: should this handle vectors? I.e. plot some value for each vector component over
# time?
def plot_over_time(times, values, log_dir, name, scaling='linear',
        ylabel=None, labels=None, kwargs_list=None, title=None):
    """
    Plot some scalar over the time of an episode.

    If possible use the actual time, not the timestep.

    Parameters
    ----------
    times : [float] or [[float]]
        Time values for the x-axis. Either one list for every set of values, or a separate list for
        each set of values.
    values : [float] or [[float]]
        The values of the y-axis. May be one list of values, or multiple for multiple lines.
    log_dir : string
        Location to save to.
    name : string
        Name of the file in log_dir to save to.
    scaling : string
        The scaling to apply to the y-axis (e.g. 'linear', 'log', 'symlog').
    ylabel : string
        Name of what is being plotted against time.
    labels : string or [string]
        Label(s) for the plotted values. Either one for every or a list for each.
    kwargs_list : dict or [dict]
        Kwargs passed to plot, e.g. color and linestyle. Either one for every or a list for each.
    title : str
        Title of the plot. No title by default.
    """
    try:
        _ = iter(times[0])
    except TypeError:
        multiple_time_lists = False
    else:
        multiple_time_lists = True

    try:
        _ = iter(values[0])
    except TypeError:
        multiple_value_lists = False
    else:
        multiple_value_lists = True
    if multiple_time_lists and not multiple_value_lists:
        raise ValueError(f"Can't use {len(times)} lists of times for only one set of values.")

    if kwargs_list is not None:
        try:
            _ = kwargs_list.items()
        except (AttributeError, TypeError):
            multiple_kwargs_dicts = True
        else:
            multiple_kwargs_dicts = False
    if labels is not None:
        try:
            _ = iter(labels[0])
        except TypeError:
            multiple_labels = False
        else:
            multiple_labels = True
        if multiple_labels and not multiple_value_lists:
            raise ValueError(f"Can't use {len(labels)} for only one set of values.")
        if multiple_value_lists and not multiple_labels:
            raise ValueError(f"Can't use the same label for {len(values)} sets of values.")

    if multiple_value_lists:
        for index, value_list in enumerate(values):
            if multiple_time_lists:
                time_list = times[index]
            else:
                time_list = times

            if kwargs_list is None:
                kwargs = {}
            elif multiple_kwargs_dicts:
                kwargs = kwargs_list[index]
            else:
                kwargs = kwargs_list

            if labels is not None:
                label = labels[index]
                plt.plot(time_list, value_list, label=label, **kwargs)
            else:
                plt.plot(time_list, value_list, **kwargs)
                
    else:
        if kwargs_list is not None:
            if multiple_kwargs_dicts:
                kwargs = kwargs_list[0]
            else:
                kwargs = kwargs_list
        else:
            kwargs = {}
        if 'color' not in kwargs or 'c' not in kwargs:
            kwargs['color'] = 'black'

        if labels is not None:
            label = labels
            plt.plot(times, values, label=label, **kwargs)
        else:
            plt.plot(times, values, **kwargs)

    ax = plt.gca()
    ax.set_xlabel("time")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_yscale(scaling)

    if labels is not None:
        ax.legend()
    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    filename = os.path.join(log_dir, name)
    plt.savefig(filename)
    print('Saved plot to ' + filename + '.')
    plt.close()


# float regex from https://stackoverflow.com/a/12929311/2860127
FLOAT_REGEX = r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?'

def add_average_with_ci(ax, x, ys, ci_type="range", label=None, plot_kwargs=None):
    """
    On an existing axis, plot the mean of multiple data series with a shaded region around it for a
    confidence interval.

    Parameters
    ----------
    ax : Axes
        The axes to plot on.
    x : [float]
        The x values of the data.
    y : [[float]]
        The y values for each series of data.
        Axes are [series, x].
    ci_type : str
        The type of confidence interval to plot. Options are:
        range: [min,max]
        Xconf: [P(lower)=(1-X)/2,P(higher)=(1-X)/2] (T dist), X in [0,1]
        Xsig: [-X std deviations,+X std deviations] (normal dist), X > 0
        Nperc: [Nth percentile,100-Nth percentile], N in [0, 50]
        none: (only plot the average, no confidence interval)
    label : str
        Label of the average line.
    plot_kwargs : dict
        Kwargs to pass to the plot function, e.g. color and linestyle.
    """
    if plot_kwargs is None:
        plot_kwargs = {}

    y_data = np.array(ys)
    y_mean = np.mean(y_data, axis=0)

    if label is None:
        mean_line = ax.plot(x, y_mean, **plot_kwargs)[0]
    else:
        mean_line = ax.plot(x, y_mean, label=label, **plot_kwargs)[0]

    if ci_type is not None and ci_type != "none":
        if ci_type == "range":
            lower = np.min(y_data, axis=0)
            upper = np.max(y_data, axis=0)
        elif re.fullmatch(f"{float_regex}conf", ci_type):
            confidence = float(re.fullmatch(f"({float_regex})conf", ci_type).group(1))
            size = len(y_data)
            if confidence < 0.0 or confidence > 1.0:
                raise ValueError()
            t_constant = float(scipy.stats.t.ppf((1.0 - confidence)/2.0, df=(size - 1)))
            ci_size = t_constant * np.std(y_values, ddof=1, axis=0)/np.sqrt(size)
            lower = y_mean - ci_size
            upper = y_mean + ci_size
        elif re.fullmatch(f"{float_regex}sig", ci_type):
            num_sigmas = float(re.fullmatch(f"({float_regex})sig", ci_type).group(1))
            if num_sigmas < 0.0:
                raise ValueError()
            ci_size = num_sigmas * np.std(y_values, axis=0)
            lower = y_mean - ci_size
            upper = y_mean + ci_size
        elif re.fullmatch(f"{float_regex}perc", ci_type):
            lower_percentile = float(re.fullmatch(f"({float_regex})perc", ci_type).group(1))
            if lower_percentile < 0.0 or lower_percentile > 50.0:
                raise ValueError()
            upper_percentile = 100.0 - lower_percentile
            lower = np.percentile(y_data, lower_percentile, axis=0)
            upper = np.percentile(y_data, upper_percentile, axis=0)
        else:
            raise ValueError()

        mean_color = mean_line.get_color()
        ax.fill_between(x, lower, upper, color=mean_color, alpha=0.1)

def crop_early_shift(ax, mode):
    """
    Crop a major change in the begining of the plot.

    We expect e.g. loss to drop quickly in the early episodes, but if it starts high enough that,
    even with a log plot, it's hard to distinguish the rest of the data, then we can crop off the
    high range of that early drop.
    If the first point is more than 2 orders of magnitude above the 95% percentile, restrict the
    range of the y axis to that point.

    Parameters
    ----------
    ax : Axes
        The axes of the plot to crop.
    mode : string
        "normal" for cropping a rapid drop from large positive values.
        "flipped" for cropping a rapid increase from large negative values.
    """

    percentile_limit = 5
    order_limit = 2

    data = [line.get_ydata() for line in ax.get_lines()]
    if data:
        firsts = [d[0] for d in data]
        all_data = np.concatenate(data)
        if mode == "normal":
            max_first = max(firsts)
            high_percentile = np.percentile(all_data, 100 - percentile_limit)
            if ((max_first > 0 and high_percentile > 0) and
                    np.log10(max_first) - np.log10(high_percentile) > order_limit):
                ax.set_ylim(top=(high_percentile * (10 ** order_limit)))
        elif mode == "flipped":
            # Values are high magnitude negative. (Not low magnitude.)
            min_first = min(firsts)
            low_percentile = np.percentile(all_data, percentile_limit)
            if ((min_first < 0 and low_percentile < 0) and
                    np.log10(-min_first) - np.log10(-low_percentile) > order_limit):
                ax.set_ylim(bottom=(low_percentile * (10 ** order_limit)))

def plot_reward_summary(csv_file, output_file, total_episodes, eval_env_names=None,
        only_eval=False):
    if not isinstance(csv_file, pd.DataFrame):
        csv_df = pd.read_csv(csv_file, comment='#')
    else:
        csv_df = csv_file

    # Assume new name format.
    if eval_env_names is None:
        eval_env_names = [re.fullmatch("eval_(.+)_reward", name).group(1)
                    for name in list(csv_df) if re.fullmatch("eval_.+_reward", name)]
        eval_env_prefixes = [f"eval_{name}" for name in eval_env_names]
    # Check for new name format.
    elif f"eval_{eval_env_names[0]}_reward" in csv_df:
        eval_env_prefixes = [f"eval_{name}" for name in eval_env_names]
    # Old name format. (Kept in case this function is being called from somewhere besides
    # run() to update an old experiment.)
    else:
        eval_env_prefixes = [f"eval{num+1}" for num in range(len(eval_env_names))]

    episodes = csv_df['episodes']

    reward_fig = plt.figure()
    ax = reward_fig.gca()
    
    all_rewards = []
    if not only_eval:
        if 'avg_train_total_reward' in csv_df:
            train_reward = csv_df['avg_train_total_reward']
            ax.plot(episodes, train_reward, color=colors.TRAIN_COLOR, label="train")
        if len(eval_env_prefixes) > 1 and 'avg_eval_total_reward' in csv_df:
            avg_eval_reward = csv_df['avg_eval_total_reward']
            ax.plot(episodes, avg_eval_reward, color=colors.AVG_EVAL_COLOR, label="eval avg")
    for i, (name, prefix) in enumerate(zip(eval_env_names, eval_env_prefixes)):
        eval_reward = csv_df[f'{prefix}_reward']
        linestyle = '-' if only_eval else '--'
        ax.plot(episodes, eval_reward,
                color=colors.EVAL_ENV_COLORS[i], ls=linestyle, label=name)

    reward_fig.legend(loc="lower right")
    ax.set_xlim((0, total_episodes))
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel('episodes')
    ax.set_ylabel('reward')
    ax.grid(True)
    # Use symlog as the rewards are negative.
    ax.set_yscale('symlog')
    crop_early_shift(ax, "flipped")

    reward_fig.savefig(output_file)
    plt.close(reward_fig)

def plot_l2_summary(csv_file, output_file, total_episodes, eval_env_names=None,
        only_eval=False):
    if not isinstance(csv_file, pd.DataFrame):
        csv_df = pd.read_csv(csv_file, comment='#')
    else:
        csv_df = csv_file

    # Assume new name format.
    if eval_env_names is None:
        eval_env_names = [re.fullmatch("eval_(.+)_end_l2", name).group(1)
                    for name in list(csv_df) if re.fullmatch("eval_.+_end_l2", name)]
        eval_env_prefixes = [f"eval_{name}" for name in eval_env_names]
    # Check for new name format.
    elif f"eval_{eval_env_names[0]}_end_l2" in csv_df:
        eval_env_prefixes = [f"eval_{name}" for name in eval_env_names]
    # Old name format. (Kept in case this function is being called from somewhere besides
    # run() to update an old experiment.)
    else:
        eval_env_prefixes = [f"eval{num+1}" for num in range(len(eval_env_names))]

    episodes = csv_df['episodes']

    l2_fig = plt.figure()
    ax = l2_fig.gca()
    if not only_eval:
        if 'avg_train_end_l2' in csv_df:
            train_l2 = csv_df['avg_train_end_l2']
            ax.plot(episodes, train_l2, color=colors.TRAIN_COLOR, label="train")
        if len(eval_env_prefixes) > 1 and 'avg_eval_end_l2' in csv_df:
            avg_eval_l2 = csv_df['avg_eval_end_l2']
            ax.plot(episodes, avg_eval_l2, color=colors.AVG_EVAL_COLOR, label="eval avg")
    for i, (name, prefix) in enumerate(zip(eval_env_names, eval_env_prefixes)):
        eval_l2 = csv_df[f'{prefix}_end_l2']
        linestyle = '-' if only_eval else '--'
        ax.plot(episodes, eval_l2,
                color=colors.EVAL_ENV_COLORS[i], ls=linestyle, label=name)

    l2_fig.legend(loc="upper right")
    ax.set_xlim((0, total_episodes))
    ax.set_title("L2 Error with WENO at End of Episode")
    ax.set_xlabel('episodes')
    ax.set_ylabel('L2 error')
    ax.grid(True)
    ax.set_yscale('log')
    crop_early_shift(ax, "normal")

    l2_fig.savefig(output_file)
    plt.close(l2_fig)

def plot_loss_summary(csv_file, output_file, total_episodes):
    if not isinstance(csv_file, pd.DataFrame):
        csv_df = pd.read_csv(csv_file, comment='#')
    else:
        csv_df = csv_file

    episodes = csv_df['episodes']

    loss_fig = plt.figure()
    ax = loss_fig.gca()
    if 'loss' in csv_df:
        loss = csv_df['loss']
    elif 'policy_loss' in csv_df:
        loss = -csv_df['policy_loss']
    else:
        raise Exception("Can't find loss in progress.csv file.")
    ax.plot(episodes, loss, color='k')
    ax.set_xlim((0, total_episodes))
    ax.set_title("Loss Function")
    ax.set_xlabel('episodes')
    ax.set_ylabel('loss')
    ax.grid(True)
    ax.set_yscale('log')
    crop_early_shift(ax, "normal")

    loss_fig.savefig(output_file)
    plt.close(loss_fig)


