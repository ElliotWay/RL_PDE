import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
        The error values for a real error plot. The polynomial curve will be scaled and
        adjusted to be near the real error. comparison_error should have the same size as
        grid_sizes.

    Returns
    -------
    polynomial_values : [float]
        Values for the polynomial plot.
    label : str
        Standard O-notation label for this order of approximation.
    """         
    values = (1.0 / grid_sizes) ** order

    # Scale so it intersects around the middle of the comparison error.
    midpoint_error = np.sqrt((max(comparison_error) * min(comparison_error)))
    midpoint_size = np.sqrt((max(grid_sizes) * min(grid_sizes)))
    values = midpoint_error * (midpoint_size ** order) * values

    label = r"$\mathcal{{O}}(\Delta x^{{{}}})$".format(order)

    return values, label
 
def convergence_plot(grid_sizes, errors, log_dir, name="convergence.png", labels=None,
        kwargs_list=None, title=None, polynomials=None):
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
        the index of the error plot that the polynomial will be based on.
    kwargs_list : [dict]
        Kwargs passed to plot(), e.g. color and linestyle, for each set of errors.
    title : str
        Title to give to the plot. No title by default.
    """
    try:
        _ = iter(grid_sizes[0])
    except:
        multiple_size_lists = False
    else:
        multiple_size_lists = True

    try:
        _ = iter(errors[0])
    except:
        multiple_error_lists = False
    else:
        multiple_error_lists = True

    if multiple_size_lists and not multiple_error_lists:
        raise ValueError("Can't use {} lists of grid sizes for only one set or L2 errors.".format(
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
        ax.legend()
    if title is not None:
        ax.set_title(title)

    filename = os.path.join(log_dir, name)
    plt.savefig(filename)
    print('Saved plot to ' + filename + '.')
    plt.close()


def error_plot(x_vals, error_vals, labels, log_dir, name="error_over_x.png", title=None,
        vector_parts=None):
    """
    Create a plot of x location vs error.

    Intended to compare between the error of many configurations, e.g. the different sizes in a
    convergence plot, or using different agents.

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
    plt.close()

