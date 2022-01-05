import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util import plots
from util.misc import soft_link_directories

def main():
    parser = argparse.ArgumentParser(
        description="Compare the error of one agent to the error of another agent, with one"
        + " agent's error on one axis and the other agent's error on the other."
        + " The files for the first agent must correspond to the files for the second agent"
        + " i.e. the ith file for both must be from identical environments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--x-error", "-x", type=str, nargs='+',
            help="CSV files containing L2 error data (progress.csv) for the agent"
            + " to be plotted on the x-axis.")
    parser.add_argument("--y-error", "-y", type=str, nargs='+',
            help="CSV files containing L2 error data for the agent"
            + " to be plotted on the y-axis.")
    parser.add_argument("--xname", type=str, default=None,
            help="Label for the x axis, e.g. the name of the first agent.")
    parser.add_argument("--yname", type=str, default=None,
            help="Label for the y axis, e.g. the name of the second agent.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the plot to.")
    parser.add_argument("--paper-mode", dest='paper_mode', default=True, action='store_true',
            help="Use paper style. Bigger text and specific tweaks.")
    parser.add_argument("--std-mode", dest='paper_mode', action='store_false',
            help="Use standard style. Smaller text, but generalize better to arbitrary data.")

    args = parser.parse_args()
    assert len(args.output) > 0

    if len(args.x_error) != len(args.y_error):
        raise Exception("Each file for the x-axis should correspond to a file for the y-axis,"
                + f" but there are {len(args.x_error)} for x and {len(args.y_error)} for y.")

    x_errors = []
    final_times = []
    y_errors = []

    for filename in args.x_error:
        csv_df = pd.read_csv(filename, comment='#')

        final_time = np.array(csv_df['t'])[-1]
        final_times.append(final_time)

        final_error = np.array(csv_df['l2'])[-1]
        x_errors.append(final_error)


    for index, filename in enumerate(args.y_error):
        csv_df = pd.read_csv(filename, comment='#')

        final_time = np.array(csv_df['t'])[-1]
        if not np.isclose(final_time, final_times[index]):
            raise Exception("Time of final timestep does not match between"
                    + f" {args.x_error[index]} and {filename}"
                    + f" (final_times[index] vs final_time).")

        final_error = np.array(csv_df['l2'])[-1]
        y_errors.append(final_error)

    if args.paper_mode:
        plt.rcParams.update({'font.size':15})

    fig = plt.figure(figsize=(4.8, 4.8))
    ax = fig.gca()
    ax.scatter(x_errors, y_errors, marker='.', color='grey')
    ax = plt.gca()
    ax.set_xscale('log')
    if args.xname is not None:
        ax.set_xlabel(args.xname)
    ax.set_yscale('log')
    if args.yname is not None:
        ax.set_ylabel(args.yname)
    if args.title is not None:
        ax.set_title(args.title)

    # Add a diagonal line.
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    limits = [min(xmin, ymin), max(xmax, ymax)]
    ax.plot(limits, limits, color='k', linestyle='-', alpha=0.75)
    ax.set_aspect('equal')
    ax.set_xlim(limits)
    ax.set_ylim(limits)

    plt.tight_layout()

    dir_name, file_name = os.path.split(args.output)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    plt.savefig(args.output)
    print(f"Saved plot to {args.output}.")

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
