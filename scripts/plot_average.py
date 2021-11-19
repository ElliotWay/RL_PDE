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
        description="Plot the average of data from multiple CSV plots including a shaded"
        + " confidence interval region.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('files', type=str, nargs='+',
            help="CSV files containing data.")
    parser.add_argument('--index-name', type=str, required=True,
            help="Name of the column containing values for the x-axis."
            + " Must be the same data for every file.")
    parser.add_argument('--value-name', type=str, required=True,
            help="Name of the column containing the values for the y-axis.")
    parser.add_argument('--ci-type', type=str, default='range',
            help = "The type of confidence interval to plot. Options are:\n"
                + " range: [min,max]\n"
                + " Xconf: [P(lower)=(1-X)/2,P(higher)=(1-X)/2] (T dist), X in [0,1]\n"
                + " Xsig: [-X std deviations,+X std deviations] (normal dist), X > 0\n"
                + " Nperc: [Nth percentile,100-Nth percentile], N in [0, 50]\n")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the generated plot to.")

    args = parser.parse_args()

    first_df = pd.read_csv(args.files[0], comment='#')
    x_values = first_df[args.index_name]
    y_data = [first_df[args.value_name]]

    for filename in args.files[1:]:
        csv_df = pd.read_csv(filename, comment='#')

        new_x = csv_df[args.index_name]
        if not all(new_x == x_values):
            raise Exception(f"{index_name} values do not match between"
                    + f" {filename} and {args.files[0]}.")
        y_data.append(csv_df[args.value_name])

    ax = plt.gca()
    plots.add_average_with_ci(ax, x_values, y_data, ci_type=args.ci_type)

    if args.title is not None:
        ax.set_title(title)

    plt.tight_layout()

    plt.savefig(args.output)
    print(f"Saved plot to {args.output}.")

    dir_name, file_name = os.path.split(args.output)

    # Create symlink for convenience, unless the file is in the current directory.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
