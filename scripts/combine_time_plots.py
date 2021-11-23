import os
import argparse
import re
import numpy as np
import pandas as pd

from util import plots
from util.misc import soft_link_directories

def main():
    parser = argparse.ArgumentParser(
        description="Combine data plotted vs time from earlier test runs into a single plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('files', type=str, nargs='+',
            help="CSV file containing data. It should have a column labeled 't' or 'time',"
            + " and exactly one other column (with any label).")
            #TODO: possibly extend to handle multiple columns in a sensible way.
    parser.add_argument("--labels", type=str, nargs='+', required=True,
            help="Labels for the values from each file.")
    parser.add_argument("--ylabel", type=str, default=None,
            help="Name of the y axis. By default, the name of the non-time column"
            + " of the first file.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to.")

    args = parser.parse_args()

    if args.labels is not None:
        if len(args.labels) != len(args.files):
            raise Exception(f"Number of labels ({len(args.labels)}) must match"
                    + f" number of files ({len(args.files)}).")

    time_values = []
    y_values = []

    for filename in args.files:
        csv_df = pd.read_csv(filename, comment='#')
        if not 'time' in csv_df and not 't' in csv_df:
            raise Exception(f"{filename} has no time data.")
        if 'time' in csv_df:
            time_values.append(csv_df['time'])
        else:
            time_values.append(csv_df['t'])

        other_column_names = [name for name in csv_df if name not in ['time', 't']]
        if len(other_column_names) == 0:
            raise Exception(f"{filename} has no data to plot against time.")
        if len(other_column_names) > 1:
            raise Exception(f"{filename} should only have one other column.")

        value_name = other_column_names[0]
        y_values.append(csv_df[value_name])
        if args.ylabel is None:
            args.ylabel = value_name

    dir_name, file_name = os.path.split(args.output)
    plots.plot_over_time(time_values, y_values, log_dir=dir_name, name=file_name,
            ylabel=args.ylabel, labels=args.labels, title=args.title)

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
