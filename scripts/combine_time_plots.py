import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import plots
from util.misc import soft_link_directories
from util.argparse import ExtendAction

def main():
    parser = argparse.ArgumentParser(
        description="Combine data plotted vs time from earlier test runs into a single plot."
        + "\nThe CSV files must have a column labeled 't' or 'time', and at least one"
        + "\nother column. The order of arguments controls the order of the legend.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(type=str, nargs='*', action=ExtendAction, dest='curves',
            metavar='FILE',
            help="CSV files containing time data. (progress.csv)")
    parser.add_argument('--files', type=str, nargs='+', action=ExtendAction, dest='curves',
            metavar='FILE',
            help="More CSV files containing training time data.")
    parser.add_argument('--avg', type=str, nargs='+', action='append', dest='curves',
            metavar='FILE',
            help="CSV files for which the mean with a confidence interval will be plotted.")
    parser.add_argument('--ci-type', type=str, default='range',
            help = "The type of confidence interval to plot. (default: range)\n"
                + " Options are:\n"
                + "   range: [min,max]\n"
                + "   Xconf: [P(lower)=(1-X)/2,P(higher)=(1-X)/2] (T dist), X in [0,1]\n"
                + "   Xsig: [-X std deviations,+X std deviations] (normal dist), X > 0\n"
                + "   Nperc: [Nth percentile,100-Nth percentile], N in [0, 50]\n")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
            help="Labels for each file/average.")
    parser.add_argument("--ycol", type=str, default=None,
            help="Name of the column containing data for the y-axis. Can be omitted."
            + "\nif the files have only one non-time column.")
    parser.add_argument("--ylabel", type=str, default=None,
            help="Name of the y axis. Defaults to the name of the column with the data.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to.")

    args = parser.parse_args()

    if args.labels is not None:
        if len(args.labels) != len(args.curves):
            raise Exception(f"Number of labels ({len(args.labels)}) must match"
                    + f" number of curves to plot ({len(args.curves)}).")

    time_values = []
    y_values = []

    for curve in args.curves:
        if isinstance(curve, str):
            is_list = False
        else:
            is_list = True

        if not is_list:
            filename = curve
        
            csv_df = pd.read_csv(filename, comment='#')
            if not 'time' in csv_df and not 't' in csv_df:
                raise Exception(f"{filename} has no time data.")
            if 'time' in csv_df:
                time_values.append(csv_df['time'])
            else:
                time_values.append(csv_df['t'])

            if args.ycol is None:
                other_column_names = [name for name in csv_df if name not in ['time', 't']]
                if len(other_column_names) == 0:
                    raise Exception(f"{filename} has no data to plot against time.")
                if len(other_column_names) > 1:
                    raise Exception(f"Multiple columns in {filename}, specify with --ycol.")
                else:
                    args.ycol = other_column_names[0]
            elif args.ycol not in csv_df:
                raise Exception(f"Cannot find '{args.ycol}' in {filename}.")
            y_values.append(csv_df[args.ycol])

        else:
            filenames = curve
            dfs = [pd.read_csv(name, comment='#') for name in filenames]
            first_time = None
            for name, df in zip(filenames, dfs):
                if not 'time' in df and not 't' in df:
                    raise Exception(f"{name} has no time data.")
                time_data = df['time' if 'time' in df else 't']
                if first_time is None:
                    first_time = time_data
                elif any(first_time != time_data):
                    raise Exception(f"Times in {filenames[0]} do not match times in {name}.")
            time_values.append(first_time)

            if args.ycol is None:
                other_column_names = [name for name in dfs[0] if name not in ['time', 't']]
                if len(other_column_names) == 0:
                    raise Exception(f"{filename} has no data to plot against time.")
                if len(other_column_names) > 1:
                    raise Exception(f"Multiple columns in {filename}, specify with --ycol.")
                else:
                    args.ycol = other_column_names[0]
            for name, df in zip(filenames, dfs):
                if args.ycol not in df:
                    raise Exception(f"Cannot find '{args.ycol}' in {name}.")
            y_values.append([df[args.ycol] for df in dfs])

    if args.ylabel is None:
        args.ylabel = args.ycol

    fig = plots.create_avg_plot(time_values, y_values,
            labels=args.labels, kwargs_list=None, # Use default matplotlib colors.
            ci_type=args.ci_type)
    if args.labels is not None:
        fig.legend(prop={'size': plots.legend_font_size(len(y_values))})
    ax = fig.gca()
    ax.set_xlabel("time")
    ax.set_xmargin(0.0)
    ax.set_ylabel(args.ylabel)
    #ax.set_yscale('log') # Add an arg to the parser for this if we need it.
    if args.title is not None:
        ax.set_title(args.title)
    plt.tight_layout()

    plt.savefig(args.output)
    print(f"Saved plot to {args.output}.")
    plt.close()

    # Create symlink for convenience.
    dir_name, file_name = os.path.split(args.output)
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
