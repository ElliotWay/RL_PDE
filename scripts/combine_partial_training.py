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
from util.argparse import ExtendAction
import util.colors as colors

def main():
    parser = argparse.ArgumentParser(
        description="""
Combine training data into a single plot. --files and --avg input for consistency, but you can only
pass two.
Index column is assumed episodes. Value column of the first is avg_train_{total_reward,end_l2} and
value column of the second is avg_eval_{total_reward,end_l2}.
""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(type=str, nargs='*', action=ExtendAction, dest='curves',
            metavar='FILE',
            help="CSV files containing time data. (progress.csv)")
    parser.add_argument('--files', type=str, nargs='+', action=ExtendAction, dest='curves',
            metavar='FILE',
            help="More CSV files containing data.")
    parser.add_argument('--avg', type=str, nargs='+', action='append', dest='curves',
            metavar='FILE',
            help="CSV files for which the mean with a confidence\ninterval will be plotted.")
    parser.add_argument('--type', type=str, default='reward',
            help="reward or l2")
    parser.add_argument('--ci-type', type=str, default='range', help="""\
The type of confidence interval to plot.
Options are: (default: range)
  range: [min,max]
  Xconf: double-tailed CI (T dist), X in [0,1]
  Xsig: [-X stddevs,+X stddevs] (normal dist), X > 0
  Nperc: [Nth percentile,100-Nth], N in [0, 50]""")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
            help="Labels for each file/average.")
    parser.add_argument("--xlabel", type=str, default='episodes',
            help="Name of the x axis. Defaults to the name of the\ncolumn with the data.")
    parser.add_argument("--ylabel", type=str, default=None,
            help="Name of the y axis. Defaults to the name of the\ncolumn with the data.")
    parser.add_argument("--yscale", type=str, default='linear',
            help="Scale of the y axis, e.g. linear vs log.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is\nadded.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the generated plot to.")

    args = parser.parse_args()
    assert len(args.output) > 0

    x_values = []
    y_values = []

    index = "episodes"

    assert len(args.curves) == 2, "Need only train and eval curves."

    curve_name = ["train", "avg eval"]
    if args.labels is None:
        args.labels = curve_name
    
    for curve, name in zip(args.curves, curve_name):
        if isinstance(curve, str):
            is_list = False
        else:
            is_list = True

        if name == "train":
            value = "avg_train_"
        else:
            value = "avg_eval_"

        if args.type == "reward":
            value += "total_reward"
        else:
            assert args.type == "l2"
            value += "end_l2"

        if not is_list:
            filename = curve
            df = pd.read_csv(filename, comment='#')
            x_values.append(df[index])
            y_values.append(df[value])
        else:
            filenames = curve
            dfs = [pd.read_csv(name, comment='#') for name in filenames]

            first_x = None
            for name, df in zip(filenames, dfs):
                x_data = df[index]
                if first_x is None:
                    first_x = x_data
                elif any(x_data != first_x):
                    raise Exception(f"{args.index} columns in {filenames[0]} and {name}"
                            + " do not match.")
            x_values.append(first_x)
            y_values.append([df[value] for df in dfs])

    plt.rcParams.update({'font.size':15})
    kwargs_list=[{'color':colors.TRAIN_COLOR}, {'color':colors.AVG_EVAL_COLOR} ]

    fig = plots.create_avg_plot(x_values, y_values,
            labels=args.labels, kwargs_list=kwargs_list,
            ci_type=args.ci_type)

    if args.xlabel is None:
        args.xlabel = args.index
    if args.ylabel is None:
        if args.type == "reward":
            args.ylabel = "reward"
        else:
            args.ylabel = "l2"

    ax = fig.gca()
    ax.grid(True)
    ax.set_xlabel(args.xlabel)
    ax.set_xmargin(0.0)
    ax.set_ylabel(args.ylabel)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_yscale(args.yscale)
    if args.title is not None:
        ax.set_title(args.title)
    if args.labels is not None:
        bbta=(0.5, 1.11)
        ax.legend(loc='upper center', bbox_to_anchor=bbta,
                ncol=2, fancybox=True, shadow=True,
                prop={'size': 'medium'})
    plt.tight_layout()

    dir_name, file_name = os.path.split(args.output)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

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
