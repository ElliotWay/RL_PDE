import os
import argparse
import pandas as pd
import numpy as np

from util import plots
from util.misc import soft_link_directories
from util.argparse import ExtendAction

WENO_COLORS = [None, None, 'g', 'b', 'r', 'y', 'c', 'm']
# Polynomial colors line up so that WENO color i is poly color 2i-1.
# Even-indexed polynomial colors are the appropriate colors between the odd-indexed colors.
POLY_COLORS = [None, 'grey', (0.4, 0.7, 0.0), 'g', 'c', 'b', 'm', 'r',
        'tab:orange', 'y', (0.375, 0.75, 0.375), 'c', (0.375, 0.375, 0.75), 'm']

class ExtendTupleAction(ExtendAction):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        values = [(self.name, value) for value in values]
        super().__call__(parser, namespace, values, option_string=None)

class ExtendTupleGen:
    def __init__(self, name):
        self.name = name
    def __call__(self, *args, **kwargs):
        return ExtendTupleAction(self.name, *args, **kwargs)

def main():
    parser = argparse.ArgumentParser(
        description="Combine data from multiple convergence plots into a single plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(type=str, nargs='*', action=ExtendTupleGen('file'), dest='curves',
            metavar='FILE',
            help="CSV files containing the relevant data. These files are in the log directory of"
            + " prior convergence plot runs.")
    parser.add_argument("--file", type=str, nargs='+', action=ExtendTupleGen('file'),
            dest='curves', metavar='FILE',
            help="More CSV files. Useful to maintain correct ordering when also using --weno."
            + " Multiple file names can also be passed with --file.")
    parser.add_argument("--weno", type=int, nargs='+', action=ExtendTupleGen('weno'), dest='curves',
            metavar='ORDER',
            help="Add the convergence curve for the given weno order. The --weno-path must"
            + " also be specified, and data must be in WENO_PATH/order_ORDER/progress.csv.")
    parser.add_argument("--weno-path", "--weno_path", type=str,
            default="test/weno_burgers/gaussian_convergence/weno",
            help="Path for data to use with the --weno argument.")
    parser.add_argument("--poly", type=int, nargs='+', action=ExtendTupleGen('poly'),
            dest='curves', metavar='ORDER',
            help="Add a curve representing exact polynomial accuracy for comparison."
            + " The curve will be near the previous curve.")
    parser.add_argument("--labels", type=str, nargs='*', default=None,
            help="Labels for each file. By default, no labels are used."
            + " Labels for WENO and polynomial curves use the default built-in labels so"
            + " should not be provided. If only WENO and polynomial curves are plotted, indicate"
            + " that labels should be added by using --labels with no arguments.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to. If an existing directory is passed,"
            + " the file will be saved to 'convergence.png' in that directory.")

    args = parser.parse_args()

    grid_sizes = []
    errors = []
    kwargs_list = []

    for index, (curve_type, curve_id) in enumerate(args.curves):
        if curve_type == "file":
            file_name = curve_id
            csv_df = pd.read_csv(file_name, comment='#')
            if 'nx' in csv_df:
                sizes = csv_df['nx']
            elif 'num_cells' in csv_df:
                sizes = csv_df['num_cells']
            else:
                raise Exception()
            error_list = csv_df['l2_error']
            kwargs_list.append({})
        elif curve_type == "weno":
            order = curve_id
            full_path = os.path.join(args.weno_path, f"order_{order}", "progress.csv")
            csv_df = pd.read_csv(full_path, comment='#')
            if 'nx' in csv_df:
                sizes = csv_df['nx']
            elif 'num_cells' in csv_df:
                sizes = csv_df['num_cells']
            else:
                raise Exception()
            error_list = csv_df['l2_error']
            if args.labels is not None:
                args.labels.insert(index, f"WENO, r={order}")
            kwargs_list.append({'color': WENO_COLORS[order], 'linestyle': '--'})
        elif curve_type == "poly":
            if len(grid_sizes) == 0:
                raise Exception("Polynomial cannot be the first curve in the plot.")
            order = curve_id
            previous_sizes = np.array(grid_sizes[-1])
            previous_errors = errors[-1]
            sizes, error_list, poly_label = plots.generate_polynomial(
                                                    order, previous_sizes, previous_errors)
            if args.labels is not None:
                args.labels.insert(index, poly_label)
            kwargs_list.append({'color': POLY_COLORS[order], 'linestyle': ':'})

        grid_sizes.append(sizes)
        errors.append(error_list)

    dir_name, file_name = os.path.split(args.output)
    if file_name == "":
        file_name = "convergence.png"
    plots.convergence_plot(grid_sizes, errors, log_dir=dir_name, name=file_name,
            labels=args.labels, kwargs_list=kwargs_list)

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")

if __name__ == "__main__":
    main()
