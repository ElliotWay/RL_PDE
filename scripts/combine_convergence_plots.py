import os
import argparse
import pandas as pd

from util import plots
from util.misc import soft_link_directories
from util.argparse import ExtendAction

WENO_COLORS = [None, None, 'g', 'b', 'r', 'y', 'c', 'm']

def main():
    parser = argparse.ArgumentParser(
        description="Combine data from multiple convergence plots into a single plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", type=str, nargs='*',
            help="CSV files containing the relevant data. These files are in the log directory of"
            + " prior convergence plot runs.")
    parser.add_argument("--file", type=str, action=ExtendAction, dest='files',
            help="More CSV files. Useful to maintain correct ordering when also using --weno."
            + " Multiple file names can also be passed with --file.")
    parser.add_argument("--weno", type=int, action=ExtendAction, dest='files', metavar='ORDER',
            help="Add the convergence curve for the given weno order. The --weno-path must"
            + " also be specified, and data must be in WENO_PATH/order_ORDER/progress.csv.")
    parser.add_argument("--weno-path", "--weno_path", type=str,
            default="test/weno_burgers/gaussian_convergence/weno",
            help="Path for data to use with the --weno argument.")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
            help="Labels for each file. By default, no labels are used.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to. If an existing directory is passed,"
            + " the file will be saved to 'convergence.png' in that directory.")

    args = parser.parse_args()

    grid_sizes = []
    errors = []
    kwargs_list = []

    for index, input_file in enumerate(args.files):
        if type(input_file) is str:
            csv_df = pd.read_csv(input_file, comment='#')
            if 'nx' in csv_df:
                sizes = csv_df['nx']
            elif 'num_cells' in csv_df:
                sizes = csv_df['num_cells']
            else:
                raise Exception()
            error_list = csv_df['l2_error']
            kwargs_list.append({})
        else:
            assert type(input_file) is int
            order = input_file
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
