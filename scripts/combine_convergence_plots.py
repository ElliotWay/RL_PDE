import os
import argparse
import pandas as pd

from util import plots

def main():
    parser = argparse.ArgumentParser(
        description="Combine data from multiple convergence plots into a single plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", type=str, nargs='+',
            help="CSV files containing the relevant data. These files are in the log directory of"
            + " prior convergence plot runs.")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
            help="Labels for each file. By default, no labels are used.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to. If an existing directory is passed,"
            + " the file will be saved to 'convergence.png' in that directory.")

    args = parser.parse_args()

    grid_sizes = []
    errors = []

    for input_file in args.files:
        csv_df = pd.read_csv(input_file, comment='#')
        if 'nx' in csv_df:
            sizes = csv_df['nx']
        elif 'num_cells' in csv_df:
            sizes = csv_df['num_cells']
        else:
            raise Exception()
        grid_sizes.append(sizes)

        error_list = csv_df['l2_error']
        errors.append(error_list)

    dir_name, file_name = os.path.split(args.output)
    if file_name == "":
        file_name = "convergence.png"
    plots.convergence_plot(grid_sizes, errors, log_dir=dir_name, name=file_name, labels=args.labels)

if __name__ == "__main__":
    main()
