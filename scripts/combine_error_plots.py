import os
import argparse
import re
import numpy as np
import pandas as pd

from util import plots
from util.misc import soft_link_directories

def main():
    parser = argparse.ArgumentParser(
        description="Combine data from earlier runs into a single plot of error vs x. (1D only.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(type=str, nargs='*', action=ExtendAction, dest='files',
            metavar='FILE',
            help="CSV files containing error data. Use --output-mode csv\n"
            +    "to create these files.")
    parser.add_argument("--files", type=str, nargs='+', action=ExtendAction, dest='files',
            metavar='FILE',
            help="CSV file containing error data. Use --output-mode csv"
            + " to create these files. Specify multiple files with --file FILE1 --file FILE2.")
    parser.add_argument("--diff", type=str, nargs=2, action='append', dest='files',
            metavar=('FILEA', 'FILEB'),
            help="CSV files containing state data, where the error is computed"
            + " as the absolute difference between each pair. Specify multiple pairs"
            + " with --diff FILEA1 FILEB1 --diff FILEA2 FILEB2.")
    parser.add_argument("--labels", type=str, nargs='+', required=True,
            help="Labels for each error.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to. If an existing directory is passed,"
            + " the plot will be saved to 'errors.png' in that directory.")

    args = parser.parse_args()
    assert len(args.output) > 0

    x_locations = []
    errors = []
    good_file = None
    column_names = None

    # Names for state variables are things like 'u' or 'rho'.
    # Names for error variables are things like 'u_error' or 'error-rho'.
    def sanitize_error_name(name):
        match = re.fullmatch("error[_\-\ ]?(.+)", name)
        if match:
            return match.group(1)
        match = re.fullmatch("([^_\-\ ]+)[\_\- ]?error", name)
        if match:
            return match.group(1)
        return name

    for input_file in args.files:
        if type(input_file) is str:
            csv_df = pd.read_csv(input_file, comment='#')
            names = [name for name in list(csv_df) if name != 'x']
            sanitized_names = [sanitize_error_name(name) for name in names]
            if column_names is None:
                column_names = sanitized_names
                good_file = input_file
            elif column_names != sanitized_names:
                raise Exception("Column names do not match between files."
                        + f" {good_file} has {column_names}; {input_file} has {sanitized_names}")

            x_locations.append(csv_df['x'])
            errors.append([csv_df[name] for name in names])
        else:
            left_file, right_file = input_file
            left_csv_df = pd.read_csv(left_file, comment='#')
            left_names = [name for name in list(left_csv_df) if name != 'x']
            if column_names is None:
                column_names = left_names
                good_file = left_file
            elif column_names != left_names:
                raise Exception("Column names do not match between files."
                        + f" {good_file} has {column_names}; {left_file} has {left_names}")

            right_csv_df = pd.read_csv(right_file, comment='#')
            right_names = [name for name in list(right_csv_df) if name != 'x']
            if column_names != right_names:
                raise Exception("Column names do not match between files."
                        + f" {good_file} has {column_names}; {right_file} has {right_names}")

            left_x = left_csv_df['x']
            right_x = right_csv_df['x']
            if (left_x != right_x).any():
                raise Exception(f"Diff files {left_file} and {right_file} do not have"
                        + " matching x values.")
            x_locations.append(left_x)
            errors.append([np.abs(left_csv_df[left_name] - right_csv_df[right_name])
                            for left_name, right_name in zip(left_names, right_names)])

    dir_name, file_name = os.path.split(args.output)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    if file_name == "":
        file_name = "errors.png"
    plots.error_plot(x_locations, errors, labels=args.labels, log_dir=dir_name, name=file_name,
            vector_parts=column_names, title=args.title)

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")



if __name__ == "__main__":
    main()
