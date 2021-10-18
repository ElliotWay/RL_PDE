import os
import argparse
import re
import numpy as np
import pandas as pd

from util import plots

def main():
    parser = argparse.ArgumentParser(
        description="Combine data from earlier runs into a single plot of error vs x. (1D only.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", type=str, nargs='*',
            help="CSV files containing error data. Use --output-mode csv"
            + " to create these files.")
    parser.add_argument("--diff", type=str, nargs='*',
            help="CSV files containing state data, where the error is computed"
            + " as the absolute difference between each pair.")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
            help="Labels for each file. By default, no labels are used.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to. If an existing directory is passed,"
            + " the plot will be saved to 'errors.png' in that directory.")

    args = parser.parse_args()

    if len(args.diff) % 2 == 1:
        raise Exception("--diff files must come in pairs, but there are"
                + f" {len(args.diff)} of them.")

    x_locations = []
    errors = []
    column_names = None

    def sanitize_name(name):
        match = re.fullmatch("error[_\-\ ]?(.+)", name)
        if match:
            return match.group(1)
        match = re.fullmatch("([^_\-\ ]+)[\_\- ]?error", name)
        if match:
            return match.group(1)
        return name

    for input_file in args.files:
        csv_df = pd.read_csv(input_file, comment='#')
        names = [name for name in list(csv_df) if name != 'x']
        sanitized_names = [sanitize_name(name) for name in names]
        if column_names is None:
            column_names = sanitized_names
        elif column_names != sanitized_names:
            raise Exception("Column names do not match between files."
                    + f" {args.files[0]} has {column_names}; {input_file} has {sanitized_names}")

        x_locations.append(csv_df['x'])
        errors.append([csv_df[name] for name in column_names])

    for (left_file, right_file) in zip(args.diff[::2], args.diff[1::2]):
        left_csv_df = pd.read_csv(left_file, comment='#')
        left_names = [name for name in list(left_csv_df) if name != 'x']
        sanitized_names = [sanitize_name(name) for name in left_names]
        if column_names is None:
            column_names = sanitized_names
        elif column_names != sanitized_names:
            original_file = args.diff[0] if len(args.files) == 0 else args.files[0]
            raise Exception("Column names do not match between files."
                    + f" {original_file} has {column_names}; {left_file} has {sanitized_names}")

        right_csv_df = pd.read_csv(right_file, comment='#')
        right_names = [name for name in list(right_csv_df) if name != 'x']
        sanitized_names = [sanitize_name(name) for name in right_names]
        if column_names != sanitized_names:
            raise Exception("Column names do not match between files."
                    + f" {left_file} has {column_names}; {right_file} has {sanitized_names}")

        left_x = left_csv_df['x']
        right_x = right_csv_df['x']
        if (left_x != right_x).any():
            raise Exception(f"Diff files {left_file} and {right_file} do not have"
                    + " matching x values.")
        x_locations.append(left_x)
        errors.append([np.abs(left_csv_df[left_name] - right_csv_df[right_name])
                        for left_name, right_name in zip(left_names, right_names)])

    dir_name, file_name = os.path.split(args.output)
    if file_name == "":
        file_name = "errors.png"
    plots.error_plot(x_locations, errors, labels=args.labels, log_dir=dir_name, name=file_name,
            vector_parts=column_names)

if __name__ == "__main__":
    main()
