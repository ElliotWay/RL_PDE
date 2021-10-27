import os
import argparse
import re

import numpy as np
import pandas as pd

from util import plots
from util.misc import soft_link_directories

def main():
    parser = argparse.ArgumentParser(
        description="Combine data from a directory of multiple convergence runs into a standard"
        + " convergence plot that compares solutions across different orders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("directory", type=str,
            help="Directory where sub-directories contain convergence runs."
            + " Sub-directories should have names formatted like 'order_3/rl' OR 'rl/order_3'.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to. If an existing directory is passed,"
            + " the file will be saved to 'convergence.png' in that directory.")

    SOLUTION_NAMES = ['rl', 'weno']
    ORDER_FORMAT = "order_(\d+)"

    args = parser.parse_args()
    outer_dir = args.directory

    files = {}
    
    order_sub_dirs = {}
    solution_sub_dirs = {}
    for entry in os.scandir(outer_dir):
        if entry.is_dir():
            if entry.name in SOLUTION_NAMES:
                solution_sub_dirs[entry.name] = entry.path
            else:
                match = re.fullmatch(ORDER_FORMAT, entry.name)
                if match is not None:
                    order_sub_dirs[int(match.group(1))] = entry.path

    if len(order_sub_dirs) > 0 and len(solution_sub_dirs) > 0:
        raise Exception("Directory contains both orders and solutions?")
    elif len(order_sub_dirs) == 0 and len(solution_sub_dirs) == 0:
        raise Exception("Directory does not contain orders or solutions.")
    elif len(order_sub_dirs) > 0:
        for order, order_path in order_sub_dirs.items():
            files[order] = {}
            for entry in os.scandir(order_path):
                if entry.is_dir() and entry.name in SOLUTION_NAMES:
                    csv_file = os.path.join(entry.path, "progress.csv")
                    files[order][entry.name] = csv_file
    else:
        assert len(solution_sub_dirs) > 0
        for solution_name, sol_path in solution_sub_dirs.items():
            for entry in os.scandir(sol_path):
                if entry.is_dir():
                    match = re.fullmatch(ORDER_FORMAT, entry.name)
                    if match is not None:
                        order = int(match.group(1))
                        csv_file = os.path.join(entry.path, "progress.csv")
                        if order in files:
                            files[order][solution_name] = csv_file
                        else:
                            files[order] = {solution_name: csv_file}


    colors = [None, None, 'g', 'b', 'r', 'y', 'c', 'm']
        
    grid_sizes = []
    errors = []
    labels = []
    kwargs_list = []

    for order in sorted(files.keys()):
        sub_dict = files[order]

        poly_added = False

        for sol_name in ['rl', 'weno']:
            if not sol_name in sub_dict:
                continue

            file_name = sub_dict[sol_name]
            csv_df = pd.read_csv(file_name, comment='#')

            if 'nx' in csv_df:
                sizes = np.array(csv_df['nx'])
            elif 'num_cells' in csv_df:
                sizes = np.array(csv_df['num_cells'])
            else:
                raise Exception()
            error_list = np.array(csv_df['l2_error'])

            # Add polynomial order curve. It goes first for this order.
            if not poly_added:
                comparison_error = error_list
                poly_order = order * 2 - 1
                poly_values, poly_label = plots.generate_polynomial(poly_order, sizes, error_list)

                grid_sizes.append(sizes)
                errors.append(poly_values)
                labels.append(poly_label)
                kwargs_list.append({'color': colors[order], 'linestyle': ':'})

                poly_added = True

            if sol_name == 'rl':
                labels.append(f"RL, r={order}")
                kwargs_list.append({'color': colors[order], 'linestyle': '-'})
            else:
                labels.append(f"WENO, r={order}")
                kwargs_list.append({'color': colors[order], 'linestyle': '--'})

            grid_sizes.append(sizes)
            errors.append(error_list)

    dir_name, file_name = os.path.split(args.output)
    if file_name == "":
        file_name = "convergence.png"
    plots.convergence_plot(grid_sizes, errors, log_dir=dir_name, name=file_name, labels=labels,
            kwargs_list=kwargs_list)

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
