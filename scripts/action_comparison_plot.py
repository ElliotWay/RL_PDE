import os
import argparse
import re
import numpy as np
import pandas as pd

from util import plots
from util.misc import soft_link_directories

def main():
    parser = argparse.ArgumentParser(
        description="Compare actions between 2 runs on one plot with one agent's actions on"
        + " one axis and the other agent's actions on the other. The files for the second axis"
        + " must correspond to the files for the first axis, e.g. files for the same timestep"
        + " but different agents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--first-actions", "-a1", type=str, nargs='+',
            help="CSV files containing action data for x-axis."
            + " Use --output-mode csv to create these files.")
    parser.add_argument("--second-actions", "-a2", type=str, nargs='+',
            help="CSV files containing action data for y-axis.")
    parser.add_argument("--ndims", type=int, default=1,
            help="The first ndims columns of each file are assumed to be spatial"
            + " coordinates and are ignored for this script.")
    parser.add_argument("--xname", type=str, default=None,
            help="Label for the x axis, e.g. the name of the first agent.")
    parser.add_argument("--yname", type=str, default=None,
            help="Label for the y axis, e.g. the name of the second agent.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the plot to.")

    args = parser.parse_args()
    assert len(args.output) > 0

    x_actions = []
    y_actions = []

    column_names = None
    good_file = None
    spatial_dims = None
    vector_parts = None
    action_parts = None

    for filename in args.first_actions:
        csv_df = pd.read_csv(filename, comment='#')
        names = list(csv_df)

        if column_names is None:
            column_names = names
            good_file = filename

            spatial_dims = np.array([csv_df[spatial_name] for spatial_name in names[:args.ndims]])

            # Column names are vector@action, see Plottable1DEnv.save_action().
            vector_parts = []
            action_parts = []
            for name in names[args.ndims:]:
                match = re.fullmatch("([^@]+)@([^@]+)", name)
                if match is None:
                    raise Exception(f"Action column name \"{name}\" corrupted."
                            + " (Is this action data?)")
                vector_part = match.group(1)
                action_part = match.group(2)
                if vector_part not in vector_parts:
                    vector_parts.append(vector_part)
                if action_part not in action_parts:
                    action_parts.append(action_part)
            for vector_part in vector_parts:
                for action_part in action_parts:
                    name = f"{vector_part}@{action_part}"
                    if name not in names:
                        raise Exception("Each vector component must have the same action parts."
                                + f" {name} is missing in {filename}.")

        elif column_names != names:
            raise Exception("Column names must match between files."
                    + f" {good_file} has {column_names}; {filename} has {names}")
        else:
            new_spatial_dims = np.array([csv_df[spatial_name]
                                for spatial_name in names[:args.ndims]])
            if not np.allclose(spatial_dims, new_spatial_dims):
                raise Exception("Spatial dims must match between files with the same order."
                        + f" {good_file} and {filename} have different values.")

        vector_actions = []
        for vector_part in vector_parts:
            sub_actions = []
            for action_part in action_parts:
                name = f"{vector_part}@{action_part}"
                sub_actions.append(csv_df[name])
            vector_actions.append(sub_actions)
        x_actions.append(vector_actions)

    for filename in args.second_actions:
        csv_df = pd.read_csv(filename, comment='#')
        names = list(csv_df)

        if column_names != names:
            raise Exception("Column names must match between files."
                    + f" {good_file} has {column_names}; {filename} has {names}")
        else:
            new_spatial_dims = np.array([csv_df[spatial_name]
                                for spatial_name in names[:args.ndims]])
            if not np.allclose(spatial_dims, new_spatial_dims):
                raise Exception("Spatial dims must match between files with the same order."
                        + f" {good_file} and {filename} have different values.")

        vector_actions = []
        for vector_part in vector_parts:
            sub_actions = []
            for action_part in action_parts:
                name = f"{vector_part}@{action_part}"
                sub_actions.append(csv_df[name])
            vector_actions.append(sub_actions)
        y_actions.append(vector_actions)

    # Current shape is [file, vector, action_part, location], but the plot function expects
    # [vector, action_part] as the first two axes.
    x_actions = np.transpose(x_actions, (1, 2, 0, 3))
    y_actions = np.transpose(y_actions, (1, 2, 0, 3))

    dir_name, file_name = os.path.split(args.output)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    plots.action_comparison_plot(x_actions, y_actions, args.xname, args.yname,
            log_dir=dir_name, name=file_name, title=args.title,
            vector_parts=vector_parts, action_parts=action_parts)

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
