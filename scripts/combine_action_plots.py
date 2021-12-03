import os
import argparse
import re
import numpy as np
import pandas as pd

from util import plots
import util.colors as colors
from util.misc import soft_link_directories

def main():
    parser = argparse.ArgumentParser(
        description="Combine data from earlier runs into a single plot of actions. (1D only.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("files", type=str, nargs='+',
            help="CSV file containing action data. Use --output-mode csv"
            + " to create these files.")
    parser.add_argument("--labels", type=str, nargs='+', required=True,
            help="Labels for each action file.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to. If an existing directory is passed,"
            + " the plot will be saved to 'actions.png' in that directory.")
    parser.add_argument("--no-default", default=False, action='store_true',
            help="Use the original styles instead of the paper\n"
            +    "unified styles.")

    args = parser.parse_args()
    assert len(args.output) > 0

    create_legend = (args.labels is not None)
    if args.labels is None:
        args.labels = [""] * len(args.files)
    else:
        if len(args.labels) != len(args.files):
            raise Exception(f"Number of labels ({len(args.labels)}) must match"
                    + f" number of curves to plot ({len(args.files)}).")

    x_locations = []
    actions = []
    kwargs_list = []
    good_file = None
    column_names = None

    x_dimension = None
    vector_parts = None
    action_parts = None

    for input_file, label in zip(args.files, args.labels):

        csv_df = pd.read_csv(input_file, comment='#')
        names = list(csv_df)

        if column_names is None:
            column_names = names
            good_file = input_file

            x_dimension = names[0]

            # Column names are vector@action, see Plottable1DEnv.save_action().
            vector_parts = []
            action_parts = []
            for name in names[1:]:
                match = re.fullmatch("([^@]+)@([^@]+)", name)
                if match is None:
                    raise Exception(f"Action column name \"{name}\" corrupted.")
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
                                + f" {name} is missing in {input_file}.")

        elif column_names != names:
            raise Exception("Column names do not match between files."
                    + f" {good_file} has {column_names}; {input_file} has {names}")

        x_locations.append(csv_df[x_dimension])

        vector_actions = []
        for vector_part in vector_parts:
            sub_actions = []
            for action_part in action_parts:
                name = f"{vector_part}@{action_part}"
                sub_actions.append(csv_df[name])
            vector_actions.append(sub_actions)
        actions.append(vector_actions)

        if args.no_default:
            kwargs = {}
        else:
            kwargs = colors.get_agent_kwargs(input_file, label, just_color=True)
        kwargs_list.append(kwargs)

    dir_name, file_name = os.path.split(args.output)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    if file_name == "":
        file_name = "actions.png"
    plots.action_plot(x_locations, actions, x_label=x_dimension, labels=args.labels,
            log_dir=dir_name, name=file_name, title=args.title, kwargs_list=kwargs_list,
            vector_parts=vector_parts, action_parts=action_parts)

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
