import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from util import plots
from util import colors
from util.misc import soft_link_directories
from util.argparse import ExtendAction

def main():
    parser = argparse.ArgumentParser(
        description="Combine data from earlier runs into a single plot of state vs x. (1D only.)"
        + " Creates multiple plots if the state is a vector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(type=str, nargs='*', action=ExtendAction, dest='files',
            metavar='FILE',
            help="CSV files containing state data. Use --output-mode csv\n"
            +    "to create these files.")
    parser.add_argument("--files", type=str, nargs='+', action=ExtendAction, dest='files',
            metavar='FILE',
            help="More CSV files containing state data.")
    parser.add_argument("--labels", type=str, nargs='+',
            help="Labels for each error.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to. If state is a vector,"
            + "the component names will be inserted into the name e.g."
            + "state.png -> [state_rho.png, state_u.png, ...]")
    parser.add_argument("--no-default", default=False, action='store_true',
            help="Use matplotlib defaults styles instead of the styles in"
            + "util/colors.py")

    args = parser.parse_args()
    assert len(args.output) > 0

    create_legend = (args.labels is not None)
    if args.labels is None:
        args.labels = [""] * len(args.files)
    else:
        if len(args.labels) != len(args.curves):
            raise Exception(f"Number of labels ({len(args.labels)}) must match"
                    + f" number of curves to plot ({len(args.curves)}).")

    first_df = pd.read_csv(args.files[0], comment='#')
    if 'x' not in first_df:
        raise Exception(f"x column not found in {args.files[0]}; is this state data?")
    if 'y' in first_df:
        print(f"Warning: y column found in {args.files[0]}. This script can only"
                + " handle 1D data; assuming y is actually a state component.")
    component_names = [name for name in first_df if name != 'x']
    if len(component_names) > 1:
        print("Detected vector with components " + ", ".join(component_names))

    x_data = []
    state_data = {name:[] for name in component_names}

    for input_file in args.files:
        csv_df = pd.read_csv(input_file, comment='#')
        x_data.append(csv_df['x'])
        for name in component_names:
            state_data[name].append(csv_df[name])

    dir_name, output_file_name = os.path.split(args.output)
    file_short, file_ext = os.path.splitext(output_file_name)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    for name in component_names:
        fig, ax = plt.subplots()
        for x, state, label, filename in zip(x_data, state_data[name], args.labels, args.files):
            if args.no_default:
                kwargs = {}
            else:
                kwargs = colors.get_agent_kwargs(filename, label)
            ax.plot(x, state, label=label, **kwargs)

        ax.set_xlabel('$x$')
        ax.set_xmargin(0.0)
        if '$' not in name:
            if name == 'rho':
                display_name = r'\rho'
            else:
                display_name = name
            ax.set_ylabel(f"${display_name}$")
        else:
            ax.set_ylabel(name)

        if create_legend:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08),
                    ncol=len(x_data), fancybox=True, shadow=True,
                    prop={'size': plots.legend_font_size(len(x_data))})
        if args.title is not None:
            if create_legend:
                ax.set_title(args.title, pad=24.0)
            else:
                ax.set_title(args.title)
        plt.tight_layout()

        if len(component_names) == 1:
            file_name = output_file_name
        else:
            file_name = f"{file_short}_{name}{file_ext}"
        full_name = os.path.join(dir_name, file_name)
        fig.savefig(full_name)
        print(f"Saved plot to {full_name}.")
        plt.close(fig)

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
