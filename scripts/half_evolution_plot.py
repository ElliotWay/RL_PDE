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

def main():
    parser = argparse.ArgumentParser(
        description="Create an evolution plot with lines for the initial condition,"
        + " halfway through the episode, and at the end of the episode."
        + " This script is specific to this purpose and does not generalize.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--init", type=str, required=True,
            help="File with the state at t=0.")
    parser.add_argument("--rl", type=str, nargs=2, default=None,
            help="Files with the RL state at half and full times. Pass two files.")
    parser.add_argument("--weno", type=str, nargs=2, default=None,
            help="Files with the WENO state at half and full times. Pass two files.")
    parser.add_argument("--true", type=str, nargs=2, default=None,
            help="Files with the analytical state at half and full times. Pass two files.")
    parser.add_argument("--times", type=float, nargs=2, default=[0.1, 0.2],
            help="Timestamps for half time and full time. Pass two numbers.")
    parser.add_argument("--xticks", type=float, nargs='+', default=[0.0, 1.0],
            help="Positions of ticks on the x axis. Default is 0.0 and 1.0;"
            + " adjust this if the grid has different bounds.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--no-legend", default=False, action='store_true',
            help="If used, do not add a legend to the plot.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the plot to. If state is a vector,"
            + "the component names will be inserted into the name e.g."
            + "state.png -> [state_rho.png, state_u.png, ...]")

    args = parser.parse_args()
    assert len(args.output) > 0

    init_df = pd.read_csv(args.init, comment='#')
    if 'x' not in init_df:
        raise Exception(f"x column not found in {args.init}; is this state data?")
    if 'y' in init_df:
        print(f"Warning: y column found in {args.init}. This script can only"
                + " handle 1D data; assuming y is actually a state component.")
    component_names = [name for name in init_df if name != 'x']
    if len(component_names) > 1:
        print("Detected vector with components " + ", ".join(component_names))

    if args.rl is not None:
        rl_dfs = [pd.read_csv(filename, comment='#') for filename in args.rl]
    else:
        rl_dfs = None
    if args.weno is not None:
        weno_dfs = [pd.read_csv(filename, comment='#') for filename in args.weno]
    else:
        weno_dfs = None
    if args.true is not None:
        true_dfs = [pd.read_csv(filename, comment='#') for filename in args.true]
    else:
        true_dfs = None

    dir_name, output_file_name = os.path.split(args.output)
    file_short, file_ext = os.path.splitext(output_file_name)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    plt.rcParams.update({'font.size':16})

    for name in component_names:
        fig, ax = plt.subplots()

        init_line = ax.plot(init_df['x'], init_df[name], label="initial condition",
                                **colors.INIT_KWARGS)[0]
        half_lines = []
        full_lines = []
        
        if args.true:
            half_df, full_df = true_dfs
            kwargs = colors.ANALYTICAL_KWARGS
            line = ax.plot(half_df['x'], half_df[name], label="true solution", **kwargs)[0]
            half_lines.append(line)
            line = ax.plot(full_df['x'], full_df[name], label="true solution", **kwargs)[0]
            full_lines.append(line)

        if args.weno:
            half_df, full_df = weno_dfs
            line = ax.plot(half_df['x'], half_df[name], label="WENO solution",
                                **colors.WENO_KWARGS2)[0]
            half_lines.append(line)
            line = ax.plot(full_df['x'], full_df[name], label="WENO solution",
                                **colors.WENO_KWARGS)[0]
            full_lines.append(line)

        if args.rl:
            half_df, full_df = rl_dfs
            line = ax.plot(half_df['x'], half_df[name], label="RL solution",
                                **colors.RL_KWARGS2)[0]
            half_lines.append(line)
            line = ax.plot(full_df['x'], full_df[name], label="RL solution",
                                **colors.RL_KWARGS)[0]
            full_lines.append(line)

        label_pad = -8.0 # Default is 4.0.
        ax.set_xlabel('$x$', labelpad=label_pad)
        ax.set_xmargin(0.0)
        ax.set_xticks(args.xticks)
        ax.locator_params(axis='y', nbins=6)
        if '$' not in name:
            if name == 'rho':
                display_name = r'\rho'
            else:
                display_name = name
            ax.set_ylabel(f"${display_name}$", labelpad=label_pad)
        else:
            ax.set_ylabel(name, labelpad=label_pad)

        if not args.no_legend:
            legend_margin = -0.03
            legend_params = {'fancybox': True, 'shadow': True}
            # Complex legend, only really works for smooth_sine.
            init_legend = ax.legend(handles=[init_line],
                    loc='upper right', bbox_to_anchor=(1.027, 1.047),
                    **legend_params)
            half_legend = ax.legend(handles=half_lines,
                    loc='upper right', bbox_to_anchor=(1.027, 0.939),
                    title=f"$t={args.times[0]}$",
                    **legend_params)

            ax.add_artist(init_legend)
            ax.add_artist(half_legend)

            full_legend = ax.legend(handles=full_lines,
                    loc='lower left', bbox_to_anchor=(-0.005, 0.03),
                    title=f"$t={args.times[1]}$",
                    **legend_params)
            
        if args.title is not None:
                ax.set_title(args.title)
        fig.tight_layout()

        if len(component_names) == 1:
            file_name = output_file_name
        else:
            file_name = f"{file_short}_{name}{file_ext}"
        full_name = os.path.join(dir_name, file_name)
        fig.savefig(full_name)
        print(f"Saved plot to {full_name}.")
        plt.close(fig)

        if len(component_names) == 1:
            gray_name = f"{file_short}_gray{file_ext}"
        else:
            gray_name = f"{file_short}_{name}_gray{file_ext}"
        full_gray_name = os.path.join(dir_name, gray_name)
        plots.make_grayscale_copy(full_name, full_gray_name)

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
