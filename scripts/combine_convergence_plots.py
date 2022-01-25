import os
import argparse
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from util import plots
import util.colors as colors
from util.misc import soft_link_directories
from util.argparse import ExtendAction

# ActionClass stuff to get argparse to do what I want here.
class ExtendTupleAction(ExtendAction):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        values = [(self.name, value) for value in values]
        super().__call__(parser, namespace, values, option_string=None)

class AppendTupleAction(argparse.Action):
    def __init__(self, name, *args, **kwargs):
        self.name = name
        super().__init__(*args, **kwargs)
    def __call__(self, parser, namespace, values, option_string=None):
        dest = getattr(namespace, self.dest)
        if dest is None:
            dest = []
        dest.append((self.name, values))
        setattr(namespace, self.dest, dest)

class ExtendTupleGen:
    def __init__(self, name):
        self.name = name
    def __call__(self, *args, **kwargs):
        return ExtendTupleAction(self.name, *args, **kwargs)

class AppendTupleGen:
    def __init__(self, name):
        self.name = name
    def __call__(self, *args, **kwargs):
        return AppendTupleAction(self.name, *args, **kwargs)

def main():
    parser = argparse.ArgumentParser(
        description="""\
Combine data from multiple convergence plots into a single plot.
The order of arguments controls the order of the legend.""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(type=str, nargs='*', action=ExtendTupleGen('file'), dest='curves',
            metavar='FILE',
            help="CSV files containing the relevant data. These files\n"
            +    "are in the log directory of prior convergence plot\n"
            +    "runs.")
    parser.add_argument("--files", type=str, nargs='+', action=ExtendTupleGen('file'),
            dest='curves', metavar='FILE',
            help="More CSV files containing convergence data.")
    parser.add_argument('--avg', type=str, nargs='+', action=AppendTupleGen('avg'), dest='curves',
            metavar='FILE',
            help="CSV files for which the GEOMETRIC mean with a\n"
            +   "confidence interval will be plotted.")
    parser.add_argument('--ci-type', type=str, default='range',
            help = "The type of confidence interval to plot. (default: range)\n"
                + " Options are:\n"
                + "   range: [min,max]\n"
                + "   Xconf: [P(lower)=(1-X)/2,P(higher)=(1-X)/2] (T dist), X in [0,1]\n"
                + "   Xsig: [-X std deviations,+X std deviations] (normal dist), X > 0\n"
                + "   Nperc: [Nth percentile,100-Nth percentile], N in [0, 50]\n")
    parser.add_argument("--weno", type=int, nargs='+', action=ExtendTupleGen('weno'), dest='curves',
            metavar='ORDER',
            help="Add the convergence curve for the given weno order.\n"
            +    "This uses the --weno-path, and data must be in\n"
            +    "WENO_PATH/order_ORDER/progress.csv.")
    parser.add_argument("--weno-path", "--weno_path", type=str,
            default="test/weno_burgers/gaussian_convergence/weno",
            help="Path for data to use with the --weno argument.")
    parser.add_argument("--poly", type=int, nargs='+', action=ExtendTupleGen('poly'),
            dest='curves', metavar='ORDER',
            help="Add a curve representing exact polynomial accuracy\n"
            +    "for comparison. The curve will be near the previous\n"
            +    "curve.")
    parser.add_argument("--labels", type=str, nargs='*', default=None,
            help="Labels for each file. By default, no labels are used.\n"
            +    "Labels for WENO and polynomial curves use the default\n"
            +    "built-in labels so should not be provided. If only\n"
            +    "WENO and polynomial curves are plotted, indicate that\n"
            +    "labels should be added by using --labels with no\n"
            +    "arguments.")
    parser.add_argument("--title", type=str, default=None,
            help="Title to add to the plot. By default, no title is added.")
    parser.add_argument("--output", "-o", type=str, required=True,
            help="Path to save the combined plot to. If an existing\n"
            +    "directory is passed, the file will be saved to\n"
            +    "'convergence.png' in that directory.")
    parser.add_argument("--paper-mode", dest='paper_mode', default=True, action='store_true',
            help="Use paper style. Bigger text and specific tweaks.")
    parser.add_argument("--std-mode", dest='paper_mode', action='store_false',
            help="Use standard style. Smaller text, but generalize better to arbitrary data.")

    args = parser.parse_args()
    assert len(args.output) > 0

    create_legend = (args.labels is not None)
    if args.labels is None:
        args.labels = [""] * len(args.curves)
    else:
        num_labeled_curves = sum((1 for (curve_type, _) in args.curves
                                        if curve_type in ['file', 'avg']))
        if len(args.labels) != num_labeled_curves:
            raise Exception(f"Number of labels ({len(args.labels)}) must match"
                    + f" number of curves to plot excluding curves that have"
                    + f" default labels ({num_labeled_curves}).")

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

            if not args.paper_mode:
                kwargs = {}
            else:
                kwargs = colors.get_agent_kwargs(file_name, args.labels[index], just_color=False)
                kwargs['linestyle'] = '-'
            kwargs_list.append(kwargs)
        elif curve_type == "avg":
            names = curve_id
            dfs = [pd.read_csv(name, comment='#') for name in names]
            first_nx = None
            for name, df in zip(names, dfs):
                if 'nx' not in df and 'num_cells' not in df:
                    raise Exception(f"Can't find nx values in {name}.")
                nx_data = dfs[0]['nx' if 'nx' in dfs[0] else 'num_cells']
                if first_nx is None:
                    first_nx = nx_data
                elif any(first_nx != nx_data):
                    raise Exception(f"nx values in {names[0]} do not match {name}.")
            sizes = first_nx
            error_list = [df['l2_error'] for df in dfs]
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
            if create_legend:
                args.labels.insert(index, f"WENO, r={order}")
            kwargs_list.append({'color': colors.WENO_ORDER_COLORS[order], 'linestyle': '--'})
        elif curve_type == "poly":
            if len(grid_sizes) == 0:
                raise Exception("Polynomial cannot be the first curve in the plot.")
            order = curve_id
            previous_sizes = np.array(grid_sizes[-1])
            previous_errors = errors[-1]
            sizes, error_list, poly_label = plots.generate_polynomial(
                                                    order, previous_sizes, previous_errors)
            if create_legend:
                args.labels.insert(index, poly_label)
            kwargs_list.append({'color': colors.POLY_COLORS[order], 'linestyle': ':'})

        grid_sizes.append(sizes)
        errors.append(error_list)

    if args.paper_mode:
        plt.rcParams.update({'font.size':18})

    fig = plots.create_avg_plot(grid_sizes, errors,
            labels=args.labels, kwargs_list=kwargs_list,
            ci_type=args.ci_type, avg_type='geometric')
    ax = fig.gca()
    if create_legend:
        if args.paper_mode:
            bbta = (1.0, 1.0)
        else:
            bbta = (1.03, 1.05)
        ax.legend(loc="upper right", bbox_to_anchor=bbta,
                ncol=1, fancybox=True, shadow=True,
                prop={'size': plots.legend_font_size(len(grid_sizes))})
    if args.paper_mode:
        ax.set_xlabel("grid size", labelpad=-8.0)
    else:
        ax.set_xlabel("grid size")
    ax.set_xscale('log')
    #ax.set_xticks(grid_sizes) # Use the actual grid sizes as ticks instead of powers of 10.
    ax.set_ylabel("L2 error")
    ax.set_yscale('log')
    if args.title is not None:
        ax.set_title(args.title)
    plt.tight_layout()

    dir_name, file_name = os.path.split(args.output)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    if file_name == "":
        file_name = "convergence.png"
        args.output = os.path.join(dir_name, file_name)

    plt.savefig(args.output)
    print(f"Saved plot to {args.output}.")
    plt.close()

    # Create symlink for convenience.
    if len(dir_name) > 0:
        log_link_name = "last"
        error = soft_link_directories(dir_name, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
