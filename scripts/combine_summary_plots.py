import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import util.plots as plots
import util.colors as colors
from util.misc import soft_link_directories
from util.argparse import ExtendAction

def main():
    parser = argparse.ArgumentParser(
        description="""\
Combine summary data from training runs into a single plot. If files do not
contain the same eval envs, the intersection of the given eval envs will be
plotted.
This script is not backwards compatible with old env names (e.g. eval1) and
can only handle new env names (e.g. eval_smooth_sine).
The order of arguments controls the order of the legend.""",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(type=str, nargs='*', action=ExtendAction, dest='curves',
            metavar='FILE',
            help="CSV files containing training data. (progress.csv)")
    parser.add_argument('--files', type=str, nargs='+', action=ExtendAction, dest='curves',
            metavar='FILE',
            help="More CSV files containing training data.")
    parser.add_argument('--avg', type=str, nargs='+', action='append', dest='curves',
            metavar='FILE',
            help="CSV files for which the mean with a confidence\ninterval will be plotted.")
    parser.add_argument('--ci-type', type=str, default='range', help="""\
The type of confidence interval to plot.
Options are: (default: range)
  range: [min,max]
  Xconf: double-tailed CI (T dist), X in [0,1]
  Xsig: [-X stddevs,+X stddevs] (normal dist), X > 0
  Nperc: [Nth percentile,100-Nth], N in [0, 50]""")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
            help="Labels for each file/average. Required for more\nthan one file/average.")
    parser.add_argument("--parts", type=str, nargs='+', default=['loss', 'reward', 'l2'],
            help="Which parts of the summary to combine. All of them by default.")
    parser.add_argument("--eval-only", default=False, action='store_true',
            help="For reward and L2 plots, only plot the eval envs,\n"
            +    "do not plot the training average and eval average.")
    parser.add_argument("--std-only", default=False, action='store_true',
            help="For reward and L2 plots, only plot the training\n"
            +    "average and eval average, do not plot the eval envs.")
    parser.add_argument("--output_dir", "--output-dir", type=str, required=True,
            help="Directory to save the data to. 3 files will be\n"
            +    "saved to that directory: l2.png, reward.png, and\n"
            +    "loss.png.")
    parser.add_argument("--paper-mode", dest='paper_mode', default=True, action='store_true',
            help="Use paper style. Bigger text and specific tweaks.")
    parser.add_argument("--std-mode", dest='paper_mode', action='store_false',
            help="Use standard style. Smaller text, but generalize better to arbitrary data.")

    args = parser.parse_args()

    if args.eval_only and args.std_only:
        raise Exception("Can't use both --eval-only and --std-only.")

    if len(args.output_dir) > 0:
        os.makedirs(args.output_dir, exist_ok=True)

    episodes = []
    intersect_names = None

    # Verify that episodes is the same for averaged files and find the intersection of available
    # eval env names.
    for curve in args.curves:
        if isinstance(curve, str):
            is_list = False
        else:
            is_list = True

        if is_list:
            first_df = pd.read_csv(curve[0])
            first_episodes = first_df['episodes']
            episodes.append(first_episodes)
            if intersect_names is None:
                intersect_names = [re.match("eval_(.*)_end_l2", name).group(1)
                                    for name in list(first_df) if re.match("eval_.*_end_l2", name)]
            else:
                other_names = [re.match("eval_(.*)_end_l2", name).group(1)
                                    for name in list(first_df) if re.match("eval_.*_end_l2", name)]
                intersect_names = [name for name in intersect_names if name in other_names]
            for sub_curve in curve[1:]:
                csv_df = pd.read_csv(sub_curve)
                other_episodes = csv_df['episodes']
                if any(first_episodes != other_episodes):
                    raise Exception(f"Episodes in {curve[0]} do not match episodes in"
                            + f" {sub_curve}.")
                other_names = [re.match("eval_(.*)_end_l2", name).group(1)
                                    for name in list(csv_df) if re.match("eval_.*_end_l2", name)]
                intersect_names = [name for name in intersect_names if name in other_names]
        else:
            csv_df = pd.read_csv(curve)
            episodes.append(csv_df['episodes'])
            if intersect_names is None:
                intersect_names  = [re.match("eval_(.*)_end_l2", name).group(1)
                                    for name in list(csv_df) if re.match("eval_.*_end_l2", name)]
            else:
                other_names = [re.match("eval_(.*)_end_l2", name).group(1)
                                    for name in list(csv_df) if re.match("eval_.*_end_l2", name)]
                intersect_names = [name for name in intersect_names if name in other_names]

    eval_env_prefixes = [f"eval_{name}" for name in intersect_names]
    if args.labels is None:
        if len(args.curves) > 1:
            raise Exception("Must specify --labels for multiple curves.")
        args.labels = [""]
    else:
        if len(args.curves) == 1:
            print("Warning: with only one curve specified, --labels will be ignored.")
        args.labels = [label + " " for label in args.labels]

    reward_and_l2_episodes = []
    loss_episodes = []
    reward_data = []
    l2_data = []
    loss_data = []
    reward_and_l2_labels = []
    reward_and_l2_kwargs = []
    loss_labels = []
    loss_kwargs = []

    # Extract data from all of the files.
    for outer_index, (curve, outer_label) in enumerate(zip(args.curves, args.labels)):
        if isinstance(curve, str):
            is_list = False
        else:
            is_list = True

        if is_list:
            dfs = [pd.read_csv(sub_curve) for sub_curve in curve]
            if not args.eval_only:
                if 'reward' in args.parts:
                    reward_data.append([df['avg_train_total_reward'] for df in dfs])
                if 'l2' in args.parts:
                    l2_data.append([df['avg_train_end_l2'] for df in dfs])
                if len(intersect_names) > 1 or args.std_only:
                    if 'reward' in args.parts:
                            reward_data.append([df['avg_eval_total_reward'] for df in dfs])
                    if 'l2' in args.parts:
                            l2_data.append([df['avg_eval_end_l2'] for df in dfs])
            if not args.std_only:
                if 'reward' in args.parts:
                    reward_data.extend([[df[f"{prefix}_reward"] for df in dfs]
                                        for prefix in eval_env_prefixes])
                if 'l2' in args.parts:
                    l2_data.extend([[df[f"{prefix}_end_l2"] for df in dfs]
                                    for prefix in eval_env_prefixes])

            if 'loss' in args.parts:
                loss_data.append([df['loss'] for df in dfs])
        else:
            df = pd.read_csv(curve)
            if not args.eval_only:
                if 'reward' in args.parts:
                    reward_data.append(df['avg_train_total_reward'])
                if 'l2' in args.parts:
                    l2_data.append(df['avg_train_end_l2'])
                if len(intersect_names) > 1 or args.std_only:
                    if 'reward' in args.parts:
                            reward_data.append(df['avg_eval_total_reward'])
                    if 'l2' in args.parts:
                            l2_data.append(df['avg_eval_end_l2'])
            if not args.std_only:
                if 'reward' in args.parts:
                    reward_data.extend([df[f"{prefix}_reward"] for prefix in eval_env_prefixes])
                if 'l2' in args.parts:
                    l2_data.extend([df[f"{prefix}_end_l2"] for prefix in eval_env_prefixes])
            if 'loss' in args.parts:
                loss_data.append(df['loss'])

        if 'reward' in args.parts or 'l2' in args.parts:
            if not args.eval_only:
                reward_and_l2_labels.append(f"{outer_label}train")
                reward_and_l2_kwargs.append({
                    'color': colors.PERMUTATIONS[outer_index](colors.TRAIN_COLOR)})
                reward_and_l2_episodes.append(episodes[outer_index])
                if len(intersect_names) > 1 or args.std_only:
                    reward_and_l2_labels.append(f"{outer_label}avg eval")
                    reward_and_l2_kwargs.append({
                        'color': colors.PERMUTATIONS[outer_index](colors.AVG_EVAL_COLOR)})
                    reward_and_l2_episodes.append(episodes[outer_index])
                eval_linestyle = '--'
            else:
                eval_linestyle = '-'

            if not args.std_only:
                reward_and_l2_labels.extend([f"{outer_label}{name}" for name in intersect_names])
                reward_and_l2_kwargs.extend([
                    {'color': colors.PERMUTATIONS[outer_index](colors.EVAL_ENV_COLORS[eval_index]),
                        'linestyle': eval_linestyle} for eval_index, _ in enumerate(intersect_names)])
                reward_and_l2_episodes.extend([episodes[outer_index] for _ in intersect_names])

        if 'loss' in args.parts:
            loss_labels.append(outer_label)
            if len(args.curves) == 1:
                loss_kwargs.append({'color': 'black'})
            else:
                loss_kwargs.append({}) # Use matplotlib color cycle.
            loss_episodes.append(episodes[outer_index])

    if args.paper_mode:
        plt.rcParams.update({'font.size':15})

    if 'reward' in args.parts:
        reward_fig = plots.create_avg_plot(reward_and_l2_episodes, reward_data,
                labels=reward_and_l2_labels, kwargs_list=reward_and_l2_kwargs,
                ci_type=args.ci_type)
        ax = reward_fig.gca()
        if len(reward_data) < 3:
            if args.paper_mode:
                bbta=(0.5, 1.11)
            else:
                bbta=(0.5, 1.08)
            ax.legend(loc='upper center', bbox_to_anchor=bbta,
                    ncol=len(reward_data), fancybox=True, shadow=True,
                    prop={'size': 'medium'})
            ax.set_title("Total Reward per Episode", pad=24.0)
        else:
            ax.legend(loc="lower right", prop={'size': plots.legend_font_size(len(reward_data))})
            ax.set_title("Total Reward per Episode")
        ax.set_xmargin(0.0)
        ax.set_xlabel('episodes')
        if args.paper_mode:
            ax.set_ylabel('reward', labelpad=-8.0)
        else:
            ax.set_ylabel('reward')
        ax.grid(True)
        # Use symlog as the rewards are negative.
        # Ugh...
        if matplotlib.__version__ == '3.2.2':
            ax.set_yscale('symlog', linthreshy=1e-9, subsy=range(2,10))
        else:
            ax.set_yscale('symlog', linthresh=1e-9, subs=range(2,10))
        plots.crop_early_shift(ax, "flipped")
        reward_fig.tight_layout()

        reward_filename = os.path.join(args.output_dir, 'reward.png')
        reward_fig.savefig(reward_filename)
        plt.close(reward_fig)
        print(f"Created {reward_filename}.")

    if 'l2' in args.parts:
        l2_fig = plots.create_avg_plot(reward_and_l2_episodes, l2_data,
                labels=reward_and_l2_labels, kwargs_list=reward_and_l2_kwargs,
                ci_type=args.ci_type)
        ax = l2_fig.gca()
        if len(l2_data) < 3:
            if args.paper_mode:
                bbta=(0.5, 1.11)
            else:
                bbta=(0.5, 1.08)
            ax.legend(loc='upper center', bbox_to_anchor=bbta,
                    ncol=len(reward_data), fancybox=True, shadow=True,
                    prop={'size': 'medium'})
            ax.set_title("L2 Error with WENO at End of Episode", pad=24.0)
        else:
            ax.legend(loc="upper right", prop={'size': plots.legend_font_size(len(l2_data))})
            ax.set_title("L2 Error with WENO at End of Episode")
        ax.set_xmargin(0.0)
        ax.set_xlabel('episodes')
        ax.set_ylabel('L2 error')
        ax.grid(True)
        ax.set_yscale('log')
        plots.crop_early_shift(ax, "normal")
        l2_fig.tight_layout()

        l2_filename = os.path.join(args.output_dir, 'l2.png')
        l2_fig.savefig(l2_filename)
        plt.close(l2_fig)
        print(f"Created {l2_filename}.")

    if 'loss' in args.parts:
        loss_fig = plots.create_avg_plot(loss_episodes, loss_data,
                labels=loss_labels, kwargs_list=loss_kwargs,
                ci_type=args.ci_type)
        if len(loss_data) > 1:
            loss_fig.legend(loc="upper right", prop={'size': plots.legend_font_size(len(loss_data))})
        ax = loss_fig.gca()
        ax.set_title("Loss Function")
        ax.set_xmargin(0.0)
        ax.set_xlabel('episodes')
        if args.paper_mode:
            ax.set_ylabel('loss', labelpad=-8.0)
        else:
            ax.set_ylabel('loss')
        ax.grid(True)
        ax.set_yscale('log')
        plots.crop_early_shift(ax, "normal")
        loss_fig.tight_layout()

        loss_filename = os.path.join(args.output_dir, 'loss.png')
        loss_fig.savefig(loss_filename)
        plt.close(loss_fig)
        print(f"Created {loss_filename}.")

    # Create symlink for convenience.
    if len(args.output_dir) > 0:
        log_link_name = "last"
        error = soft_link_directories(args.output_dir, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
