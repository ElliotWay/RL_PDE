import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import plots
import util.colors as colors
from util.misc import soft_link_directories
from util.argparse import ExtendAction

def main():
    parser = argparse.ArgumentParser(
        description="Combine summary data from training runs into a single plot."
        + "\nIf files do not contain the same eval envs, the intersection of the"
        + "\ngiven eval envs will be plotted."
        + "\nThis script is not backwards compatible with old env names (e.g. eval1)"
        + "\nand can only handle new env names (e.g. eval_smooth_sine)."
        + "\nThe order of arguments affects the order of the legend.",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(type=str, nargs='*', action=ExtendAction, dest='curves',
            metavar='FILE',
            help="CSV files containing training data. (progress.csv)")
    parser.add_argument('--files', type=str, nargs='+', action=ExtendAction, dest='curves',
            metavar='FILE',
            help="More CSV files containing training data.")
    parser.add_argument('--avg', type=str, nargs='+', action='append', dest='curves',
            metavar='FILE',
            help="CSV files for which the mean with a confidence interval will be plotted.")
    parser.add_argument('--ci-type', type=str, default='range',
            help = "The type of confidence interval to plot. (default: range)\n"
                + " Options are:\n"
                + "   range: [min,max]\n"
                + "   Xconf: [P(lower)=(1-X)/2,P(higher)=(1-X)/2] (T dist), X in [0,1]\n"
                + "   Xsig: [-X std deviations,+X std deviations] (normal dist), X > 0\n"
                + "   Nperc: [Nth percentile,100-Nth percentile], N in [0, 50]\n")
    parser.add_argument("--labels", type=str, nargs='+', default=None,
            help="Labels for each file/average. Required for >1 file/average.")
    parser.add_argument("--eval-only", default=False, action='store_true',
            help="For reward and L2 plots, only plot the eval envs, do not plot the"
            + "\ntraining average and eval average.")
    parser.add_argument("--output_dir", "--output-dir", type=str, required=True,
            help="Directory to save the data to. 3 files will be saved to that directory:"
            + "\nl2.png, reward.png, and loss.png.")

    args = parser.parse_args()

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
                reward_data.append([df['avg_train_total_reward'] for df in dfs])
                l2_data.append([df['avg_train_end_l2'] for df in dfs])
                if len(intersect_names) > 1:
                    reward_data.append([df['avg_eval_total_reward'] for df in dfs])
                    l2_data.append([df['avg_eval_end_l2'] for df in dfs])

            reward_data.extend([[df[f"{prefix}_reward"] for df in dfs]
                                    for prefix in eval_env_prefixes])
            l2_data.extend([[df[f"{prefix}_end_l2"] for df in dfs]
                                for prefix in eval_env_prefixes])

            loss_data.append([df['loss'] for df in dfs])
        else:
            df = pd.read_csv(curve)
            if not args.eval_only:
                reward_data.append(df['avg_train_total_reward'])
                l2_data.append(df['avg_train_end_l2'])
                if len(intersect_names) > 1:
                    reward_data.append(df['avg_eval_total_reward'])
                    l2_data.append(df['avg_eval_end_l2'])
            reward_data.extend([df[f"{prefix}_reward"] for prefix in eval_env_prefixes])
            l2_data.extend([df[f"{prefix}_end_l2"] for prefix in eval_env_prefixes])
            loss_data.append(df['loss'])

        if not args.eval_only:
            reward_and_l2_labels.append(f"{outer_label}train")
            reward_and_l2_kwargs.append({
                'color': colors.PERMUTATIONS[outer_index](colors.TRAIN_COLOR)})
            reward_and_l2_episodes.append(episodes[outer_index])
            if len(intersect_names) > 1:
                reward_and_l2_labels.append(f"{outer_label}avg eval")
                reward_and_l2_kwargs.append({
                    'color': colors.PERMUTATIONS[outer_index](colors.AVG_EVAL_COLOR)})
                reward_and_l2_episodes.append(episodes[outer_index])
            eval_linestyle = '--'
        else:
            eval_linestyle = '-'

        reward_and_l2_labels.extend([f"{outer_label}{name}" for name in intersect_names])
        reward_and_l2_kwargs.extend([
            {'color': colors.PERMUTATIONS[outer_index](colors.EVAL_ENV_COLORS[eval_index]),
                'linestyle': eval_linestyle} for eval_index, _ in enumerate(intersect_names)])
        reward_and_l2_episodes.extend([episodes[outer_index] for _ in intersect_names])

        loss_labels.append(outer_label)
        loss_kwargs.append({}) # Use matplotlib color cycle.
        loss_episodes.append(episodes[outer_index])

    reward_fig = plots.create_avg_plot(reward_and_l2_episodes, reward_data,
            labels=reward_and_l2_labels, kwargs_list=reward_and_l2_kwargs,
            ci_type=args.ci_type)
    reward_fig.legend(loc="lower right", prop={'size': plots.legend_font_size(len(reward_data))})
    ax = reward_fig.gca()
    ax.set_title("Total Reward per Episode")
    ax.set_xmargin(0.0)
    ax.set_xlabel('episodes')
    ax.set_ylabel('reward')
    ax.grid(True)
    # Use symlog as the rewards are negative.
    ax.set_yscale('symlog')
    plots.crop_early_shift(ax, "flipped")

    reward_filename = os.path.join(args.output_dir, 'reward.png')
    reward_fig.savefig(reward_filename)
    plt.close(reward_fig)
    print(f"Created {reward_filename}.")

    l2_fig = plots.create_avg_plot(reward_and_l2_episodes, l2_data,
            labels=reward_and_l2_labels, kwargs_list=reward_and_l2_kwargs,
            ci_type=args.ci_type)
    l2_fig.legend(loc="upper right", prop={'size': plots.legend_font_size(len(l2_data))})
    ax = l2_fig.gca()
    ax.set_title("L2 Error with WENO at End of Episode")
    ax.set_xmargin(0.0)
    ax.set_xlabel('episodes')
    ax.set_ylabel('L2 error')
    ax.grid(True)
    ax.set_yscale('log')
    plots.crop_early_shift(ax, "normal")

    l2_filename = os.path.join(args.output_dir, 'l2.png')
    l2_fig.savefig(l2_filename)
    plt.close(l2_fig)
    print(f"Created {l2_filename}.")

    loss_fig = plots.create_avg_plot(loss_episodes, loss_data,
            labels=loss_labels, kwargs_list=loss_kwargs,
            ci_type=args.ci_type)
    if len(loss_data) > 1:
        loss_fig.legend(loc="upper right", prop={'size': plots.legend_font_size(len(loss_data))})
    ax = loss_fig.gca()
    ax.set_title("Loss Function")
    ax.set_xmargin(0.0)
    ax.set_xlabel('episodes')
    ax.set_ylabel('loss')
    ax.grid(True)
    ax.set_yscale('log')
    plots.crop_early_shift(ax, "normal")

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
