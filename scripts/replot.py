import argparse
import os
import re
import yaml
import pandas as pd

import util.plots as plots
from util import metadata
from util import param_manager
from util.misc import soft_link_directories

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate plots based on data collected during training."
        + " Useful if you've changed the plot functions or added extra data since the run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("progress", type=str,
            help="Path of the progress.csv file containing data to plot.")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
            help="Path of the directory into which to write plots."
            + " By default, add a subdirectory summary_plots into the same"
            + " directory as the progress.csv file.")

    args = parser.parse_args()

    csv_df = pd.read_csv(args.progress, comment='#')

    episode_numbers = csv_df['episodes']
    total_episodes = max(episode_numbers)

    if args.output_dir is None:
        log_dir, _ = os.path.split(args.progress)
        args.output_dir = os.path.join(log_dir, "summary_plots")
    os.makedirs(args.output_dir, exist_ok=True)

    column_names = list(csv_df)
    # Current name style.
    eval_env_names = [re.match("eval_(.*)_end_l2", name).group(1) for name in column_names
                            if re.match("eval_.*_end_l2", name)]
    if len(eval_env_names) == 0:
        # Old name style.
        if eval_env_type not in ["std", "custom"]:
            eval_env_names = [init_type]
        else:
            if env_type == "weno_burgers":
                eval_env_names = ['smooth_sine', 'smooth_rare', 'accelshock']
            elif env_type == "weno_burgers_2d":
                eval_env_names = ["gaussian", "1d-smooth_sine-x", "jsz7"]
            else:
                raise Exception("Unknown eval env names.")

    try:
        avg_train_reward = csv_df['avg_train_total_reward']
    except KeyError:
        avg_train_reward = None
    try:
        avg_eval_reward = csv_df['avg_eval_total_reward']
    except KeyError:
        avg_eval_reward = None
    eval_rewards = [csv_df[f"eval_{env_name}_reward"] for env_name in eval_env_names]
    
    if avg_train_reward is not None or avg_eval_reward is not None:
        main_reward_filename = os.path.join(args.output_dir, "rewards.png")
        plots.plot_reward_summary(main_reward_filename, episodes=episode_numbers,
                total_episodes=total_episodes, eval_envs=eval_rewards,
                eval_env_names=eval_env_names, avg_train=avg_train_reward,
                avg_eval=avg_eval_reward)
    eval_reward_filename = os.path.join(args.output_dir, "final_rewards.png")
    plots.plot_reward_summary(eval_reward_filename, episodes=episode_numbers,
            total_episodes=total_episodes, eval_envs=eval_rewards,
            eval_env_names=eval_env_names)

    try:
        avg_train_l2 = csv_df['avg_train_end_l2']
    except KeyError:
        avg_train_l2 = None
    try:
        avg_eval_l2 = csv_df['avg_eval_end_l2']
    except KeyError:
        avg_eval_l2 = None
    eval_l2s = [csv_df[f"eval_{env_name}_end_l2"] for env_name in eval_env_names]
    
    if avg_train_l2 is not None or avg_eval_l2 is not None:
        main_l2_filename = os.path.join(args.output_dir, "l2.png")
        plots.plot_l2_summary(main_l2_filename, episodes=episode_numbers,
                total_episodes=total_episodes, eval_envs=eval_l2s,
                eval_env_names=eval_env_names, avg_train=avg_train_l2,
                avg_eval=avg_eval_l2)
    eval_l2_filename = os.path.join(args.output_dir, "final_l2.png")
    plots.plot_l2_summary(eval_l2_filename, episodes=episode_numbers,
            total_episodes=total_episodes, eval_envs=eval_l2s,
            eval_env_names=eval_env_names)

    print(f"Updated plots in {args.output_dir}.")

    # Create symlink for convenience.
    if len(log_dir) > 0:
        log_link_name = "last"
        error = soft_link_directories(log_dir, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")


if __name__ == "__main__":
    main()
