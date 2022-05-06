import os

import argparse
import re
import pandas as pd
import numpy as np

from util.misc import soft_link_directories

def main():
    parser = argparse.ArgumentParser(
            description="Calculate or re-calculate average rewards and l2s"
            + " This can be necessary if you've added additional columns to the"
            + " progress.csv file created during training, typically by using"
            + " scripts/reeval.py. The avg_eval_total_reward and avg_eval_end_l2"
            + " columns will be updated or created if they do not exist.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("progress_file", type=str,
                        help="Path to the progress.csv file with training log data.")
    parser.add_argument("--env-names", type=str, nargs='*', default=None,
                        help="By default, average over all of the evaluation environments"
                        + " found in the file. To average over a subset, specify them"
                        + " here.")
    parser.add_argument('--env-prefix', default=None, type=str,
                        help="Prefix to use in column headers."
                        + " Overrides the --eval and --train options.")
    parser.add_argument('--eval', dest='eval', action='store_true',
                        help="Use 'eval' as the column name prefix.")
    parser.add_argument('--train', dest='eval', action='store_false',
                        help="Use 'train' as the column name prefix. (If you need to repeat"
                        + " evaluation using the training environments.)")
    parser.set_defaults(eval=True)

    args = parser.parse_args()

    csv_df = pd.read_csv(args.progress_file, comment='#')

    if args.env_prefix is None:
        if args.eval:
            env_prefix = "eval"
        else:
            env_prefix = "train"
    else:
        env_prefix = args.env_prefix

    if args.env_names is None:
        env_names = [re.match(f"{env_prefix}_(.*)_end_l2", name).group(1)
                            for name in list(csv_df) if re.match(f"{env_prefix}_.*_end_l2", name)]
        if len(env_names) == 0:
            raise Exception(f"Nothing fitting the {env_prefix}_(.*)_(end_l2|reward) pattern to"
                + " average over.")
        print("Averaging over the following evaluation environments:")
        print(env_names)
    else:
        for name in args.env_names:
            if f"{env_prefix}_{name}_end_l2" not in csv_df:
                raise Exception(f"Couldn't find {env_prefix}_{name} in {args.progress_file}.")
        env_names = args.env_names

    reward_data = [np.array(csv_df[f"{env_prefix}_{name}_reward"]) for name in env_names]
    average_reward = np.mean(reward_data, axis=0)
    if f'avg_{env_prefix}_total_reward' in csv_df:
        print(f"Updating avg_{env_prefix}_total_reward.")
    else:
        print(f"Creating avg_{env_prefix}_total_reward.")
    csv_df[f'avg_{env_prefix}_total_reward'] = average_reward

    l2_data = [np.array(csv_df[f"{env_prefix}_{name}_end_l2"]) for name in env_names]
    average_l2 = np.mean(l2_data, axis=0)
    if f'avg_{env_prefix}_end_l2' in csv_df:
        print(f"Updating avg_{env_prefix}_end_l2.")
    else:
        print(f"Creating avg_{env_prefix}_end_l2.")
    csv_df[f'avg_{env_prefix}_end_l2'] = average_l2

    csv_df.to_csv(args.progress_file, index=False)
    print(f"Saved data to {args.progress_file}.")

    log_dir, file_name = os.path.split(args.progress_file)
    # Create symlink for convenience.
    if len(log_dir) > 0:
        log_link_name = "last"
        error = soft_link_directories(log_dir, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")



if __name__ == "__main__":
    main()
