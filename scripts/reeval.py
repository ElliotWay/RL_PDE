import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #Block GPU for now.

import argparse
import re
import yaml
import pandas as pd
import numpy as np

# Import tf now to set error level.
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from rl_pde.run import rollout
from rl_pde.emi import DimensionalAdapterEMI, VectorAdapterEMI
from envs import builder as env_builder
from envs import Plottable1DEnv, Plottable2DEnv
from models import builder as model_builder
from util import metadata
from util import sb_logger as logger
from util import plots
from util import filelock
from util.function_dict import numpy_fn
from util.param_manager import ArgTreeManager
from util.lookup import get_model_class, get_emi_class, get_model_dims
from util.misc import soft_link_directories

# Keys in the old style meta files to ignore.
META_IGNORE_KEYS = ['time started', 'time finished', 'initiated by user', 'pid',
                        'status', 'git status', 'git branch', 'git commit id', 'git commit hash']

def main():
    arg_manager = ArgTreeManager()
    parser = argparse.ArgumentParser(
            description="Test the agents from training on a different environment,"
            + " as if the new environment had been used for evaluation during training."
            + " This requires that all saved agents are still available and have not been deleted"
            + " since training."
            + " Note that this does not update the average eval values in existing files.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--meta", type=str,
                        help="Path to the meta file of the original training run."
                        + " The directory of the meta file must contain the other experiment files.")
    parser.add_argument("--start", type=int, default=None,
                        help="Episode to start reevalutation on. Models from earlier episodes will not"
                        + " be reevalutated. Note that this will probably break the --copy "
                        + " or --append options.")
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Show the environment parameters not listed here.")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment to test on.")
    parser.add_argument('--model', type=str, default="full",
                        help="This model will not be used; however, some environment defaults"
                        + " depend on the model. Those can be controlled with this.")
    parser.add_argument('--output-dir', '--output_dir', '-o', default=None, type=str,
                        help="Directory to save the output to, if --append is not specified.")
    parser.add_argument('--plot-l2', '--plot_l2', default=False, action='store_true',
                        help="Plot the L2 error vs time for each episode.")
    parser.add_argument('--append', default=False, action='store_true',
                        help="Update the experiment files with the new data, instead of"
                        + " creating/updating files in --output-dir.")
    parser.add_argument('--copy', default=False, action='store_true',
                        help="Copy the data of the original experiment files into the new"
                        + " directory.")
    parser.add_argument('--output-mode', '--output_mode', default=['csv'], nargs='+',
                        help="Which kinds of files to create/update. csv creates/updates"
                        + " progress.csv with the reward and L2 error for each run. plot creates"
                        + " evolution plots for each run. Multiple output modes can be selected.")
    parser.add_argument('--replot', default=False, action='store_true',
                        help="If --output-mode csv is used, create/update summary plots using"
                        + " the new data.")
    parser.add_argument('--env-name', default=None, type=str,
                        help="If --output-mode csv is specified, the name of the environment"
                        + " to use as a column header. Defaults to the --init-type.")

    arg_manager.set_parser(parser)
    env_arg_manager = arg_manager.create_child("e", long_name="Environment Parameters")
    env_arg_manager.set_parser(env_builder.get_env_arg_parser())

    reeval_args, rest = arg_manager.parse_known_args()

    if reeval_args.help_env:
        env_arg_manager.print_help()
        sys.exit(0)

    if len(rest) > 0:
        print("Unrecognized arguments: " + " ".join(rest))
        sys.exit(0)

    old_log_dir, file_name = os.path.split(reeval_args.meta)

    if reeval_args.output_dir is not None:
        if reeval_args.append:
            raise Exception("Cannot use both --append and --output-dir.")
        else:
            new_log_dir = reeval_args.output_dir
            if len(new_log_dir) > 0:
                os.makedirs(new_log_dir, exist_ok=True)
    else:
        new_log_dir = old_log_dir
    if reeval_args.copy and reeval_args.append:
        raise Exception("Cannot use both --copy and --append.")

    if reeval_args.env_name is None:
        reeval_args.env_name = reeval_args.e.init_type

    # Build the new environment
    env_builder.set_contingent_env_defaults(reeval_args, reeval_args.e, test=True)
    env = env_builder.build_env(reeval_args.env, reeval_args.e, test=True)
    dims = env_builder.env_dimensions(reeval_args.env)

    # Load the meta file of the original experiment.
    if os.path.basename(reeval_args.meta) == metadata.META_FILE_NAME:
        open_file = open(reeval_args.meta, 'r')
        args_dict = yaml.safe_load(open_file)
        open_file.close()

        loaded_arg_manager = ArgTreeManager()
        loaded_arg_manager.init_from_dict(args_dict, children_names=['e', 'm'])
    elif os.path.basename(reeval_args.meta) == metadata.OLD_META_FILE_NAME:
        env_args = vars(env_builder.get_env_arg_parser().parse_args(""))
        model_args = vars(model_builder.get_model_arg_parser().parse_args(""))

        full_dict = metadata.load_meta_file(filename)
        main_dict = {}
        env_dict = {}
        model_dict = {}
        for k, v in full_dict.items():
            v = metadata.destring_value(v)
            if k in META_IGNORE_KEYS:
                pass
            elif k in env_args:
                env_dict[k] = v
            elif k in model_args:
                model_dict[k] = v
            else:
                main_dict[k] = v

        loaded_arg_manager = ArgTreeManager()
        loaded_arg_manager.init_from_dict(main_dict, children_names=[])

        loaded_env_manager = arg_manager.create_child('e')
        loaded_env_manager.init_from_dict(env_dict, children_names=[])

        loaded_model_manager = arg_manager.create_child('m')
        loaded_model_manager.init_from_dict(model_dict, children_names=[])

        # Need to do this manually since we're initializing in an unusual way.
        arg_dict = vars(loaded_arg_manager.args)
        arg_dict['e'] = loaded_env_manager
        arg_dict['m'] = loaded_model_manager
    else:
        raise Exception(f"{reeval_args.meta} not a recognized meta file type.")

    args = loaded_arg_manager.args
    # Some things refer to logger to access the log_dir, so we need to set it here.
    logger.configure(folder=new_log_dir, format_strs=['stdout'])
    logger.set_level(logger.DEBUG)

    # Get a list of all the agents to run.
    old_csv_filename = os.path.join(old_log_dir, 'progress.csv')
    old_csv_df = pd.read_csv(old_csv_filename, comment='#', float_precision='round_trip')
    episode_numbers = old_csv_df['episodes']
    if reeval_args.start is not None:
        episode_numbers = [int(num) for num in episode_numbers if int(num) >= reeval_args.start]

    # Some work needed to check for possible variable number formatting.
    agent_filenames = {num: None for num in episode_numbers}
    for entry in os.scandir(old_log_dir):
        if not entry.is_file():
            continue
        match = re.fullmatch("model(\d+).zip", entry.name)
        if match is not None:
            number = int(match.group(1))
            if number in episode_numbers:
                assert agent_filenames[number] is None, f"Already have agent for {number}"
                agent_filenames[number] = entry.path
    missing_filenames = [num for num, name in agent_filenames.items() if name is None]
    if len(missing_filenames) > 0:
        raise Exception("Can't reevaluate, missing agents for logged episodes:\n"
                + str(missing_filenames[:20]))

    # Build the emi and model to load parameters into. (Copied from run_test.py.)
    obs_adjust = numpy_fn(args.obs_scale)
    action_adjust = numpy_fn(args.action_scale)

    model_cls = get_model_class(args.model)
    emi_cls = get_emi_class(args.emi)
    model_dims = get_model_dims(args.model)

    if model_dims < dims:
        if model_dims == 1:
            emi = DimensionalAdapterEMI(emi_cls, env, model_cls, args,
                    obs_adjust=obs_adjust, action_adjust=action_adjust)
        else:
            raise Exception("Cannot adapt {}-dimensional model to {}-dimensional environment."
                    .format(model_dims, dims))
    elif args.env == 'weno_euler':
        emi = VectorAdapterEMI(emi_cls, env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)
    else:
        emi = emi_cls(env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)

    # Run tests with each agent.
    reward_data = []
    l2_data = []
    ep_precision = int(np.ceil(np.log(1+args.total_episodes) / np.log(10)))
    for index, ep_num in enumerate(episode_numbers):
        agent_file = agent_filenames[ep_num]

        emi.load_model(agent_file)
        agent = emi.get_policy()
 
        _, _, rewards, _, _ = rollout(env, agent, deterministic=True)

        ep_string = ("{:0" + str(ep_precision) + "}").format(ep_num)
        if 'csv' in reeval_args.output_mode:
            reward_parts = list(zip(*rewards))
            avg_total_reward = np.mean([np.mean(np.sum(reward_part, axis=0))
                                            for reward_part in reward_parts])
            reward_data.append(avg_total_reward)
            l2_data.append(env.compute_l2_error())

            if reeval_args.plot_l2:
                times = env.timestep_history
                l2s = env.compute_l2_error(timestep="all")
                episode_filename = os.path.join(new_log_dir, f"ep_{ep_string}_progress.csv")
                progress_df = pd.DataFrame({'t':times, 'l2': l2s})
                progress_df.to_csv(episode_filename, index=False)

        if 'plot' in reeval_args.output_mode:
            ep_string = ("{:0" + str(ep_precision) + "}").format(ep_num)
            suffix = f"_{reeval_args.env_name}_ep_{ep_string}"
            if isinstance(env, Plottable1DEnv):
                env.plot_state_evolution(num_states=10, full_true=False, no_true=False,
                        plot_weno=False, suffix=suffix, silent=True)
            elif isinstance(env, Plottable2DEnv):
                env.plot_state_evolution(num_frames=20, suffix=suffix, silent=True)
            else:
                raise Exception()
            if reeval_args.plot_l2:
                times = env.timestep_history
                l2s = env.compute_l2_error(timestep="all")
                filename = f"ep_{ep_string}_l2.png"
                plots.plot_over_time(times, l2s, log_dir=new_log_dir, name=filename,
                        title="L2 Error", silent=False)
        if index % 10 == 9:
            print(ep_num, flush=True)
        else:
            print('.', end='', flush=True)

    if 'csv' in reeval_args.output_mode:
        if reeval_args.append:
            output_filename = old_csv_filename
        else:
            output_filename = os.path.join(new_log_dir, "progress.csv")
        # We are plausibly running this from multiple processes; e.g. re-evaluating several environments
        # at once. In that case, we need to handle updating the progress.csv file carefully.
        with filelock.exclusive_open(output_filename, 'a+') as output_file:
            
            if reeval_args.append or reeval_args.copy:
                if reeval_args.append:
                    output_file.seek(0)
                    output_df = pd.read_csv(output_file, comment='#', float_precision='round_trip')
                else: # reeval_args.copy
                    output_df = old_csv_df

                # If the existing column names use the old numeric format, update them with the newer
                # descriptive format, as the new entry will definitely have the new format.
                column_names = list(output_df)
                old_names = [name for name in column_names
                                if re.fullmatch("eval\d+_(reward|end_l2)", name)]
                if old_names:
                    if args.env == "weno_burgers":
                        eval_names = {'1': 'smooth_sine', '2': 'smooth_rare', '3': 'accelshock'}
                    elif args.env == "weno_burgers_2d":
                        eval_names = {'1': 'gaussian', '2': '1d-smooth_sine-x', '3': 'jsz7'}
                    elif args.env == "weno_euler":
                        eval_names = {'1': 'sod'}
                    numbers = {re.match("eval(\d+)_", name).group(1) for name in old_names}
                    replacement_dict = {}
                    for name in old_names:
                        match = re.fullmatch("eval(\d+)_(reward|end_l2)", name)
                        try:
                            eval_name = eval_names[match.group(1)]
                        except KeyError:
                            eval_name = input(f"The name of eval env {match.group(1)} is unknown."
                                    + " Enter the name here: ")
                            if eval_name in eval_names.values():
                                raise Exception(f"{eval_name} is already associated.")
                            eval_names[match.group(1)] = eval_name
                        new_name = f"eval_{eval_name}_{match.group(2)}"
                        replacement_dict[name] = new_name
                    output_df.rename(columns=replacement_dict, inplace=True)

                if reeval_args.copy:
                    # If copying the original data but the target file is non-empty, add any columns
                    # from the target file that aren't in the original data.
                    try:
                        output_file.seek(0)
                        existing_df = pd.read_csv(output_file, comment='#',
                                float_precision='round_trip')
                    except pd.errors.EmptyDataError:
                        # The file does not yet exist. We can use the data already copied from the
                        # original experiment file.
                        pass
                    else:
                        num_rows = len(output_df.index)
                        old_rows = len(existing_df)
                        merged_df = output_df.merge(existing_df)
                        outer_merge_df = output_df.merge(existing_df, how='outer')
                        if len(merged_df.index) != num_rows:
                            print("Failed to merge old csv file with the target one.")
                            if num_rows != old_rows:
                                raise Exception("Failed to merge CSV files."
                                        + f"Original rows is {num_rows}, target rows is {old_rows}"
                                        + f" (merged rows is {len(merged_df.index)}).")
                            else:
                                cols_intersection = [name for name in output_df if name in existing_df]
                                mismatched_cols = [name for name in cols_intersection if
                                        any(output_df[name] != existing_df[name])]
                                if len(mismatched_cols) > 0:
                                    #left = output_df[mismatched_cols[0]]
                                    #right = existing_df[mismatched_cols[0]]
                                    #mismatched_rows = [index for index in range(len(output_df))
                                    #        if left[index] != right[index]]
                                    #print("row: original != target")
                                    #for row in mismatched_rows:
                                    #    print(f"{row}: {left[row]} != {right[row]}")
                                    raise Exception("Failed to merge CSV files. These rows were in"
                                            + " both files but are not identical:" +
                                            str(mismatched_cols))
                                else:
                                    raise Exception("Failed to merge CSV files. The reason for"
                                            + " this is unclear - they seem to have the"
                                            + " same number of rows and identical matching"
                                            + " columns.")
                        else:
                            output_df = merged_df

            else: # not reeval.args and not reeval.copy
                try:
                    output_file.seek(0)
                    output_df = pd.read_csv(output_file, comment='#', float_precision='round_trip')
                except pd.errors.EmptyDataError:
                    # The file does not yet exist; create a new DataFrame.
                    output_df = pd.DataFrame({'episodes': episode_numbers})

            env_name = reeval_args.env_name
            reward_name = f"eval_{env_name}_reward"
            l2_name = f"eval_{env_name}_end_l2"
            original_env_name = env_name

            # Ensure that the specified env_name does not overwrite an existing column.
            while reward_name in output_df or l2_name in output_df:
                match = re.fullmatch("([^\d]+)(\d+)", env_name)
                if match:
                    env_name = match.group(1) + str(int(match.group(2))+1)
                else:
                    env_name = env_name + "1"
                reward_name = f"eval_{env_name}_reward"
                l2_name = f"eval_{env_name}_end_l2"
            if env_name != original_env_name:
                print(f"Environment name '{original_env_name}' is already in {output_filename}!"
                        + f" Adjusted to '{env_name}'.")

            output_df[reward_name] = reward_data
            output_df[l2_name] = l2_data

            output_file.seek(0)
            output_file.truncate()
            output_df.to_csv(output_file, index=False)
        # /with Close and unlock output_file.
        reread_df = pd.read_csv(output_filename, comment='#', float_precision='round_trip')

        if reeval_args.replot:
            summary_plot_dir = os.path.join(new_log_dir, "summary_plots")
            os.makedirs(summary_plot_dir, exist_ok=True)
            total_episodes = np.max(episode_numbers)
            eval_env_names = [re.fullmatch("eval_(.+)_reward", name).group(1)
                        for name in list(output_df) if re.fullmatch("eval_.+_reward", name)]

            try:
                avg_train_reward = output_df['avg_train_total_reward']
            except KeyError:
                avg_train_reward = None
            try:
                avg_eval_reward = output_df['avg_eval_total_reward']
            except KeyError:
                avg_eval_reward = None
            eval_rewards = [output_df[f"eval_{env_name}_reward"] for env_name in eval_env_names]
            
            if avg_train_reward is not None or avg_eval_reward is not None:
                main_reward_filename = os.path.join(summary_plot_dir, "rewards.png")
                plots.plot_reward_summary(main_reward_filename, episodes=episode_numbers,
                        total_episodes=total_episodes, eval_envs=eval_rewards,
                        eval_env_names=eval_env_names, avg_train=avg_train_reward,
                        avg_eval=avg_eval_reward)
            eval_reward_filename = os.path.join(summary_plot_dir, "final_rewards.png")
            plots.plot_reward_summary(eval_reward_filename, episodes=episode_numbers,
                    total_episodes=total_episodes, eval_envs=eval_rewards,
                    eval_env_names=eval_env_names)

            try:
                avg_train_l2 = output_df['avg_train_end_l2']
            except KeyError:
                avg_train_l2 = None
            try:
                avg_eval_l2 = output_df['avg_eval_end_l2']
            except KeyError:
                avg_eval_l2 = None
            eval_l2s = [output_df[f"eval_{env_name}_end_l2"] for env_name in eval_env_names]
            
            if avg_train_l2 is not None or avg_eval_l2 is not None:
                main_l2_filename = os.path.join(summary_plot_dir, "l2.png")
                plots.plot_l2_summary(main_l2_filename, episodes=episode_numbers,
                        total_episodes=total_episodes, eval_envs=eval_l2s,
                        eval_env_names=eval_env_names, avg_train=avg_train_l2,
                        avg_eval=avg_eval_l2)
            eval_l2_filename = os.path.join(summary_plot_dir, "final_l2.png")
            plots.plot_l2_summary(eval_l2_filename, episodes=episode_numbers,
                    total_episodes=total_episodes, eval_envs=eval_l2s,
                    eval_env_names=eval_env_names)

            print(f"Updated plots in {summary_plot_dir}.")

    # Create symlink for convenience.
    if len(new_log_dir) > 0:
        log_link_name = "last"
        error = soft_link_directories(new_log_dir, log_link_name, safe=True)
        if error:
            print("Note: Failed to create \"last\" symlink.")



if __name__ == "__main__":
    main()
