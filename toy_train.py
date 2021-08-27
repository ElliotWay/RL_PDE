import os
os.environ['PYTHONHASHSEED'] = "42069"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #Block GPU for now.
import argparse
from argparse import Namespace
import shutil
import signal
import sys
import time
import subprocess

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#TODO: Remove dependency on this logger. The functionality we need can be easily implemented.
from stable_baselines import logger

from envs.toy import SweeperEnv
from envs import builder as env_builder
from rl_pde.emi import ToyBatchGlobalEMI
from rl_pde.toy_run import train
from models import get_model_arg_parser
from models import GlobalBackpropModel
from util import metadata
from util.lookup import get_model_class
from util.function_dict import numpy_fn
from util.misc import set_global_seed

ON_POSIX = 'posix' in sys.builtin_module_names

def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent in a toy environment. Note that this script also takes"
        + " various arguments not listed here. These can be shown by using the --help-model and"
        + " --help-env arguments; however, not all of those arguments will be relevant to toy"
        + " environments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-model', default=False, action='store_true',
                        help="Do not train and show the model training parameters not listed here.")
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not train and show the environment parameters not listed here.")
    parser.add_argument('--model', type=str, default='full',
                        help="Type of model to train. Options are 'full', 'sac', and 'pg'.")
    parser.add_argument('--env', type=str, default="sweeper",
                        help="Name of the environment in which to train the agent.")
    #parser.add_argument('--emi', type=str, default='batch',
                        #help="Environment-model interface. Options are 'batch' and 'std'.")
    parser.add_argument('--obs-scale', '--obs_scale', type=str, default='none',
                        help="Adjustment function to observation. Compute Z score along the last"
                        + " dimension (the stencil) with 'z_score_last', the Z score along every"
                        + " dimension with 'z_score_all', or leave them the same with 'none'.")
    parser.add_argument('--action-scale', '--action_scale', type=str, default='binary',
                        help="Adjustment function to action. Default depends on environment."
                        + " 'softmax' computes softmax, 'rescale_from_tanh' scales to [0,1] then"
                        + " divides by the sum of the weights, 'none' does nothing.")
    parser.add_argument('--log-dir', '--log_dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is"
                        + " log/env/model/timestamp.")
    parser.add_argument('--ep-length', '--ep_length', type=int, default=25,
                        help="Number of timesteps in an episode.")
    parser.add_argument('--total-episodes', '--total_episodes', type=int, default=100,
                        help="Total number of episodes to train.")
    parser.add_argument('--log-freq', '--log_freq', type=int, default=5,
                        help="Number of episodes to wait between logging information.")
    parser.add_argument('--n-best-models', '--n_best_models', type=int, default=3,
                        help="Number of best models so far to keep track of.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--repeat', type=str, default=None,
                        help="Copy parameters from a meta.txt file to run a similar or identical experiment."
                        + " Explicitly passed parameters override parameters in the meta file."
                        + " Passing an explicit --log-dir is recommended.")
    parser.add_argument('-y', '--y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', '--n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files."
                        + " Useful for scripts. Overrides the -y option.")

    main_args, rest = parser.parse_known_args()

    env_arg_parser = env_builder.get_env_arg_parser()
    env_args, rest = env_arg_parser.parse_known_args(rest)
    #env_builder.set_contingent_env_defaults(main_args, env_args) # Not needed for toy envs?

    model_arg_parser = get_model_arg_parser()
    model_args, rest = model_arg_parser.parse_known_args(rest)
 
    args = Namespace(**vars(main_args), **vars(env_args), **vars(model_args))

    if args.help_model:
        if args.help_model:
            model_arg_parser.print_help()
        sys.exit(0)

    if len(rest) > 0:
        raise Exception("Unrecognized arguments: " + " ".join(rest))

    if args.repeat is not None:
        metadata.load_to_namespace(args.repeat, args)

    set_global_seed(args.seed)

    if args.env == "sweeper":
        if args.num_cells is None:
            env_length = 25
        else:
            env_length = args.num_cells[0]
        env = SweeperEnv(env_length, max_timesteps=args.ep_length)
        if args.obs_scale is None:
            args.obs_scale = "none"
        if args.action_scale is None:
            args.action_scale = "binary"
    else:
        raise Exception("Toy environment \"{}\" not recognized.".format(args.env))

    # Fill in default args that depend on other args (at least in the main version).
    if args.batch_size is None:
        args.batch_size = 10 if args.model == "full" else 64

    #if args.replay_style == 'marl':
        #if args.emi != 'marl':
            #args.emi = 'marl'
            #print("EMI set to MARL for MARL-style replay buffer.")
        #if args.batch_size % (args.nx + 1) != 0:
            #old_batch_size = args.batch_size
            #new_batch_size = old_batch_size + (args.nx + 1) - (old_batch_size % (args.nx + 1))
            #args.batch_size = new_batch_size
            #print("Batch size changed from {} to {} to align with MARL-style replay buffer."
                    #.format(old_batch_size, new_batch_size))
        #if args.buffer_size % (args.nx + 1) != 0:
            #old_buffer_size = args.buffer_size
            #new_buffer_size = old_buffer_size + (args.nx + 1) - (old_buffer_size % (args.nx + 1))
            #args.buffer_size = new_buffer_size
            #print("Replay buffer size changed from {} to {} to align with MARL-style buffer."
                    #.format(old_buffer_size, new_buffer_size))

    #if args.model == 'pg' or args.model == 'reinforce':
        #if args.gamma == 0.0:
            #args.return_style = "myopic"

    model_cls = get_model_class(args.model)
    if args.model == "full":
        emi_cls = ToyBatchGlobalEMI
    else:
        raise Exception()

    obs_adjust = numpy_fn(args.obs_scale)
    action_adjust = numpy_fn(args.action_scale)

    emi = emi_cls(env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)

    # Set up logging.
    start_time = time.localtime()
    if args.log_dir is None:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        default_log_dir = os.path.join("log", args.env, args.model, timestamp)
        args.log_dir = default_log_dir
        print("Using default log directory: {}".format(default_log_dir))
    try:
        os.makedirs(args.log_dir)
    except FileExistsError:
        if args.n:
            raise Exception("Logging directory \"{}\" already exists!.".format(args.log_dir))
        elif not args.y:
            _ignore = input(("(\"{}\" already exists! Hit <Enter> to overwrite and"
                            + " continue, Ctrl-C to stop.)").format(args.log_dir))
        shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)

    # Create symlink for convenience.
    log_link_name = "last"
    if ON_POSIX:
        try:
            if os.path.islink(log_link_name):
                os.unlink(log_link_name)
            os.symlink(args.log_dir, log_link_name, target_is_directory=True)
        except OSError:
            print("Failed to create \"last\" symlink. Continuing without it.")
    else:
        # On Windows, creating a symlink requires admin priveleges, but creating
        # a "junction" does not, even though a junction is just a symlink on directories.
        # I think there may be some support in Python3.8 for this,
        # but we need Python3.7 for Tensorflow 1.15.
        try:
            if os.path.isdir(log_link_name):
                os.rmdir(log_link_name)
            subprocess.run("mklink /J {} {}".format(log_link_name, args.log_dir), shell=True)
        except OSError:
            print("Failed to create \"last\" symlink. Continuing without it.")

    metadata.create_meta_file(args.log_dir, args)

    # Put stable-baselines logs in same directory.
    logger.configure(folder=args.log_dir, format_strs=['stdout', 'log', 'csv'])  # ,tensorboard'
    logger.set_level(logger.DEBUG)  # logger.INFO

    # Call train().
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        train(env, emi, args)
    except KeyboardInterrupt:
        print("Training stopped by interrupt.")
        metadata.log_finish_time(args.log_dir, status="stopped by interrupt")
        sys.exit(0)
    except Exception as e:
        metadata.log_finish_time(args.log_dir, status="stopped by exception: {}".format(type(e).__name__))
        raise  # Re-raise so exception is also printed.

    print("Finished!")
    metadata.log_finish_time(args.log_dir, status="finished cleanly")

    model_file_name = os.path.join(args.log_dir, "model_final")
    emi.save_model(model_file_name)


if __name__ == "__main__":
    main()
