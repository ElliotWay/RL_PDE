import argparse
import os
import shutil
import signal
import sys
import time
from argparse import Namespace

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib

matplotlib.use("Agg")

from stable_baselines import logger

from burgers import Grid1d
from burgers_env import WENOBurgersEnv

from models.sac import SACBatch
from models.sac import LnScaledMlpPolicy
from util import metadata


# TODO put this in separate environment file
def build_env(args):
    if args.env == "weno_burgers":
        num_ghosts = args.order + 1
        grid = Grid1d(nx=args.nx, ng=num_ghosts, xmin=args.xmin, xmax=args.xmax, bc=args.boundary)
        env = WENOBurgersEnv(grid=grid, C=args.C, weno_order=args.order, episode_length=args.ep_length,
                             init_type=args.init_type)
    else:
        print("Unrecognized environment type: \"" + str(args.env) + "\".")
        sys.exit(0)

    return env


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--show-hidden', default=False, action='store_true',
                        help="Do not train and show the hidden parameters not listed here.")
    parser.add_argument('--algo', '-a', type=str, default="sac",
                        help="Algorithm to train with. Currently only \"sac\" accepted.")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--log-dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is log/env/algo/timestamp.")
    parser.add_argument('--ep-length', type=int, default=300,
                        help="Number of timesteps in an episode.")
    parser.add_argument('--total_timesteps', type=int, default=int(1e5),
                        help="Total number of timesteps to train.")
    parser.add_argument('--log_freq', type=int, default=10,
                        help="Number of timesteps to wait between logging information.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--render', type=str, default="file",
                        help="How to render output. Options are file, human, and none.")

    main_args, rest = parser.parse_known_args()

    sub_parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub_parser.add_argument('--xmin', type=float, default=0.0)
    sub_parser.add_argument('--xmax', type=float, default=1.0)
    sub_parser.add_argument('--nx', type=int, default=128,
                            help="Number of cells into which to discretize x dimension.")
    sub_parser.add_argument('--order', type=int, default=3,
                            help="Order of WENO approximation (assuming using WENO environment or agent).")

    sub_parser.add_argument('--fixed-timesteps', dest='fixed_timesteps', action='store_true',
                            help="TODO: not implemented!")
    sub_parser.add_argument('--variable-timesteps', dest='fixed_timesteps', action='store_false')
    sub_parser.set_defaults(fixed_timesteps=True)

    sub_parser.add_argument('--timestep', type=float, default=0.0005,
                            help="Set fixed timestep length. TODO: not implemented!")
    sub_parser.add_argument('--C', type=float, default=0.1,
                            help="Constant used in choosing variable timestep.")

    sub_parser.add_argument('--init_type', type=str, default="sine",
                            help="Shape of the initial state.")
    sub_parser.add_argument('--boundary', '--bc', type=str, default="periodic")

    sub_args, rest = sub_parser.parse_known_args(rest)

    args = Namespace(**vars(main_args), **vars(sub_args))

    if len(rest) > 0:
        print("Ignoring unrecognized arguments: " + " ".join(rest))
        print()

    if args.show_hidden:
        sub_parser.print_help()
        return

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    env = build_env(args)

    if args.algo == "sac":
        policy_kwargs = dict(layers=[32, 32])
        model = SACBatch(LnScaledMlpPolicy, env, policy_kwargs=policy_kwargs, learning_rate=3e-4, buffer_size=50000,
                 learning_starts=100, batch_size=64, verbose=1, tensorboard_log="./log/weno_burgers/tensorboard")
    else:
        print("Algorithm type \"" + str(args.algo) + "\" not implemented.")

    if args.render == "none":
        args.render = None

    # Set up logging.
    start_time = time.localtime()
    if args.log_dir is None:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        default_log_dir = os.path.join("log", args.env, args.algo, timestamp)
        args.log_dir = default_log_dir
        print("Using default log directory: {}".format(default_log_dir))
    try:
        os.makedirs(args.log_dir)
    except FileExistsError:
        _ignore = input("(\"{}\" already exists! Hit <Enter> to overwrite and"
                        + " continue, Ctrl-C to stop.)".format(args.log_dir))
        shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)

    metadata.create_meta_file(args.log_dir, args)

    # Put stable-baselines logs in same directory.
    logger.configure(folder=args.log_dir, format_strs=['stdout', 'log', 'csv'])  # ,tensorboard'
    logger.set_level(logger.DEBUG)  # logger.INFO

    # Call model.learn().
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        model.learn(total_timesteps=args.total_timesteps, log_interval=args.log_freq, render=args.render)
    except KeyboardInterrupt:
        print("Training stopped by interrupt.")
        metadata.log_finish_time(args.log_dir, status="stopped by interrupt")
        sys.exit(0)
    except Exception as e:
        metadata.log_finish_time(args.log_dir, status="stopped by exception: {}".format(type(e).__name__))
        raise  # Re-raise so excption is also printed.

    print("Finished!")
    metadata.log_finish_time(args.log_dir, status="finished cleanly")

    model_file_name = os.path.join(args.log_dir, "model")
    model.save(model_file_name)


if __name__ == "__main__":
    main()
