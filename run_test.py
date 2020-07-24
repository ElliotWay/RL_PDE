import argparse
import os
import shutil
import signal
import sys
import time
from argparse import Namespace

import matplotlib
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

matplotlib.use("Agg")

from stable_baselines import logger

from envs import build_env
from envs import get_env_arg_parser
from agents import StandardWENOAgent, StationaryAgent, EqualAgent, MiddleAgent, LeftAgent, RightAgent, RandomAgent
from models.sac import SACBatch
from util import metadata


def do_test(env, agent, args):
    if args.plot_weights:
        env.set_record_weights(True)

    state = env.reset()

    done = False
    t = 0
    next_update = 0

    rewards = []
    total_reward = 0

    start_time = time.time()

    while not done:

        # The agent's policy function takes a batch of states and returns a batch of actions.
        # However, we take that batch of actions and pass it to the environment like a single action.
        actions, _ = agent.predict(state)
        state, reward, done, info = env.step(actions)

        rewards.append(reward)
        total_reward += reward

        if args.animate:
            env.render(fixed_axes=True)

        if t >= args.ep_length * (next_update / 10):
            print("step = " + str(t))
            next_update += 1
            if not args.animate:
                env.render(fixed_axes=False)
            if args.plot_weights:
                env.plot_weights()

            # TODO: log other information, probably

        t += 1

    end_time = time.time()

    print("Test finished in " + str(end_time - start_time) + " seconds.")
    print("Total reward was " + str(total_reward) + ".")

    if args.animate:
        env.render(fixed_axes=True)
    else:
        env.render(fixed_axes=False)

    if args.plot_weights:
        env.plot_weights()


def main():
    parser = argparse.ArgumentParser(
        description="Deploy an existing RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not test and show the environment parameters not listed here.")
    parser.add_argument('--agent', '-a', type=str, default="default",
                        help="Agent to test. Either a file (unimplemented) or a string for a standard agent. \"default\" uses standard weno coefficients.")
    parser.add_argument('--algo', type=str, default="sac",
                        help="Algorithm used to create the agent. Unfortunately necessary to open a model file.")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--log-dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is test/env/agent/timestamp.")
    parser.add_argument('--ep-length', type=int, default=500,
                        help="Number of timesteps in an episode.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--plot-weights', default=False, action='store_true',
                        help="Plot a comparison of weights across the episode instead of plotting the state.")
    parser.add_argument('--animate', default=False, action='store_true',
                        help="Enable animation mode. Plot the state at every timestep, and keep the axes fixed across every plot.")
    parser.add_argument('-y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files. Useful for scripts. Overrides the -y option.")

    main_args, rest = parser.parse_known_args()

    # TODO: add system to automatically read parameters from model's meta file.


    env_arg_parser = get_env_arg_parser()
    env_args, rest = env_arg_parser.parse_known_args(rest)

    args = Namespace(**vars(main_args), **vars(env_args))

    if len(rest) > 0:
        print("Unrecognized arguments: " + " ".join(rest))
        sys.exit(0)

    if args.help_env:
        env_arg_parser.print_help()
        sys.exit(0)

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    env = build_env(args.env, args)

    # Set up logging.
    start_time = time.localtime()
    if args.log_dir is None:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        default_log_dir = os.path.join("test", args.env, args.agent, timestamp)
        args.log_dir = default_log_dir
        print("Using default log directory: {}".format(default_log_dir))
    try:
        os.makedirs(args.log_dir)
    except FileExistsError:
        if args.n:
            raise Exception("Logging directory \"{}\" already exists!.".format(args.log_dir))
        elif args.y:
            print("\"{}\" already exists, overwriting...".format(args.log_dir))
        else:
            _ignore = input(("\"{}\" already exists! Hit <Enter> to overwrite and"
                             + " continue, Ctrl-C to stop.").format(args.log_dir))
        shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)

    # Create symlink for convenience.
    try:
        log_link_name = "last"
        if os.path.islink(log_link_name):
            os.unlink(log_link_name)
        os.symlink(args.log_dir, log_link_name, target_is_directory=True)
    except OSError:
        print("Failed to create \"last\" symlink. Maybe you're a non-admin on a Windows machine?")

    metadata.create_meta_file(args.log_dir, args)

    # Put stable-baselines logs in same directory.
    logger.configure(folder=args.log_dir, format_strs=['stdout'])  # ,tensorboard'
    logger.set_level(logger.DEBUG)  # logger.INFO

    if args.agent == "default":
        agent = StandardWENOAgent(order=args.order)
    elif args.agent == "stationary":
        agent = StationaryAgent(order=args.order)
    elif args.agent == "equal":
        agent = EqualAgent(order=args.order)
    elif args.agent == "middle":
        agent = MiddleAgent(order=args.order)
    elif args.agent == "left":
        agent = LeftAgent(order=args.order)
    elif args.agent == "right":
        agent = RightAgent(order=args.order)
    elif args.agent == "random":
        agent = RandomAgent(order=args.order)
    else:
        if args.algo == "sac":
            agent = SACBatch.load(args.agent)
        else:
            print("Algorithm {} not recognized.".format(args.algo))
        

    # Run test.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        do_test(env, agent, args)
    except KeyboardInterrupt:
        print("Test stopped by interrupt.")
        metadata.log_finish_time(args.log_dir, status="stopped by interrupt")
        sys.exit(0)
    except Exception as e:
        metadata.log_finish_time(args.log_dir, status="stopped by exception: {}".format(type(e).__name__))
        raise  # Re-raise so exception is also printed.

    print("Done.")
    metadata.log_finish_time(args.log_dir, status="finished cleanly")


if __name__ == "__main__":
    main()
