import argparse
import os
import shutil
import signal
import sys
import time
from argparse import Namespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

from stable_baselines import logger

from burgers import Grid1d
from burgers_env import WENOBurgersEnv
from weno_agent import StandardWENOAgent
from stationary_agent import StationaryAgent
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
        actions = agent.predict(state)
        state, reward, done, info = env.step(actions)

        rewards.append(reward)
        total_reward += reward

        if t >= args.ep_length * (next_update / 10):
            print("t = " + str(t))
            next_update += 1
            if not args.plot_weights:
                env.render()

            # TODO: log other information, probably

        t += 1

    end_time = time.time()

    print("Test finished in " + str(end_time - start_time) + " seconds.")
    print("Total reward was " + str(total_reward) + ".")

    if not args.plot_weights:
        env.render()

    if args.plot_weights:
        env.plot_weights()


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
        description="Deploy an existing RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--show-hidden', default=False, action='store_true',
                        help="Do not test and show the hidden parameters not listed here.")
    parser.add_argument('--agent', '-a', type=str, default="default",
                        help="Agent to test. Either a file (unimplemented) or a string for a standard agent. \"default\" uses standard weno coefficients.")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--log-dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is test/env/agent/timestamp.")
    parser.add_argument('--ep-length', type=int, default=300,
                        help="Number of timesteps in an episode.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--plot-weights', default=False, action='store_true',
                        help="Plot a comparison of weights across the episode instead of plotting the state.")

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

    sub_parser.add_argument('--init_type', '--init-type', type=str, default="sine",
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

    env = build_env(args)

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
        _ignore = input(("\"{}\" already exists! Hit <Enter> to overwrite and"
                         + " continue, Ctrl-C to stop.").format(args.log_dir))
        shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)

    metadata.create_meta_file(args.log_dir, args)

    # Put stable-baselines logs in same directory.
    logger.configure(folder=args.log_dir, format_strs=['stdout'])  # ,tensorboard'
    logger.set_level(logger.DEBUG)  # logger.INFO

    # TODO build agent from file
    if args.agent == "default":
        agent = StandardWENOAgent(order=args.order)
    elif args.agent == "stationary":
        agent = StationaryAgent(order=args.order)

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
