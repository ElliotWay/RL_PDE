import argparse
import os
import shutil
import signal
import sys
import time
from argparse import Namespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

matplotlib.use("Agg")

from stable_baselines import logger

from envs import get_env_arg_parser, build_env
from agents import StandardWENOAgent, StationaryAgent, EqualAgent, MiddleAgent, LeftAgent, RightAgent, RandomAgent
from models.sac import SACBatch
from util import metadata
from util.misc import set_global_seed

def save_convergence_plot(grid_sizes, error, args):
    plt.plot(grid_sizes, error, ls='-', color='k')

    ax = plt.gca()
    ax.set_xlabel("grid size")
    ax.set_xticks(grid_sizes)
    ax.set_ylabel("L2 error")
    ax.set_yticks([0] + error)

    filename = os.path.join(args.log_dir, "convergence.png")
    plt.savefig(filename)
    print('Saved plot to ' + filename + '.')
    plt.close()

def save_error_plot(x_vals, y_vals, labels, args):
    for x, y, label in zip(x_vals, y_vals, labels):
        plt.plot(x, y, ls='-', label=str(label))

    ax = plt.gca()
    ax.set_xlabel("x")
    ax.set_ylabel("|error|")

    ax.set_yscale('log')
    ax.set_ymargin(0.0)
    #extreme_cutoff = 3.0
    #max_not_extreme = max([np.max(y[y < extreme_cutoff]) for y in y_vals])
    #ymax = max_not_extreme*1.05 if max_not_extreme > 0.0 else 0.01
    #ax.set_ylim((None, ymax))

    plt.legend()

    filename = os.path.join(args.log_dir, "convergence_over_x.png")
    plt.savefig(filename)
    print("Saved plot to " + filename + ".")
    plt.close()

def do_test(env, agent, args):
    state = env.reset()

    done = False
    t = 0
    next_update = 0
    NUM_UPDATES = 10
    update_step = 0

    rewards = []
    total_reward = 0

    render_args = {}
    if args.animate:
        render_args["fixed_axes"] = True
        render_args["no_x_borders"] = True
        render_args["show_ghost"] = False
    else:
        render_args["fixed_axes"] = False

    render_args["show_ghost"] = False
    if args.evolution_plot:
        state_record = []

    start_time = time.time()

    while not done:
 
        if t >= update_step:
            print("step = " + str(t))
            if not args.animate:
                env.render(mode="file", **render_args)
                if args.plot_error:
                    env.plot_state(plot_error=True, **render_args)

        if args.animate:
            env.render(mode="file", **render_args)

        if not args.rk4:
            # The agent's policy function takes a batch of states and returns a batch of actions.
            # However, we take that batch of actions and pass it to the environment like a single action.
            actions, _ = agent.predict(state)
            state, reward, done, _ = env.step(actions)
        else:
            for _ in range(4):
                actions, _ = agent.predict(state)
                state, reward, done = env.rk4_step(actions)

        if t >= update_step:
            next_update += 1
            update_step = args.ep_length * (next_update / NUM_UPDATES)
            if not args.animate and args.plot_actions:
                env.render(mode="actions", **render_args)

        if args.animate and args.plot_actions:
            env.render(mode="actions", **render_args)

        rewards.append(reward)
        total_reward += reward

        if args.animate:
            env.render(**render_args)

        t += 1

    end_time = time.time()

    print("step = {} (done)".format(t))

    env.render(mode="file", **render_args)
    if args.plot_error:
        env.plot_state(plot_error=True, **render_args)
    if args.plot_actions:
        env.render(mode="actions", **render_args)
    if args.evolution_plot:
        env.plot_state_evolution(num_states=10, full_true=False, no_true=False)
        if args.plot_error:
            env.plot_state_evolution(num_states=10, plot_error=True)

    print("Test finished in " + str(end_time - start_time) + " seconds.")
    print("Reward: mean = {}, min = {} @ {}, max = {} @ {}".format(
        np.mean(total_reward), np.amin(total_reward), np.argmin(total_reward), np.amax(total_reward), np.argmax(total_reward)))

    error = np.sqrt(env.grid.dx * np.sum(np.square(env.grid.get_real() - env.solution.get_real())))
    print("Final error with solution was {}.".format(error))

    return error


def main():
    parser = argparse.ArgumentParser(
        description="Deploy an existing RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not test and show the environment parameters not listed here.")
    parser.add_argument('--agent', '-a', type=str, default="default",
                        help="Agent to test. Either a file or a string for a standard agent. \"default\" uses standard weno coefficients."
                        "\"none\" doesn't use an agent at all and just plots the true solution (ONLY IMPLEMENTED FOR EVOLUTION PLOTS).")
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
    parser.add_argument('--plot-actions', '--plot_actions', default=False, action='store_true',
                        help="Plot the actions in addition to the state.")
    parser.add_argument('--animate', default=False, action='store_true',
                        help="Enable animation mode. Plot the state at every timestep, and keep the axes fixed across every plot.")
    parser.add_argument('--plot-error', '--plot_error', default=False, action='store_true',
                        help="Plot the error between the agent and the solution. Combines with evolution-plot.")
    parser.add_argument('--evolution-plot', '--evolution_plot', default=False, action='store_true',
                        help="Instead of usual rendering create 'evolution plot' which plots several states on the"
                        + " same plot in increasingly dark color.")
    parser.add_argument('--convergence-plot', '--convergence_plot', default=False, action='store_true',
                        help="Do several runs with different grid sizes to create a convergence plot."
                        " Overrides the --nx argument with 64, 128, 256, and 512, successively."
                        " Sets the --analytical flag.")
    parser.add_argument('--rk4', default=False, action='store_true',
                        help="Use RK4 steps instead of Euler steps. Only available for testing,"
                        + " since the reward during training doesn't make sense.")
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

    set_global_seed(args.seed)

    if args.memoize is None:
        args.memoize = False

    if not args.convergence_plot:
        env = build_env(args.env, args, test=True)
    else:
        args.analytical = True
        CONVERGENCE_PLOT_GRID_RANGE = [64, 128, 256, 512]
        envs = []
        for nx in CONVERGENCE_PLOT_GRID_RANGE:
            args.nx = nx
            envs.append(build_env(args.env, args, test=True))

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

    metadata.create_meta_file(args.log_dir, args)

    # Put stable-baselines logs in same directory.
    logger.configure(folder=args.log_dir, format_strs=['stdout'])  # ,tensorboard'
    logger.set_level(logger.DEBUG)  # logger.INFO

    if args.env.startswith("weno"):
        mode = "weno"
    elif args.env.startswith("split_flux"):
        mode = "split_flux"
    elif args.env.startswith("flux"):
        mode = "flux"
    else:
        mode = "n/a"

    # TODO: create standard agent lookup function in agents.py.
    if args.agent == "default" or args.agent == "none":
        agent = StandardWENOAgent(order=args.order, mode=mode)
    elif args.agent == "stationary":
        agent = StationaryAgent(order=args.order, mode=mode)
    elif args.agent == "equal":
        agent = EqualAgent(order=args.order)
    elif args.agent == "middle":
        agent = MiddleAgent(order=args.order)
    elif args.agent == "left":
        agent = LeftAgent(order=args.order)
    elif args.agent == "right":
        agent = RightAgent(order=args.order)
    elif args.agent == "random":
        agent = RandomAgent(order=args.order, mode=mode)
    else:
        if args.algo == "sac":
            agent = SACBatch.load(args.agent)
        else:
            print("Algorithm {} not recognized.".format(args.algo))
 
    # Create symlink for convenience. (Do this after loading the agent in case we are loading from last.)
    try:
        log_link_name = "last"
        if os.path.islink(log_link_name):
            os.unlink(log_link_name)
        os.symlink(args.log_dir, log_link_name, target_is_directory=True)
    except OSError:
        print("Failed to create \"last\" symlink. Maybe you're a non-admin on a Windows machine?")

    # Run test.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        if not args.convergence_plot:
            do_test(env, agent, args)
        else:
            error = []
            x_vals = []
            error_vals = []
            for nx, env in zip(CONVERGENCE_PLOT_GRID_RANGE, envs):
                args.nx = nx

                sub_dir = os.path.join(args.log_dir, "nx_{}".format(nx))
                os.makedirs(sub_dir)
                logger.configure(folder=sub_dir, format_strs=['stdout'])  # ,tensorboard'

                error.append(do_test(env, agent, args))

                x_vals.append(env.grid.real_x)
                error_vals.append(np.abs(env.grid.get_real() - env.solution.get_real()))

            save_convergence_plot(CONVERGENCE_PLOT_GRID_RANGE, error, args)
            save_error_plot(x_vals, error_vals, CONVERGENCE_PLOT_GRID_RANGE, args)
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
