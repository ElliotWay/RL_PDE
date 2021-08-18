import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #Block GPU for now.
import argparse
from argparse import Namespace
import shutil
import signal
import sys
import time
import subprocess
import gc # manual garbage collection

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines import logger

from rl_pde.run import rollout
from rl_pde.emi import BatchEMI, StandardEMI, TestEMI, DimensionalAdapterEMI
from rl_pde.agents import get_agent, ExtendAgent2D
from envs import builder as env_builder
from envs import AbstractBurgersEnv
from envs import Plottable1DEnv, Plottable2DEnv
from models import get_model_arg_parser
from models import SACModel, PolicyGradientModel, TestModel
from util import metadata
from util.function_dict import numpy_fn
from util.lookup import get_model_class, get_emi_class, get_model_dims
from util.misc import set_global_seed
from util.misc import human_readable_time_delta

ON_POSIX = 'posix' in sys.builtin_module_names

def save_convergence_plot(grid_sizes, error, args):
    plt.plot(grid_sizes, error, ls='-', color='k')

    ax = plt.gca()
    ax.set_xlabel("grid size")
    #ax.set_xticks(grid_sizes)
    ax.set_xscale('log')
    ax.set_ylabel("L2 error")
    #ax.set_yticks([0] + error)
    ax.set_yscale('log')

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
    if args.animate:
        NUM_UPDATES = 50
    else:
        NUM_UPDATES = 10
    update_count = 0
    next_update = 0

    render_args = {}
    if args.animate:
        render_args["fixed_axes"] = True
        render_args["no_borders"] = True
        render_args["show_ghost"] = False
    else:
        render_args["fixed_axes"] = False

    # For now, always hide the ghost cells.
    render_args["show_ghost"] = False

    next_update = 0
    def every_step(step):
        # Write to variables in parent scope.
        nonlocal next_update
        nonlocal update_count
        #if args.animate or step == next_update:
        if step == next_update:
            if step == next_update:
                print("step = {}".format(step))

            env.plot_state(**render_args)
            if args.plot_error:
                env.plot_state(plot_error=True, **render_args)
        if (args.animate or step == next_update + 1) and args.plot_actions:
            env.plot_action(**render_args)
        if step == next_update + 1:
            update_count += 1
            next_update = int(args.ep_length * (update_count / NUM_UPDATES))

    start_time = time.time()
    _, _, rewards, _, _ = rollout(env, agent,
                  rk4=args.rk4, deterministic=True, every_step_hook=every_step)
    end_time = time.time()

    print("step = {} (done)".format(env.steps))

    env.plot_state(**render_args)
    if args.plot_error:
        env.plot_state(plot_error=True, **render_args)
    if args.plot_actions:
        env.plot_action(**render_args)
    print("Test finished in {}.".format(human_readable_time_delta(time.time() - start_time)))
 
    if env.dimensions == 1:
        total_reward = np.sum(rewards, axis=0)
        print("Reward: mean = {:g}, min = {:g} @ {}, max = {:g} @ {}".format(
                np.mean(total_reward),
                np.amin(total_reward), np.argmin(total_reward),
                np.amax(total_reward), np.argmax(total_reward)))
    else:
        dim_names = ['x', 'y']

        rewards_reshaped = [np.array([step_reward[i] for step_reward in rewards]) for i in
                range(len(rewards[0]))]
        total_reward = [np.sum(reward_dim, axis=0) for reward_dim in rewards_reshaped]
        avg_reward = np.mean([np.mean(reward_dim) for reward_dim in total_reward])

        min_reward = [np.amin(reward_dim) for reward_dim in total_reward]
        dim_with_min = np.argmin(min_reward)
        actual_min = min_reward[dim_with_min]
        actual_argmin = np.unravel_index(np.argmin(total_reward[dim_with_min]),
                            shape=total_reward[dim_with_min].shape)

        max_reward = [np.amax(reward_dim) for reward_dim in total_reward]
        dim_with_max = np.argmax(max_reward)
        actual_max = max_reward[dim_with_max]
        actual_argmax = np.unravel_index(np.argmax(total_reward[dim_with_max]),
                            shape=total_reward[dim_with_max].shape)

        print("Reward: mean = {:g}, min = {:g} @ {} in {}, max = {:g} @ {} in {}".format(
                avg_reward,
                actual_min, actual_argmin, dim_names[dim_with_min],
                actual_max, actual_argmax, dim_names[dim_with_max]))

    error = env.compute_l2_error()
    print("Final error with solution was {}.".format(error))

    if args.evolution_plot:
        if isinstance(env, Plottable1DEnv):
            env.plot_state_evolution(num_states=10, full_true=False, no_true=False, plot_weno=False)
            if args.plot_error:
                env.plot_state_evolution(num_states=10, plot_error=True)
        elif isinstance(env, Plottable2DEnv):
            env.plot_state_evolution(num_frames=20)
        else:
            raise Exception()

    return error

def main():
    parser = argparse.ArgumentParser(
        description="Deploy an existing RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not test and show the environment parameters not listed here.")
    parser.add_argument('--agent', '-a', type=str, default="default",
                        help="Agent to test. Either a file or a string for a standard agent."
                        + " Parameters are loaded from 'meta.txt' in the same directory as the"
                        + " agent file, but can be overriden."
                        + " 'default' uses standard weno coefficients. 'none' forces no agent and"
                        + " only plots the true solution (ONLY IMPLEMENTED FOR EVOLUTION PLOTS).")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--model', type=str, default='sac',
                        help="Type of model to be loaded. (Overrides the meta file.)")
    parser.add_argument('--emi', type=str, default='batch',
                        help="Environment-model interface. (Overrides the meta file.)")
    parser.add_argument('--obs-scale', '--obs_scale', type=str, default='z_score_last',
                        help="Adjustment function to observation. Compute Z score along the last"
                        + " dimension (the stencil) with 'z_score_last', the Z score along every"
                        + " dimension with 'z_score_all', or leave them the same with 'none'.")
    parser.add_argument('--action-scale', '--action_scale', type=str, default=None,
                        help="Adjustment function to action. Default depends on environment."
                        + " 'softmax' computes softmax, 'rescale_from_tanh' scales to [0,1] then"
                        + " divides by the sum of the weights, 'none' does nothing.")
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

    env_arg_parser = env_builder.get_env_arg_parser()
    env_args, rest = env_arg_parser.parse_known_args(rest)
    env_builder.set_contingent_env_defaults(main_args, env_args)

    # run_test.py has model arguments that can be overidden, if desired,
    # but are intended to be loaded from a meta file.
    model_arg_parser = get_model_arg_parser()
    model_args, rest = model_arg_parser.parse_known_args(rest)

    internal_args = {}
    internal_args['total_episodes'] = 1000

    if main_args.env.startswith("weno"):
        mode = "weno"
    elif main_args.env.startswith("split_flux"):
        mode = "split_flux"
    elif main_args.env.startswith("flux"):
        mode = "flux"
    else:
        mode = "n/a"
    internal_args['mode'] = mode

    args = Namespace(**vars(main_args), **vars(env_args), **vars(model_args), **internal_args)

    if len(rest) > 0:
        print("Unrecognized arguments: " + " ".join(rest))
        sys.exit(0)

    if args.help_env:
        env_arg_parser.print_help()
        sys.exit(0)

    dims = env_builder.env_dimensions(args.env)

    # Load basic agent, if one was specified.
    agent = get_agent(args.agent, order=args.order, mode=mode, dimensions=dims)
    # If the returned agent is None, assume it is the file name of a real agent.
    # If a real agent is specified, load parameters from its meta file.
    # This only overrides arguments that were not explicitly specified.
    #TODO: Find more reliable way of doing this so we are robust to argument changes.
    if agent is None:
        if not os.path.isfile(args.agent):
            raise Exception("Agent file \"{}\" not found.".format(args.agent))

        model_file = os.path.abspath(args.agent)
        model_directory = os.path.dirname(model_file)
        meta_file = os.path.join(model_directory, "meta.txt")
        if not os.path.isfile(meta_file):
            raise Exception("Meta file \"{}\" for agent not found.".format(meta_file))

        metadata.load_to_namespace(meta_file, args, ignore_list=['log_dir', 'ep_length'])
    #env_builder.set_contingent_env_defaults(args, args)

    set_global_seed(args.seed)

    if args.memoize is None:
        args.memoize = False

    if not args.convergence_plot:
        env = env_builder.build_env(args.env, args, test=True)
    else:
        if dims > 1:
            raise NotImplementedError("Convergence plots not adapted to 2D yet.")
        #args.analytical = True # Compare to analytical solution (preferred)
        args.analytical = False # Compare to WENO (necessary when WENO isn't accurate either)
        if args.reward_mode is not None and 'one-step' in args.reward_mode:
            print("TODO: compute error with analytical solution when using one-step error.")
            print("(Currently forcing the error to change to full instead.)")
            args.reward_mode = 'full'
        CONVERGENCE_PLOT_GRID_RANGE = [64, 128, 256, 512]#, 1024, 2048, 4096, 8192]
        envs = []
        env_args = []
        for nx in CONVERGENCE_PLOT_GRID_RANGE:
            #TODO Handle variable timesteps?
            if args.C is None:
                args.C = 0.1
            eval_env_args = Namespace(**vars(args))
            time_max = args.timestep * args.ep_length
            _, dt, ep_length = AbstractBurgersEnv.fill_default_time_vs_space(
                xmin=args.xmin, xmax=args.xmax, nx=nx,
                dt=None, C=args.C, ep_length=None, time_max=time_max)
            eval_env_args.nx = nx
            eval_env_args.timestep = dt
            eval_env_args.ep_length = ep_length

            envs.append(env_builder.build_env(eval_env_args.env, eval_env_args, test=True))
            env_args.append(eval_env_args)
        env = envs[0]

    if agent is None:
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
        else:
            emi = emi_cls(env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)

        emi.load_model(args.agent)
        agent = emi.get_policy()
 
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

    # Create symlink for convenience. (Do this after loading the agent in case we are loading from last.)
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

    # Run test.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        if not args.convergence_plot:
            do_test(env, agent, args)
        else:
            error = []
            x_vals = []
            error_vals = []
            for env, env_args in zip(envs, env_args):

                sub_dir = os.path.join(args.log_dir, "nx_{}".format(env_args.nx))
                os.makedirs(sub_dir)
                logger.configure(folder=sub_dir, format_strs=['stdout'])  # ,tensorboard'

                print("Convergence run with {} grid cells and {}s timesteps ({}s total).".format(
                    env_args.nx, env_args.timestep, env_args.ep_length*env_args.timestep))
                l2_error = do_test(env, agent, env_args)
                error.append(l2_error)

                x_vals.append(env.grid.real_x)
                error_vals.append(np.abs(env.grid.get_real() - env.solution.get_real()))

                # Some of these trajectories can get big, so clean them out when we don't need
                # them.
                #TODO Make sure it's actually freeing them.
                env.close()
                gc.collect()

            envs = []

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
