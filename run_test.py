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

import yaml
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines import logger

from rl_pde.run import rollout
from rl_pde.emi import BatchEMI, StandardEMI, TestEMI, DimensionalAdapterEMI, VectorAdapterEMI
from rl_pde.agents import get_agent, ExtendAgent2D
from envs import builder as env_builder
from envs import AbstractPDEEnv
from envs import Plottable1DEnv, Plottable2DEnv
from models import builder as model_builder
from models import SACModel, PolicyGradientModel, TestModel
from util import plots
from util import metadata
from util.param_manager import ArgTreeManager
from util.function_dict import numpy_fn
from util.lookup import get_model_class, get_emi_class, get_model_dims
from util.misc import set_global_seed
from util.misc import human_readable_time_delta
from util.misc import soft_link_directories

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

    dt_list = []
    def every_step(step):
        # Write to variables in parent scope.
        nonlocal next_update
        nonlocal update_count

        #if args.animate or step == next_update:
        if step == next_update:
            if step == next_update:
                print(f"step = {env.steps}, t = {env.t:.4f}")

            if 'plot' in args.output_mode:
                env.plot_state(**render_args)
            if 'csv' in args.output_mode:
                env.save_state()

            if args.plot_error:
                if 'plot' in args.output_mode:
                    env.plot_state(plot_error=True, **render_args)
                if 'csv' in args.output_mode:
                    env.save_state(use_error=True)
        if (args.animate or step == next_update + 1) and args.plot_actions:
            if 'plot' in args.output_mode:
                env.plot_action(**render_args)
            if 'csv' in args.output_mode:
                env.save_action()
        if step == next_update + 1:
            update_count += 1
            next_update = int(args.e.ep_length * (update_count / NUM_UPDATES))

    start_time = time.time()
    _, _, rewards, _, _ = rollout(env, agent,
                  deterministic=True, every_step_hook=every_step)
    end_time = time.time()

    print(f"step = {env.steps}, t = {env.t:.4f} (done)")

    if 'plot' in args.output_mode:
        env.plot_state(**render_args)
        if args.plot_error:
            env.plot_state(plot_error=True, **render_args)
        if args.plot_actions:
            env.plot_action(**render_args)
    if 'csv' in args.output_mode:
        env.save_state()
        if args.plot_error:
            env.save_state(use_error=True)
        if args.plot_actions:
            env.save_action()

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
        if 'plot' in args.output_mode:
            if isinstance(env, Plottable1DEnv):
                env.plot_state_evolution(num_states=10, full_true=False, no_true=False, plot_weno=False)
                if args.plot_error:
                    env.plot_state_evolution(num_states=10, plot_error=True)
            elif isinstance(env, Plottable2DEnv):
                env.plot_state_evolution(num_frames=20)
            else:
                raise Exception()
        if 'csv' in args.output_mode:
            print("CSV format for evolution is not implemented.")

    return error

def main():
    arg_manager = ArgTreeManager()
    parser = argparse.ArgumentParser(
        description="Deploy an existing RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not test and show the environment parameters not listed here.")
    parser.add_argument('--agent', '-a', type=str, default="default",
                        help="Agent to test. Either a file or a string for a standard agent."
                        + " Parameters are loaded from 'meta.[yaml|txt]' in the same directory as the"
                        + " agent file, but can be overriden."
                        + " 'default' uses standard weno coefficients. 'none' forces no agent and"
                        + " only plots the true solution (ONLY IMPLEMENTED FOR EVOLUTION PLOTS).")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--model', type=str, default=None,
                        help="Type of model to be loaded. (Overrides the meta file.)")
    parser.add_argument('--emi', type=str, default=None,
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
    parser.add_argument('--convergence-plot', '--convergence_plot', nargs='*', type=int,
                        default=None,
                        help="Do several runs with different grid sizes to create a convergence plot."
                        + " Overrides the --num-cells parameter and sets the --analytical flag."
                        + " Use e.g. '--convergence-plot' to use the default grid sizes."
                        + " Use e.g. '--convergence-plot A B C D' to specify your own.")
    parser.add_argument('--output-mode', '--output_mode', default=['plot'], nargs='+',
                        help="Type of output from the test. Default 'plot' creates the usual plot"
                        + " files. 'csv' puts the data that would be used for a plot in a csv"
                        + " file. Currently 'csv' is not implemented for evolution plots."
                        + " Multiple modes can be used at once, e.g. --output-mode plot csv.")
    parser.add_argument('--repeat', type=str, default=None,
                        help="Load all of the parameters from a previous test's meta file to run a"
                        + " similar or identical test. Explicitly passed parameters override"
                        + " loaded paramters.")
    parser.add_argument('-y', '--y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', '--n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files. Useful for scripts. Overrides the -y option.")

    arg_manager.set_parser(parser)
    env_arg_manager = arg_manager.create_child("e", long_name="Environment Parameters")
    env_arg_manager.set_parser(env_builder.get_env_arg_parser())
    # Testing has 'model parameters', but these are really intended to be loaded from a file.
    model_arg_manager = arg_manager.create_child("m", long_name="Model Parameters")
    model_arg_manager.set_parser(lambda args: model_builder.get_model_arg_parser(args.model))

    args, rest = arg_manager.parse_known_args()

    if args.help_env:
        env_arg_manager.print_help()
        sys.exit(0)

    if len(rest) > 0:
        print("Unrecognized arguments: " + " ".join(rest))
        sys.exit(0)

    if args.repeat is not None:
        _, extension = os.path.splitext(args.repeat)

        if extension == '.yaml':
            open_file = open(args.repeat, 'r')
            args_dict = yaml.safe_load(open_file)
            open_file.close()
            arg_manager.load_from_dict(args_dict)
            args = arg_manager.args
        else:
            # Original meta.txt format.
            metadata.load_to_namespace(args.repeat, arg_manager)

    for mode in args.output_mode:
        if mode not in ['plot', 'csv']:
            raise Exception(f"{mode} output mode not recognized.")

    # Convergence plots have different defaults.
    if args.convergence_plot is not None:
        if not arg_manager.check_explicit('e.init_type'):
            args.e.init_type = 'gaussian'
            args.e.time_max = 0.05
            args.e.C = 0.5
            print("Convergence Plot default environment loaded.")
            print("(Gaussian, t_max = 0.05, C = 0.5)")
            # Mark these arguments as explicitly specified so cannot be overridden
            # by a loaded agent.
            # (Do we need to do this? We do for the old system that loads all the parameters, but
            # the new system only loads the model parameters and a few others, not the environment
            # parameters. Does it make more sense to leave them as implicit defaults so they can be
            # loaded by an intentionally loaded environment file?)
            arg_manager.set_explicit('e.init_type', 'e.time_max', 'e.C')
            # The old way to mark explicit. Remove if we don't need backwards compatability with
            # old meta files.
            sys.argv += ['--init-type', 'gaussian', '--time-max', '0.05', '--C', '0.5']
        if not arg_manager.check_explicit('e.rk'):
            args.e.rk = 'rk4'
            arg_manager.set_explicit('e.rk')
            sys.argv += ['--rk', '4']
        if not arg_manager.check_explicit('e.fixed_timesteps'):
            args.e.fixed_timesteps = False
            arg_manager.set_explicit('e.fixed_timesteps')
            sys.argv += ['--variable-timesteps']

    env_builder.set_contingent_env_defaults(args, args.e, test=True)
    model_builder.set_contingent_model_defaults(args, args.m, test=True)

    env_action_type = env_builder.env_action_type(args.env)
    dims = env_builder.env_dimensions(args.env)

    # Load basic agent, if one was specified.
    agent = get_agent(args.agent, order=args.e.order, action_type=env_action_type, dimensions=dims)
    # If the returned agent is None, assume it is the file name of a real agent.
    # If a real agent is specified, load the model parameters from its meta file.
    # This only overrides arguments that were not explicitly specified.
    if agent is None:
        if not os.path.isfile(args.agent):
            raise Exception("Agent file \"{}\" not found.".format(args.agent))

        model_file = os.path.abspath(args.agent)
        model_directory = os.path.dirname(model_file)
        meta_file = os.path.join(model_directory, metadata.META_FILE_NAME)
        if os.path.isfile(meta_file):
            open_file = open(meta_file, 'r')
            args_dict = yaml.safe_load(open_file)
            open_file.close()

            # action scale and obs_scale ought to be model parameters.
            arg_manager.load_keys(args_dict, ['model', 'emi', 'action_scale', 'obs_scale'])
            
            model_arg_manager.load_from_dict(args_dict['m'])
        else:
            meta_file = os.path.join(model_directory, metadata.OLD_META_FILE_NAME)
            if not os.path.isfile(meta_file):
                raise Exception("Meta file \"{}\" for agent not found.".format(meta_file))

            metadata.load_to_namespace(meta_file, arg_manager,
                    ignore_list=['log_dir', 'ep_length', 'time_max', 'timestep',
                                'num_cells', 'min_value', 'max_value', 'C', 'fixed_timesteps',
                                'reward_mode'])

    set_global_seed(args.seed)

    if args.convergence_plot is None:
        env = env_builder.build_env(args.env, args.e, test=True)
    else:
        env_manager_copy = env_arg_manager.copy()
        env_args = env_manager_copy.args
        env_args.analytical = True # Compare to analytical solution (preferred)
        #env_args.analytical = False # Compare to WENO (necessary when WENO isn't accurate either)
        if env_args.reward_mode is not None and 'one-step' in env_args.reward_mode:
            print("Reward mode switched to 'full' instead of 'one-step' for convergence plots.")
            env_args.reward_mode = env_args.reward_mode.replace('one-step', 'full')
        if len(args.convergence_plot) > 0:
            convergence_grid_range = args.convergence_plot
        elif dims == 1:
            convergence_grid_range = [64, 81, 108, 128, 144, 192, 256]
            #convergence_grid_range = [64, 128, 256, 512]#, 1024, 2048, 4096, 8192]
            #convergence_grid_range = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
            #convergence_grid_range = (2**np.linspace(6.0, 8.0, 50)).astype(np.int)
        elif dims == 2:
            convergence_grid_range = [32, 64, 128, 256]

        # Set ep_length and timestep based on number of cells and time_max.
        env_args.ep_length = None
        env_args.timestep = None
        if env_args.C is None:
            env_args.C = 0.1

        conv_envs = []
        conv_env_args = []
        for nx in convergence_grid_range:
            specific_env_copy = env_manager_copy.copy()
            env_args = specific_env_copy.args
            env_args.num_cells = nx

            env_builder.set_contingent_env_defaults(args, env_args, test=True)

            conv_envs.append(env_builder.build_env(args.env, env_args, test=True))
            conv_env_args.append(env_args)
        env = conv_envs[0]

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
        elif args.env == 'weno_euler':
            emi = VectorAdapterEMI(emi_cls, env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)
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

    meta_file = metadata.MetaFile(args.log_dir, arg_manager)
    meta_file.write_new()

    # Put stable-baselines logs in same directory.
    logger.configure(folder=args.log_dir, format_strs=['csv', 'log'])
    logger.set_level(logger.DEBUG)  # logger.INFO
    outer_logger = logger.Logger.CURRENT

    # Create symlink for convenience. (Do this after loading the agent in case we are loading from last.)
    log_link_name = "last"
    error = soft_link_directories(args.log_dir, log_link_name, safe=True)
    if error:
        print("Failed to create \"last\" symlink. Continuing without it.")

    # Run test.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        if args.convergence_plot is None:
            do_test(env, agent, args)
        else:
            convergence_start_time = time.time()
            error = []
            x_vals = []
            error_vals = []
            for env, env_args in zip(conv_envs, conv_env_args):
                nx = env_args.num_cells[0] if dims > 1 else env_args.num_cells

                sub_dir = os.path.join(args.log_dir, "nx_{}".format(nx))
                os.makedirs(sub_dir)
                logger.configure(folder=sub_dir, format_strs=['stdout'])  # ,tensorboard'

                if args.e.fixed_timesteps:
                    print(("Convergence run with {} grid cells and {}s timesteps ({}s total).")
                            .format(env_args.num_cells, env_args.timestep,
                                env_args.ep_length*env_args.timestep))
                else:
                    print(("Convergence run with {} grid cells and dynamic timesteps ({}s total).")
                            .format(env_args.num_cells, env_args.time_max))
                l2_error = do_test(env, agent, args)
                error.append(l2_error)

                if dims == 1:
                    x_vals.append(env.grid.real_x)
                    error_vals.append(np.abs(env.grid.get_real() - env.solution.get_real()))

                # Some of these trajectories can get big, so clean them out when we don't need
                # them.
                #TODO Make sure it's actually freeing them.
                env.close()
                gc.collect()
            envs = []

            plots.convergence_plot(convergence_grid_range, error, args.log_dir)
            # Also log convergence data.
            for nx, error in zip(convergence_grid_range, error):
                outer_logger.logkv("nx", nx)
                outer_logger.logkv("l2_error", error)
                outer_logger.dumpkvs()

            if dims == 1:
                plots.error_plot(x_vals, error_vals, convergence_grid_range, args.log_dir,
                        name="convergence_over_x.png")
            print("Convergence plot created in {}.".format(
                    human_readable_time_delta(time.time() - convergence_start_time)))
    except KeyboardInterrupt:
        print("Test stopped by interrupt.")
        meta_file.log_finish_time(status="stopped by interrupt")
        sys.exit(0)
    except Exception as e:
        meta_file.log_finish_time(status="stopped by exception: {}".format(type(e).__name__))
        raise  # Re-raise so exception is also printed.

    print("Done.")
    meta_file.log_finish_time(status="finished cleanly")


if __name__ == "__main__":
    main()
