import argparse
from argparse import Namespace
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from stable_baselines import logger

from envs import get_env_arg_parser, build_env
from rl_pde.agents import StationaryAgent, EqualAgent, MiddleAgent, LeftAgent, RightAgent, RandomAgent

standard_envs = None

def declare_standard_envs(args):
    # Make a copy - we need to change the args, but the caller should not
    # have to worry about args changing.
    args = Namespace(**vars(args))
    args.env = "weno_burgers" # Should we allow for other sorts of 1D environments here?
    args.memoize = False # No point memoizing any of these environments.
    args.analytical = False
    args.min_value = 0.0
    args.max_value = 1.0

    global standard_envs

    standard_envs = []

    flat_state = np.full(10, 1.0)
    def flat(params):
        return flat_state, {}
    args.init_type = flat
    args.boundary = "outflow"
    args.num_cells = len(flat_state)
    flat_env = build_env(args.env, args)
    standard_envs.append(flat_env)

    rising_state = np.arange(1, 1.2, 0.02)
    def rising(params):
        return rising_state, {}
    args.init_type = rising
    args.boundary = "first"
    args.num_cells = len(rising_state)
    rising_env = build_env(args.env, args)
    standard_envs.append(rising_env)

    falling_state = np.arange(1.2, 1, -0.02)
    def falling(params):
        return falling_state, {}
    args.init_type = falling
    args.boundary = "first"
    args.num_cells = len(falling_state)
    falling_env = build_env(args.env, args)
    standard_envs.append(falling_env)

    shock_state = np.full(10, 1.0)
    shock_state[5:] = -1.0
    def shock(params):
        return shock_state, {}
    args.init_type = shock
    args.boundary = "outflow"
    args.num_cells = len(shock_state)
    shock_env = build_env(args.env, args)
    standard_envs.append(shock_env)

    rare_state = np.full(10, -1.0)
    rare_state[5:] = 1.0
    def rare(params):
        return rare_state, {}
    args.init_type = rare
    args.boundary = "outflow"
    args.num_cells = len(rare_state)
    rare_env = build_env(args.env, args)
    standard_envs.append(rare_env)

def save_action_snapshot(agent, weno_agent=None, suffix=""):
    global standard_envs

    if standard_envs is None:
        raise Exception("Need to pass args to action_snapshot.declare_standard_envs"
                + " before saving an action snapshot.")

    action_dimensions = np.prod(list(standard_envs[0].action_space.shape)[1:])
    fig, axes = plt.subplots(1 + action_dimensions, len(standard_envs), sharex='col', sharey='row', gridspec_kw={'wspace':0, 'hspace':0})

    agent_color = "tab:orange"
    weno_color = "tab:blue"

    for index, env in enumerate(standard_envs):
        state = env.reset()
        grid_state = env.grid.get_full()
        action, _ = agent.predict(state, deterministic=True)
        action = action.reshape((-1, action_dimensions))

        if weno_agent is not None:
            weno_action, _ = weno_agent.predict(state)
            weno_action = weno_action.reshape((-1, action_dimensions))
            assert weno_action.shape == action.shape, \
                    ("weno shape {} does not match action shape {}"
                            .format(weno_action.shape, action.shape))

        state_axis = axes[0, index]
        cell_x_values = env.grid.x
        state_axis.plot(cell_x_values, grid_state[0], linestyle='-', color='black')
        state_axis.set_xmargin(0.0)
        state_axis.set_ylim((-2.0, 2.0))

        interface_x_values = env.grid.inter_x[env.ng:-env.ng]
        for dim in range(action_dimensions):
            action_axis = axes[1+dim, index]
            if weno_agent is not None:
                action_axis.plot(interface_x_values, weno_action[:, dim], linestyle='-', color=weno_color)
            action_axis.plot(interface_x_values, action[:, dim], linestyle='-', color=agent_color)

            action_axis.set_xlim(state_axis.get_xlim())
            action_axis.set_ylim((0.0, 1.0))
            if dim < action_dimensions - 1:
                action_axis.yaxis.set_ticklabels([])

    log_dir = logger.get_dir()
    filename = "action_snap" + suffix + ".png"
    filename = os.path.join(log_dir, filename)
    fig.tight_layout()
    plt.savefig(filename)
    print('Saved plot to ' + filename + '.')
    
    plt.close(fig)
    return filename

def main():
    parser = argparse.ArgumentParser(
        description="Inspect how an agent behaves with certain standard solution shapes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Show the environment parameters not listed here.")
    parser.add_argument('--agent', '-a', type=str, default="default",
                        help="Agent to test. Either a file or a string for a standard agent. \"default\" uses standard weno coefficients.")
    parser.add_argument('--algo', type=str, default="sac",
                        help="Algorithm used to create the agent. Unfortunately necessary to open a model file.")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--log-dir', type=str, default=None,
                        help="Directory to place output. Default is test/snap/agent/timestamp.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('-y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files. Useful for scripts. Overrides the -y option.")
    raise Exception("The main method in this file hasn't been used in a while;"
            + " it almost certainly won't work. Update it if you're trying to get"
            + " an action snapshot of an existing agent.")

    main_args, rest = parser.parse_known_args()

    env_arg_parser = get_env_arg_parser()
    env_args, rest = env_arg_parser.parse_known_args(rest)

    # Some arguments don't make sense here, but the env builder still expects them.
    dummy_args = {"ep_length": 1}

    args = Namespace(**vars(main_args), **vars(env_args), **dummy_args)

    if len(rest) > 0:
        print("Unrecognized arguments: " + " ".join(rest))
        sys.exit(0)

    if args.help_env:
        env_arg_parser.print_help()
        sys.exit(0)

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if args.log_dir is None:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        default_log_dir = os.path.join("test", "snap", args.agent, timestamp)
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

    
    declare_standard_envs(args)
    global standard_envs
    print("There are currently {} standard environment shapes.".format(len(standard_envs)))

    weno_agent = StandardWENOAgent(order=args.order, mode=mode)

    save_action_snapshot(agent, weno_agent=weno_agent)

if __name__ == "__main__":
    main()
