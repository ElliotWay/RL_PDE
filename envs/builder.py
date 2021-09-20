import argparse
import sys
import re

from util.misc import positive_int, nonnegative_float, positive_float, float_dict
from envs.abstract_pde_env import AbstractPDEEnv
from envs import WENOBurgersEnv, SplitFluxBurgersEnv, FluxBurgersEnv
from envs.burgers_2d_env import WENOBurgers2DEnv
from envs.euler_env import WENOEulerEnv


# Could pass name of env, and only have relevant parameters instead of allowing all of them?
def get_env_arg_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-cells', '--num_cells', '--nx', type=positive_int, nargs='+',
                            default=None,
                        help="Number of cells in the grid. A single value e.g."
                        + " '--num-cells 128' will use the same value for each dimension."
                        + " Multiple values e.g. '--num-cells 128 64' will use a different value"
                        + " for each dimension. The default number of cells"
                        + " depends on the value of --timestep and of --C; if those are both"
                        + " defaults, then it defaults to a square grid of 125.")
    parser.add_argument('--min-value', '--min_value', '--xmin', type=float, nargs='+',
                            default=None,
                        help="Lower bounds of the physical space. May have multiple values,"
                        + " see --num-cells. Default is 0.0, but some initial conditions may"
                        + " specify different defaults.")
    parser.add_argument('--max-value', '--max_value', '--xmax', type=float, nargs='+',
                            default=None,
                        help="Upper bounds of the physical space. May have multiple values,"
                        + " see --num-cells. Default is 1.0, but some initial conditions may"
                        + " specify different defaults.")
    parser.add_argument('--order', type=positive_int, default=2,
                        help="Order of WENO approximation (assuming using WENO environment or agent).")
    parser.add_argument('--state_order', '--state-order', type=positive_int, default=None,
                        help="\"Order\" of state space; use a larger state that corresponds"
                        + " to a different order. The action space is still based on --order.")

    parser.add_argument('--fixed-timesteps', dest='fixed_timesteps', action='store_true',
                        help="Use fixed timesteps. (This is enabled by default.)")
    parser.add_argument('--variable-timesteps', dest='fixed_timesteps', action='store_false',
                        help="Use variable length timesteps.")
    parser.set_defaults(fixed_timesteps=True)

    parser.add_argument('--timestep', '--dt', type=positive_float, default=None,
                        help="Set fixed timestep length. This value is ignored if"
                        + " --variable-timesteps is enabled. The default value is 0.0004.")
    parser.add_argument('--C', type=positive_float, default=0.1,
                        help="Constant used in choosing variable timestep if --variable-timesteps"
                        + " is enabled, or choosing defaults if --timestep and --nx are not both"
                        + " used.")
    parser.add_argument('--time-max', '--time_max', '--tmax', type=positive_float, default=None,
                        help="Set max time of episode (in seconds). Defaults to 0.1s for training"
                        + " and 0.2s for testing. Some initial conditions may override these"
                        + " defaults.")
    parser.add_argument('--ep-length', '--ep_length', type=positive_int, default=None,
                        help="The number of timesteps in the episode. Overrides time-max."
                        + " If --variable-timesteps is used, this is only an approximation"
                        + " based on the value of --timestep.")

    parser.add_argument('--init_type', '--init-type', type=str, default="smooth_sine",
                        help="The type of initial condition.")
    parser.add_argument('--schedule', type=str, nargs='+', default=None,
                        help="List of init-types used with the 'schedule' and 'random'"
                        + " init-type. The default depends on the environment.")
    parser.add_argument('--boundary', '--bc', type=str, default=None, nargs='+',
                        help="The boundary condition of the space. May have multiple values,"
                        + " see --num-cells. The default boundary conditions depend on the"
                        + " init-type, which may vary if using e.g. '--init-type schedule'."
                        + " Set --boundary if you want boundary conditions to be something in"
                        + " particular.")
    parser.add_argument('--init-params',  '--init_params', type=float_dict, default=None,
                        help="Some initial conditions accept parameters. For example, smooth_sine"
                        + " accepts A for the amplitude of the wave. Pass these parameters as a"
                        + " comma-separated list, e.g. \"a=1.0,b=2.5\". Note that this overrides"
                        + " any random sampling that might exist for a given parameters. Dig into"
                        + " envs/grid.py for information on which parameters are available. Omit"
                        + " spaces!",)
    parser.add_argument('--nu', '--eps', type=nonnegative_float, default=0.0,
                        help="Viscosity parameter. Higher values mean more viscous.")
    parser.add_argument('--srca', type=nonnegative_float, default=0.0,
                        help="Random source amplitude. Higher srca means stronger source.")
    parser.add_argument('--analytical', dest='analytical', action='store_true',
                        help="Use an analytical solution. Raises an exception if no analytical"
                        + " solution is available.")
    parser.add_argument('--precise_order', type=positive_int, default=None,
                        help="By default the true solution is computed using the same order as"
                        + " used by the agent. Use this parameter to change it.")
    parser.add_argument('--precise_scale', type=positive_int, default=1,
                        help="By default the true solution is computed with the same precision as"
                        + " the agent sees. Use this parameter to scale up the precision. Only odd"
                        + " numbered scales are accepted to make downsampling easier.")
    parser.add_argument('--reward-adjustment', type=nonnegative_float, default=1000.0,
                        help="Constant that affects the relative importance of small errors compared to big errors."
                        + " Larger values mean that smaller errors are still important compared to big errors.")
    default_reward_mode = AbstractPDEEnv.fill_default_reward_mode("")
    parser.add_argument('--reward-mode', '--reward_mode', type=str, default=None,
                        help="String that controls how the reward is calculated."
                        + " The curent default is '{}'.".format(default_reward_mode)
                        + " This argument may contain multiple parts, so 'stencil-L2dist-nosquash'"
                        + " will use the L2 distance of the stencil with no squash function"
                        + " applied as the reward. It need not contain every part, so 'stencil'"
                        + " uses the stencil to calculate the reward combined with the other"
                        + " default parts.")
    parser.add_argument('--memo', dest='memoize', action='store_true', default=None,
                        help="Use a memoized solution to save time. Enabled by default. See --no-memo.")
    parser.add_argument('--no-memo', dest='memoize', action='store_false', default=None,
                        help="Do not use a memoized solution.")
    parser.add_argument('--follow-solution', default=False, action='store_true',
                        help="Force the environment to follow the solution. This means that"
                        + " actions produce the corresponding reward, but not the "
                        + " corresponding change in state. Only used for training environment.")

    return parser

def set_contingent_env_defaults(args, test=False):
    if args.memoize is None:
        if args.fixed_timesteps:
            args.memoize = True
        else:
            args.memoize = False
        #if args.init_type in ['random', 'random-many-shocks', 'schedule', 'sample']:
            #args.memoize = False
        #else:
            #args.memoize = True

    if args.model == "full" and args.reward_mode is None:
        print("Reward mode forced to use 'one-step' to work with 'full' model.")
        args.reward_mode = "one-step"
    
    args.reward_mode = AbstractPDEEnv.fill_default_reward_mode(args.reward_mode)
    print("Full reward mode is '{}'.".format(args.reward_mode))

    just_defaults = (args.num_cells is None and args.timestep is None
            and args.time_max is None and args.ep_length is None)
    default_time_max = (args.time_max is None)

    # Some environments have specific defaults.
    if args.init_type == "jsz7":
        if args.min_value is None:
            args.min_value = (0.0,)
        if args.max_value is None:
            args.max_value = (4.0,)
        if args.num_cells is None:
            args.num_cells = (160,)
        if args.time_max is None:
            args.time_max = 0.5
    else:
        if args.min_value is None:
            args.min_value = (0.0,)
        if args.max_value is None:
            args.max_value = (1.0,)

    try:
        if args.num_cells is not None:
            _ = iter(args.num_cells)
    except TypeError:
        args.num_cells = (args.num_cells,)
        print("num cells changed to ", args.num_cells)
    try:
        _ = iter(args.min_value)
    except TypeError:
        args.min_value = (args.min_value,)
    try:
        _ = iter(args.max_value)
    except TypeError:
        args.max_value = (args.max_value,)

    dims = env_dimensions(args.env)

    # Most functions expect num_cells to be a tuple of length dims.
    if args.num_cells is not None and len(args.num_cells) == 1:
        args.num_cells = (args.num_cells[0],) * dims

    if dims > 1 and len(args.min_value) == 1:
        args.min_value = (args.min_value[0],) * dims
    if dims > 1 and len(args.max_value) == 1:
        args.max_value = (args.max_value[0],) * dims

    if args.time_max is None and args.ep_length is None:
        args.time_max = 0.1
    if default_time_max and test and args.init_type != "jsz7":
        # Keeping jsz7 always 0.5s, for whatever reason. Not sure why I'm handling this one
        # differently.
        args.time_max = 2 * args.time_max

    # Make timestep length depend on grid size or vice versa.
    num_cells, dt, ep_length, time_max = AbstractPDEEnv.fill_default_time_vs_space(
            args.num_cells, args.min_value, args.max_value,
            dt=args.timestep, C=args.C, ep_length=args.ep_length,
            time_max=args.time_max)

    args.num_cells = num_cells
    args.timestep = dt
    args.ep_length = ep_length
    args.time_max = time_max

    print("Using {} cells and {}s timesteps.".format(args.num_cells, args.timestep)
            + " Episode length is {} steps, for a total of {}s.".format(
                args.ep_length, args.time_max))
    if not just_defaults:
        # Add to argv - if we load an agent later, this prevents the agent's parameters
        # from overwriting these, as at least one of which was explicit.
        sys.argv += ['--num-cells', str(num_cells)]
        sys.argv += ['--timestep', str(dt)]
        sys.argv += ['--time_max', std(time_max)]
        sys.argv += ['--ep_length', str(ep_length)]
    if not args.fixed_timesteps:
        sys.argv += ['--C', str(args.C)]

    # Grid constructors expect singletons for 1 dimension.
    if len(args.num_cells) == 1:
        args.num_cells = args.num_cells[0]
    if len(args.min_value) == 1:
        args.min_value = args.min_value[0]
    if len(args.max_value) == 1:
        args.max_value = args.max_value[0]
    if args.boundary is not None and type(args.boundary) is not str:
        args.boundary = args.boundary[0]


def env_dimensions(env_name):
    match = re.search(r"(\d+)(-|_)?(d|D)", env_name)
    if match is not None:
        return int(match.group(1))
    # There are probably exceptions to put here,
    # e.g. environments that are inherently 3d but don't
    # have "3d" in their name.
    else:
        return 1

def build_env(env_name, args, test=False):

    if args.fixed_timesteps:
        args.C = None

    # These all apply to AbstractBurgersEnvs, but might not to other envs.
    kwargs = {  'num_cells': args.num_cells,
                'min_value': args.min_value,
                'max_value': args.max_value,
                'init_type': args.init_type,
                'schedule': args.schedule,
                'init_params': args.init_params,
                'boundary': args.boundary,
                'C': args.C,
                'fixed_step': args.timestep,
                'weno_order': args.order,
                'state_order': args.state_order,
                'nu': args.nu,
                'episode_length': args.ep_length,
                'analytical': args.analytical,
                'precise_weno_order': args.precise_order,
                'precise_scale': args.precise_scale,
                'reward_adjustment': args.reward_adjustment,
                'reward_mode': args.reward_mode,
                'memoize': args.memoize,
                'srca': args.srca,
                'follow_solution': args.follow_solution,
                'time_max': args.time_max,
                'test': test,
                }

    if env_name == "weno_burgers":
        env = WENOBurgersEnv(**kwargs)
    elif env_name == "weno_burgers_2d":
        env = WENOBurgers2DEnv(**kwargs)
    elif env_name == "split_flux_burgers":
        env = SplitFluxBurgersEnv(**kwargs)
    elif env_name == "flux_burgers":
        env = FluxBurgersEnv(**kwargs)
    elif env_name == "weno_euler":
        env = WENOEulerEnv(**kwargs)
    else:
        raise Exception("Unrecognized environment type: \"" + str(env_name) + "\".")

    return env


