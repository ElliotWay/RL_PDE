import argparse
import sys
import re

#import scipy.stats
import numpy as np

from util.argparse import positive_int, nonnegative_float, positive_float, misc_dict
from envs.abstract_pde_env import AbstractPDEEnv
from envs import WENOBurgersEnv, SplitFluxBurgersEnv, FluxBurgersEnv
from envs.burgers_2d_env import WENOBurgers2DEnv
from envs.euler_env import WENOEulerEnv
from envs.weno_solution import RKMethod


# Could pass name of env, and only have relevant parameters instead of allowing all of them?
def get_env_arg_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num-cells', '--num_cells', '--nx', type=str, nargs='+',
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
    parser.add_argument('--rk', type=str, default='euler',
                        help="RK method for this environment, e.g. euler, rk4, or ssp_rk3.")
    parser.add_argument('--solution-rk', '--solution_rk', type=str, default=None,
                        help="RK method for the solution. By default use the same RK method as"
                        + " the environment.")
    parser.add_argument('--init-params',  '--init_params', type=misc_dict, default=None,
                        help="Some initial conditions accept parameters. For example, smooth_sine"
                        + " accepts A for the amplitude of the wave. Pass these parameters as a"
                        + " comma-separated list, e.g. \"a=1.0,b=2.5\". Note that this overrides"
                        + " any random sampling that might exist for given parameters. Dig into"
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

def set_contingent_env_defaults(main_args, env_args, arg_manager=None, test=False, print_prefix=""):
    """
    Set the defaults of environment parameters that depend on other parameters.

    Parameters
    ----------
    main_args : Namespace
        The main 'args' namespace, containing high level parameters like 'total_episodes'.
    env_args : Namespace
        The sub namespace containing parameters for the environment.
    arg_manager : ArgTreeManager
        Argument manager used to flag some arguments as explicit if they depend tightly on
        arguments that actually were explicit. If None is passed, then those arguments will not be
        marked as explicit. Pass the arg_manager when you want to set these defaults before loading
        from another file.
    test : bool
        Whether this is a test or training run. Some defaults depend on this.
    print_prefix : str
        This function print messages so the user is aware of how the default parameters may have
        changed. However, this can be confusing if multiple environments with different parameters
        are configured this way. Use print_prefix to distinguish these messages. Include
        whitespace.
    """
    pfx = print_prefix
    if not test and main_args.env is not "weno_burgers":
        if env_args.rk != 'euler' or (
                env_args.solution_rk is not None and env_args.solution_rk != 'euler'):
            #env_args.rk = 'euler'
            #env_args.solution_rk = 'euler'
            raise Exception("RK methods during training are only implemented for weno_burgers.")
    # Allow choosing an rk method like --rk 4 in addition to --rk rk4.
    if re.fullmatch('\d+', env_args.rk):
        env_args.rk = f"rk{env_args.rk}"
    if env_args.solution_rk is not None and re.fullmatch('\d+', env_args.solution_rk):
        env_args.solution_rk = f"rk{env_args.solution_rk}"

    if main_args.model == "full" and env_args.reward_mode is None:
        print(f"{pfx}Reward mode defaulted to 'one-step' as expected by 'full' model.")
        env_args.reward_mode = "one-step"

    if env_args.memoize is None:
        if test or 'one-step' in env_args.reward_mode or env_args.analytical:
            env_args.memoize = False
        elif env_args.fixed_timesteps:
            env_args.memoize = True
        else:
            env_args.memoize = False
        #if env_args.init_type in ['random', 'random-many-shocks', 'schedule', 'sample']:
            #env_args.memoize = False
        #else:
            #env_args.memoize = True
    
    env_args.reward_mode = AbstractPDEEnv.fill_default_reward_mode(env_args.reward_mode)
    print(f"{pfx}Full reward mode is '{env_args.reward_mode}'.")

    just_defaults = (env_args.num_cells is None and env_args.timestep is None
            and env_args.time_max is None and env_args.ep_length is None)
    default_time_max = (env_args.time_max is None)

    if env_args.num_cells is not None and type(env_args.num_cells) is not int:
        if env_args.num_cells[0] == "random":
            # Build a new random generator to make sure randomness depends on the seed.
            rng = np.random.RandomState(main_args.seed)
            random_size = int(2**rng.uniform(6,10))
            #random_size = int(scipy.stats.loguniform(64,1024).rvs())
            env_args.num_cells = (random_size,)
            print(f"{pfx}Random grid size is {random_size}.")
        else:
            env_args.num_cells = [int(num) for num in env_args.num_cells]

    # Some environments have specific defaults.
    if env_args.init_type == 'jsz7':
        assert main_args.env == "weno_burgers_2d"
        if env_args.min_value is None:
            env_args.min_value = (0.0,)
        if env_args.max_value is None:
            env_args.max_value = (4.0,)
        if env_args.num_cells is None:
            env_args.num_cells = (160,)
        if env_args.time_max is None:
            env_args.time_max = 0.5
    elif env_args.init_type == 'sod':
        assert main_args.env == "weno_euler"
        if env_args.min_value is None:
            env_args.min_value = (-0.5)
        if env_args.max_value is None:
            env_args.max_value = (0.5)
    else:
        if env_args.min_value is None:
            env_args.min_value = (0.0,)
        if env_args.max_value is None:
            env_args.max_value = (1.0,)


    try:
        if env_args.num_cells is not None:
            _ = iter(env_args.num_cells)
    except TypeError:
        env_args.num_cells = (env_args.num_cells,)
        print(f"{pfx}num cells changed to {env_args.num_cells}.")
    try:
        _ = iter(env_args.min_value)
    except TypeError:
        env_args.min_value = (env_args.min_value,)
    try:
        _ = iter(env_args.max_value)
    except TypeError:
        env_args.max_value = (env_args.max_value,)

    dims = env_dimensions(main_args.env)

    # Most functions expect num_cells to be a tuple of length dims.
    if env_args.num_cells is not None and len(env_args.num_cells) == 1:
        env_args.num_cells = (env_args.num_cells[0],) * dims

    if dims > 1 and len(env_args.min_value) == 1:
        env_args.min_value = (env_args.min_value[0],) * dims
    if dims > 1 and len(env_args.max_value) == 1:
        env_args.max_value = (env_args.max_value[0],) * dims

    if env_args.time_max is None and env_args.ep_length is None:
        env_args.time_max = 0.1
    if (default_time_max and test and env_args.time_max is not None
            and env_args.init_type != "jsz7"):
        # Keeping jsz7 always 0.5s, for whatever reason. Not sure why I'm handling this one
        # differently.
        env_args.time_max = 2 * env_args.time_max

    # Make timestep length depend on grid size or vice versa.
    num_cells, dt, ep_length, time_max = AbstractPDEEnv.fill_default_time_vs_space(
            env_args.num_cells, env_args.min_value, env_args.max_value,
            dt=env_args.timestep, C=env_args.C, ep_length=env_args.ep_length,
            time_max=env_args.time_max)

    env_args.num_cells = num_cells
    env_args.timestep = dt
    env_args.ep_length = ep_length
    env_args.time_max = time_max

    print(f"{pfx}Using {env_args.num_cells} cells and {env_args.timestep}s timesteps."
            + f" Episode length is {env_args.ep_length} steps,"
            + f" for a total of {env_args.time_max}s.")
    if arg_manager is not None:
        # Mark these arguments as explicitly passed.
        if not just_defaults:
            env_arg_manager = arg_manager.get_child('e')
            env_arg.manager.set_explicit('num_cells', 'timestep', 'time_max', 'ep_length')

            # The original way to do this before creating the argument manager was to add the
            # arguments directly to argv. We still need these if we want to load from an old file.
            # Remove them if we no longer need that backwards compatability.
            sys.argv += ['--num-cells', str(num_cells)]
            sys.argv += ['--timestep', str(dt)]
            sys.argv += ['--time_max', str(time_max)]
            sys.argv += ['--ep_length', str(ep_length)]
        if not env_args.fixed_timesteps:
            env_arg_manager = arg_manager.get_child('e')
            env_arg_manager.explicit['C'] = True
            sys.argv += ['--C', str(env_args.C)]

    # Grid constructors expect singletons for 1 dimension.
    if len(env_args.num_cells) == 1:
        env_args.num_cells = env_args.num_cells[0]
    if len(env_args.min_value) == 1:
        env_args.min_value = env_args.min_value[0]
    if len(env_args.max_value) == 1:
        env_args.max_value = env_args.max_value[0]
    if env_args.boundary is not None and type(env_args.boundary) is not str:
        env_args.boundary = env_args.boundary[0]


def env_dimensions(env_name):
    match = re.search(r"(\d+)(-|_)?(d|D)", env_name)
    if match is not None:
        return int(match.group(1))
    # There are probably exceptions to put here,
    # e.g. environments that are inherently 3d but don't
    # have "3d" in their name.
    else:
        return 1


def env_action_type(env_name):
    if env_name.startswith("weno"):
        return "weno"
    elif env_name.startswith("split_flux"):
        return "split_flux"
    elif env_name.startswith("flux"):
        return "flux"
    else:
        return "n/a"
 

def build_env(env_name, env_args, test=False):

    if env_args.fixed_timesteps:
        env_args.C = None

    rk_method = RKMethod[env_args.rk.upper()]
    if env_args.solution_rk is not None:
        solution_rk_method = RKMethod[env_args.solution_rk.upper()]
    else:
        solution_rk_method = rk_method

    # These all apply to AbstractBurgersEnvs, but might not to other envs.
    kwargs = {  'num_cells': env_args.num_cells,
                'min_value': env_args.min_value,
                'max_value': env_args.max_value,
                'init_type': env_args.init_type,
                'schedule': env_args.schedule,
                'rk_method': rk_method,
                'solution_rk_method': solution_rk_method,
                'init_params': env_args.init_params,
                'boundary': env_args.boundary,
                'C': env_args.C,
                'fixed_step': env_args.timestep,
                'weno_order': env_args.order,
                'state_order': env_args.state_order,
                'nu': env_args.nu,
                'episode_length': env_args.ep_length,
                'analytical': env_args.analytical,
                'precise_weno_order': env_args.precise_order,
                'precise_scale': env_args.precise_scale,
                'reward_adjustment': env_args.reward_adjustment,
                'reward_mode': env_args.reward_mode,
                'memoize': env_args.memoize,
                'srca': env_args.srca,
                'follow_solution': env_args.follow_solution,
                'time_max': env_args.time_max,
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


