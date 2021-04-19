import argparse

from util.misc import positive_int, nonnegative_float, positive_float, float_dict
from envs import WENOBurgersEnv, SplitFluxBurgersEnv, FluxBurgersEnv


# Could pass name of env, and only have relevant parameters instead of allowing all of them?
def get_env_arg_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--xmin', type=float, default=0.0)
    parser.add_argument('--xmax', type=float, default=1.0)
    parser.add_argument('--nx', type=positive_int, default=128,
                        help="Number of cells into which to discretize x dimension.")
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

    parser.add_argument('--timestep', '--dt', type=positive_float, default=0.0004,
                        help="Set fixed timestep length. This value is ignored if --variable-timesteps is enabled.")
    parser.add_argument('--C', type=positive_float, default=0.1,
                        help="Constant used in choosing variable timestep. This value is not used unless --variable-timesteps is enabled.")

    parser.add_argument('--init_type', '--init-type', type=str, default="sine",
                        help="Shape of the initial state.")
    parser.add_argument('--boundary', '--bc', type=str, default=None,
                        help="The default boundary conditions depend on the init_type. Use --boundary if you want them to be something in particular.")
    parser.add_argument('--init-params',  '--init_params', type=float_dict, default=None,
                        help="Some initial conditions accept parameters. For example, smooth_sine"
                        + " accepts A for the amplitude of the wave. Pass these parameters as a"
                        + " comma-separated list, e.g. \"a=1.0,b=2.5\". Note that this overrides"
                        + " any random sampling that might exist for a given parameters. Dig into"
                        + " envs/grid.py for information on which parameters are available. Omit"
                        + " spaces!",)
    parser.add_argument('--eps', type=nonnegative_float, default=0.0,
                        help="Viscosity parameter. Higher eps means more viscous.")
    parser.add_argument('--srca', type=nonnegative_float, default=0.0,
                        help="Source strength, Higher srca means stronger source.")
    parser.add_argument('--analytical', dest='analytical', action='store_true',
                        help="Use an analytical solution. Raises an exception if no analytical solution is available.")
    parser.add_argument('--precise_order', type=positive_int, default=None,
                        help="By default the true solution is computed using the same order as used by the agent. Use this parameter to change it.")
    parser.add_argument('--precise_scale', type=positive_int, default=1,
                        help="By default the true solution is computed with the same precision as the agent sees."
                        + " Use this parameter to scale up the precision. Only odd numbered scales are accepted"
                        + " to make downsampling easier.")
    parser.add_argument('--reward-adjustment', type=nonnegative_float, default=1000.0,
                        help="Constant that affects the relative importance of small errors compared to big errors."
                        + " Larger values mean that smaller errors are still important compared to big errors.")
    parser.add_argument('--reward-mode', '--reward_mode', type=str, default=None,
                        help="String that controls how the reward is calculated. The default"
                        + " depends on the current implementation in burgers_env.py."
                        + " This argument may contain multiple parts, so 'stencil-L2dist-nosquash'"
                        + " will use the L2 distance of the stencil with no squash function"
                        + " applied as the reward. It need not contain every part, so 'stencil'"
                        + " uses the stencil to calculate the reward combined with the other"
                        + " default parts.")
    parser.add_argument('--memo', dest='memoize', action='store_true', default=None,
                        help="Use a memoized solution to save time. Enabled by default except with random, "
                        + " schedule, and sample initial conditions, and in run_test.py. See --no-memo.")
    parser.add_argument('--no-memo', dest='memoize', action='store_false', default=None,
                        help="Do not use a memoized solution.")
    parser.add_argument('--follow-solution', default=False, action='store_true',
                        help="Force the environment to follow the solution. This means that"
                        + " actions produce the corresponding reward, but not the "
                        + " corresponding change in state. Only used for training environment.")

    return parser

def build_env(env_name, args, test=False):
    if args.fixed_timesteps:
        args.C = None

    if args.memoize is None:
        if args.init_type in ['random', 'schedule', 'sample']:
            args.memoize = False
        else:
            args.memoize = True

    kwargs = {  'xmin': args.xmin,
                'xmax': args.xmax,
                'nx': args.nx,
                'init_type': args.init_type,
                'init_params': args.init_params,
                'boundary': args.boundary,
                'C': args.C,
                'fixed_step': args.timestep,
                'weno_order': args.order,
                'state_order': args.state_order,
                'eps': args.eps,
                'episode_length': args.ep_length,
                'analytical': args.analytical,
                'precise_weno_order': args.precise_order,
                'precise_scale': args.precise_scale,
                'reward_adjustment': args.reward_adjustment,
                'reward_mode': args.reward_mode,
                'memoize': args.memoize,
                'srca': args.srca,
                'test': test,
                }

    if env_name == "weno_burgers":
        env = WENOBurgersEnv(follow_solution=args.follow_solution, **kwargs)
    elif env_name == "split_flux_burgers":
        env = SplitFluxBurgersEnv(**kwargs)
    elif env_name == "flux_burgers":
        env = FluxBurgersEnv(**kwargs)
    else:
        raise Exception("Unrecognized environment type: \"" + str(env_name) + "\".")

    return env


