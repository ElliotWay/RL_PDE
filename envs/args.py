import argparse
from util.misc import positive_int, nonnegative_float, positive_float

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

    parser.add_argument('--fixed-timesteps', dest='fixed_timesteps', action='store_true',
                        help="Use fixed timesteps. (This is enabled by default.)")
    parser.add_argument('--variable-timesteps', dest='fixed_timesteps', action='store_false',
                        help="Use variable length timesteps.")
    parser.set_defaults(fixed_timesteps=True)

    parser.add_argument('--timestep', type=positive_float, default=0.0005,
                        help="Set fixed timestep length. This value is ignored if --variable-timesteps is enabled.")
    parser.add_argument('--C', type=positive_float, default=0.1,
                        help="Constant used in choosing variable timestep. This value is not used unless --variable-timesteps is enabled.")

    parser.add_argument('--init_type', '--init-type', type=str, default="sine",
                        help="Shape of the initial state.")
    parser.add_argument('--boundary', '--bc', type=str, default="periodic")
    parser.add_argument('--eps', type=nonnegative_float, default=0.0,
                        help="Viscosity parameter. Higher eps means more viscous.")

    return parser
