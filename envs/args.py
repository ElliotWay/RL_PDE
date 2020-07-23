import argparse

# Could pass name of env, and only have relevant parameters instead of allowing all of them?

def get_env_arg_parser():
    parser = argparse.ArgumentParser(
        add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--xmin', type=float, default=0.0)
    parser.add_argument('--xmax', type=float, default=1.0)
    parser.add_argument('--nx', type=int, default=128,
                        help="Number of cells into which to discretize x dimension.")
    parser.add_argument('--order', type=int, default=3,
                        help="Order of WENO approximation (assuming using WENO environment or agent).")

    parser.add_argument('--fixed-timesteps', dest='fixed_timesteps', action='store_true',
                        help="TODO: not implemented!")
    parser.add_argument('--variable-timesteps', dest='fixed_timesteps', action='store_false',
                        help="TODO: not implemented!")
    parser.set_defaults(fixed_timesteps=True)

    parser.add_argument('--timestep', type=float, default=0.0005,
                        help="Set fixed timestep length. TODO: not implemented!")
    parser.add_argument('--C', type=float, default=0.1,
                        help="Constant used in choosing variable timestep. TODO: not implemented!")

    parser.add_argument('--init_type', '--init-type', type=str, default="sine",
                        help="Shape of the initial state.")
    parser.add_argument('--boundary', '--bc', type=str, default="periodic")

    return parser
