from envs import WENOBurgersEnv
from envs.burgers import Grid1d

def build_env(env_name, args):
    if env_name == "weno_burgers":
        num_ghosts = args.order + 1
        grid = Grid1d(nx=args.nx, ng=num_ghosts, xmin=args.xmin, xmax=args.xmax, bc=args.boundary)
        env = WENOBurgersEnv(grid=grid, C=args.C, weno_order=args.order, episode_length=args.ep_length,
                             init_type=args.init_type)
    else:
        print("Unrecognized environment type: \"" + str(env_name) + "\".")
        sys.exit(0)

    return env


