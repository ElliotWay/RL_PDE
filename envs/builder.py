from envs import WENOBurgersEnv
from envs.burgers import Grid1d

def build_env(env_name, args):
    if args.fixed_timesteps:
        args.C = None

    if env_name == "weno_burgers":
        env = WENOBurgersEnv(xmin=args.xmin, xmax=args.xmax, nx=args.nx,
                             init_type=args.init_type, boundary=args.boundary,
                             C=args.C, fixed_step=args.timestep,
                             weno_order=args.order, eps=args.eps, episode_length=args.ep_length,
                             )
    else:
        print("Unrecognized environment type: \"" + str(env_name) + "\".")
        sys.exit(0)

    return env


