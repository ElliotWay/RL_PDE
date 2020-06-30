import sys
import time
import argparse
from argparse import Namespace
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from burgers import Grid1d
from burgers_env import WENOBurgersEnv
from weno_agent import StandardWENOAgent


def do_test(env, agent, args):

    #TODO possibly put rollouts in a separate file

    state = env.reset()

    #Ugly and possibly Burgers specific. How to replace?
    uinit = env.grid.u.copy()

    done = False
    t = 0
    next_update = 0

    rewards = []
    total_reward = 0

    start_time = time.time()

    while not done:

        # The agent's policy function takes a batch of states and returns a batch of actions.
        # However, we take that batch of actions and pass it to the environment like a single action.
        actions = agent.policy(state)
        state, reward, done, info = env.step(actions)

        rewards.append(reward)
        total_reward += reward

        if t >= args.ep_length * (next_update / 10):
            print("t = " + str(t))
            next_update += 1

        #TODO log other information

        t += 1

    end_time = time.time()

    print("Test finished in " + str(end_time - start_time) + " seconds.")
    print("Total reward was " + str(total_reward) + ".")

    #TODO more general means of saving final output
    ufinal = env.grid.u.copy()
    plt.plot(uinit[env.grid.ilo:env.grid.ihi+1],ls="-", color="k", label = "start")
    plt.plot(ufinal[env.grid.ilo:env.grid.ihi+1],ls=":", color="k", label = "env")
    plt.draw()
    #TODO save figures to log directory
    plt.savefig("test.png", bbox_inches='tight', pad_inches=0)
    print("Saved plot to test.png.")


#TODO put this in separate environment file
def build_env(args):
    if args.env == "weno_burgers":
        num_ghosts = args.order
        grid = Grid1d(nx=args.nx, ng=num_ghosts, xmin=args.xmin, xmax=args.xmax, bc=args.boundary)
        env = WENOBurgersEnv(grid=grid, C=args.C, weno_order=args.order, episode_length=args.ep_length, init_type=args.init_type)
    else:
        print("Unrecognized environment type: \"" + str(args.env) + "\".")
        sys.exit(0)

    return env

def main():
    parser = argparse.ArgumentParser(
            description="Deploy an existing RL agent in an environment. Note that this script also takes various arguments not listed here.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--show-hidden', default=False, action='store_true',
            help="Do not test and show the hidden parameters not listed here.")
    parser.add_argument('--agent', '-a', type=str, default="default", 
            help="Agent to test. Either a file (unimplemented) or a string for a standard agent. \"default\" uses standard weno coefficients.")
    parser.add_argument('--env', type=str, default="weno_burgers",
            help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--log-dir', type=str, default=None,
            help="Directory to place log file and other results. TODO: implement, create default log dir")
    parser.add_argument('--ep-length', type=int, default=300,
            help="Number of timesteps in an episode.")
    parser.add_argument('--seed', type=int, default=1,
            help="Set random seed for reproducibility.")

    main_args, rest = parser.parse_known_args()

    sub_parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub_parser.add_argument('--xmin', type=float, default=0.0)
    sub_parser.add_argument('--xmax', type=float, default=1.0)
    sub_parser.add_argument('--nx', type=int, default=128,
            help="Number of cells into which to discretize x dimension.")
    sub_parser.add_argument('--order', type=int, default=3,
            help="Order of WENO approximation (assuming using WENO environment or agent).")

    sub_parser.add_argument('--fixed-timesteps', dest='fixed_timesteps', action='store_true',
            help="TODO: not implemented!")
    sub_parser.add_argument('--variable-timesteps', dest='fixed_timesteps', action='store_false')
    sub_parser.set_defaults(fixed_timesteps=True)

    sub_parser.add_argument('--timestep', type=float, default=0.0005,
            help="Set fixed timestep length. TODO: not implemented!")
    sub_parser.add_argument('--C', type=float, default=0.1,
            help="Constant used in choosing variable timestep.")

    sub_parser.add_argument('--init_type', type=str, default="sine",
            help="Shape of the initial state.")
    sub_parser.add_argument('--boundary', '--bc', type=str, default="periodic")


    sub_args, rest = sub_parser.parse_known_args(rest)

    args = Namespace(**vars(main_args), **vars(sub_args))

    if len(rest) > 0:
        print("Ignoring unrecognized arguments: " + " ".join(rest))
        print()

    if args.show_hidden:
        sub_parser.print_help()
        return

    np.random.seed(args.seed)

    env = build_env(args)

    #TODO build log directory
    #TODO save parameters + git commit id to log directory

    #TODO build agent
    if args.agent == "default":
        agent = StandardWENOAgent(order=args.order)

    do_test(env, agent, args)


if __name__ == "__main__":
    main()
