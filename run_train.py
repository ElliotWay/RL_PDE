import argparse
import os
import shutil
import signal
import sys
import time
from argparse import Namespace

import numpy as np
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import matplotlib

matplotlib.use("Agg")

from stable_baselines import logger
from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise

from envs import build_env
from envs import get_env_arg_parser
from models.sac import SACBatch
from models.ddpg import DDPGBatch
from models.sac import LnScaledMlpPolicy as SACPolicy
from models.ddpg import LnMlpPolicy as DDPGPolicy
from util import metadata


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not train and show the environment parameters not listed here.")
    parser.add_argument('--help-algo', default=False, action='store_true',
                        help="Do not train and show the algorithm parameters not listed here.")
    parser.add_argument('--algo', '-a', type=str, default="sac",
                        help="Algorithm to train with. sac or ddpg, though ddpg hasn't been updated in a while.")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to deploy the agent.")
    parser.add_argument('--log-dir', '--log_dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is log/env/algo/timestamp.")
    parser.add_argument('--ep-length', '--ep_length', type=int, default=250,
                        help="Number of timesteps in an episode.")
    parser.add_argument('--total-timesteps', '--total_timesteps', type=int, default=int(1e5),
                        help="Total number of timesteps to train.")
    parser.add_argument('--log-freq', '--log_freq', type=int, default=10,
                        help="Number of episodes to wait between logging information.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--render', type=str, default="file",
                        help="How to render output. Options are file, human, and none.")
    parser.add_argument('--animate', type=int, default=None, 
                        help="Enable animation mode. Plot the state at every nth timestep, and keep the axes fixed across every plot.")
    parser.add_argument('-y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files. Useful for scripts. Overrides the -y option.")

    main_args, rest = parser.parse_known_args()

    env_arg_parser = get_env_arg_parser()
    env_args, rest = env_arg_parser.parse_known_args(rest)

    algo_arg_parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    algo_arg_parser.add_argument('--layers', type=int, nargs='+', default=[32, 32],
            help="Size of network layers.")
    algo_arg_parser.add_argument('--learning-rate', '--learning_rate', '--lr', type=float, default=3e-4,
            help="Learning rate for SAC, which uses same rate for all networks.")
    algo_arg_parser.add_argument('--actor-lr', '--actor_lr', type=float, default=1e-4,
            help="Learning rate for actor network (pi).")
    algo_arg_parser.add_argument('--critic-lr', '--critic_lr', type=float, default=1e-3,
            help="Learning rate for critic network (Q).")
    algo_arg_parser.add_argument('--buffer-size', '--buffer_size', type=int, default=50000,
            help="Size of the replay buffer.")
    algo_arg_parser.add_argument('--batch-size', '--batch_size', type=int, default=64,
            help="Size of batch samples from replay buffer.")
    algo_arg_parser.add_argument('--noise-type', '--noise_type', type=str, default='adaptive-param_0.2',
            help="Noise used in DDPG.")
    algo_args, rest = algo_arg_parser.parse_known_args(rest)

    args = Namespace(**vars(main_args), **vars(env_args), **vars(algo_args))

    if len(rest) > 0:
        print("Unrecognized arguments: " + " ".join(rest))
        sys.exit(0)

    if args.help_env or args.help_algo:
        if args.help_env:
            env_arg_parser.print_help()
        if args.help_algo:
            algo_arg_parser.print_help()
        sys.exit(0)

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    env = build_env(args.env, args)
    eval_env = build_env(args.env, args) # Some algos like an extra env for evaluation.

    if args.algo == "sac":
        policy_kwargs = dict(layers=[32, 32])
        model = SACBatch(SACPolicy, env, policy_kwargs=policy_kwargs, learning_rate=args.learning_rate, buffer_size=args.buffer_size,
                 learning_starts=100, batch_size=args.batch_size, verbose=1, tensorboard_log="./log/weno_burgers/tensorboard")
    elif args.algo == "ddpg":
        # Parse noise_type
        action_noise = None
        param_noise = None
        nb_actions = env.action_space.shape[-1]
        for current_noise_type in args.noise_type.split(','):
            current_noise_type = current_noise_type.strip()
            if current_noise_type == 'none':
                pass
            elif 'adaptive-param' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
            elif 'normal' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = NormalActionNoise(mean=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
            elif 'ou' in current_noise_type:
                _, stddev = current_noise_type.split('_')
                action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(nb_actions),
                                                            sigma=float(stddev) * np.ones(nb_actions))
            else:
                raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

        policy_kwargs = dict(layers=[32, 32])
        model = DDPGBatch(DDPGPolicy, env, eval_env=eval_env,
                policy_kwargs=policy_kwargs, actor_lr=args.actor_lr, critic_lr=args.critic_lr, buffer_size=args.buffer_size,
                batch_size=args.batch_size, action_noise=action_noise, param_noise=param_noise,
                verbose=1)
    else:
        print("Algorithm type \"" + str(args.algo) + "\" not implemented.")

    if args.render == "none":
        args.render = None

    # Set up logging.
    start_time = time.localtime()
    if args.log_dir is None:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        default_log_dir = os.path.join("log", args.env, args.algo, timestamp)
        args.log_dir = default_log_dir
        print("Using default log directory: {}".format(default_log_dir))
    try:
        os.makedirs(args.log_dir)
    except FileExistsError:
        if args.n:
            raise Exception("Logging directory \"{}\" already exists!.".format(args.log_dir))
        elif not args.y:
            _ignore = input(("(\"{}\" already exists! Hit <Enter> to overwrite and"
                            + " continue, Ctrl-C to stop.)").format(args.log_dir))
        shutil.rmtree(args.log_dir)
        os.makedirs(args.log_dir)

    # Create symlink for convenience.
    log_link_name = "last"
    if os.path.islink(log_link_name):
        os.unlink(log_link_name)
    os.symlink(args.log_dir, log_link_name, target_is_directory=True)

    metadata.create_meta_file(args.log_dir, args)

    # Put stable-baselines logs in same directory.
    logger.configure(folder=args.log_dir, format_strs=['stdout', 'log', 'csv'])  # ,tensorboard'
    logger.set_level(logger.DEBUG)  # logger.INFO

    # Call model.learn().
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        model.learn(total_timesteps=args.total_timesteps, log_interval=args.log_freq, render=args.render, render_every=args.animate)
    except KeyboardInterrupt:
        print("Training stopped by interrupt.")
        metadata.log_finish_time(args.log_dir, status="stopped by interrupt")
        sys.exit(0)
    except Exception as e:
        metadata.log_finish_time(args.log_dir, status="stopped by exception: {}".format(type(e).__name__))
        raise  # Re-raise so excption is also printed.

    print("Finished!")
    metadata.log_finish_time(args.log_dir, status="finished cleanly")

    model_file_name = os.path.join(args.log_dir, "model_final")
    model.save(model_file_name)


if __name__ == "__main__":
    main()
