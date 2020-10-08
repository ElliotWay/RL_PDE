import os
os.environ['PYTHONHASHSEED'] = "42069"
import argparse
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

from envs import get_env_arg_parser, build_env
from util import metadata
from util.misc import rescale, set_global_seed
from util import action_snapshot
from rl_pde.emi import BatchEMI, StandardEMI, TestEMI


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not train and show the environment parameters not listed here.")
    parser.add_argument('--help-model', default=False, action='store_true',
                        help="Do not train and show the model training parameters not listed here.")
    parser.add_argument('--emi', type=str, default='batch',
                        help="Environment-model interface. Options are 'batch' and 'std'.")
    parser.add_argument('--algo', '-a', type=str, default="sac",
                        help="Algorithm to train with. sac or ddpg, though ddpg hasn't been updated in a while.")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to train the agent.")
    parser.add_argument('--eval-env', '--eval_env', default=None,
                        help="Evaluation env. Default is to use an identical environment to the training environment."
                        + " Pass 'custom' to use a representative sine/rarefaction/accelshock combination.")
    parser.add_argument('--log-dir', '--log_dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is log/env/algo/timestamp.")
    parser.add_argument('--ep-length', '--ep_length', type=int, default=250,
                        help="Number of timesteps in an episode.")
    parser.add_argument('--total-episodes', '--total_episodes', type=int, default=1000,
                        help="Total number of episodes to train.")
    parser.add_argument('--log-freq', '--log_freq', type=int, default=10,
                        help="Number of episodes to wait between logging information.")
    parser.add_argument('--n-best-models', '--b_best_models', type=int, default=5,
                        help="Number of best models so far to keep track of.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--repeat', type=str, default=None,
                        help="Copy parameters from a meta.txt file to run a similar or identical experiment."
                        + " Explicitly passed parameters override parameters in the meta file."
                        + " Passing an explicit --log-dir is recommended.")
    parser.add_argument('--render', type=str, default="file",
                        help="How to render output. Options are file and none.")
    parser.add_argument('--animate', type=int, default=None, 
                        help="Enable animation mode. Plot the state at every nth timestep, and keep the axes fixed across every plot."
                        + " This option also forces the same_eval_env option.")
    parser.add_argument('-y', '--y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', '--n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files. Useful for scripts. Overrides the -y option.")

    main_args, rest = parser.parse_known_args()

    env_arg_parser = get_env_arg_parser()
    env_args, rest = env_arg_parser.parse_known_args(rest)

    model_arg_parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    model_arg_parser.add_argument('--layers', type=int, nargs='+', default=[32, 32],
            help="Size of network layers.")
    model_arg_parser.add_argument('--layer-norm', '--layer_norm', default=False, action'store_true',
            help="Use layer normalization between network layers.")
    model_arg_parser.add_argument('--gamma', type=float, default=0.0,
            help="Discount factor on future rewards.")
    model_arg_parser.add_argument('--learning-rate', '--learning_rate', '--lr', type=float, default=3e-4,
            help="(some) Learning rate if the model uses only one for all networks.")
    model_arg_parser.add_argument('--actor-lr', '--actor_lr', type=float, default=1e-4,
            help="(some) Learning rate for actor network (pi).")
    model_arg_parser.add_argument('--critic-lr', '--critic_lr', type=float, default=1e-3,
            help="(some) Learning rate for critic network (Q).")
    model_arg_parser.add_argument('--buffer-size', '--buffer_size', type=int, default=50000,
            help="Size of the replay buffer.")
    model_arg_parser.add_argument('--batch-size', '--batch_size', type=int, default=64,
            help="Size of batch samples from replay buffer.")
    #model_arg_parser.add_argument('--noise-type', '--noise_type', type=str, default='adaptive-param_0.2',
            #help="(DDPG) Noise added to actions.")
    model_arg_parser.add_argument('--learning-starts', type=int, default=None,
            help="(SAC) At what step the learning starts; before this, actions are take at"
            + " random. Note: a full timestep actually contains dx steps. Default is one episode's"
            + " worth.")
    model_args, rest = model_arg_parser.parse_known_args(rest)
     
    if main_args.env.startswith("weno"):
        mode = "weno"
    elif main_args.env.startswith("split_flux"):
        main_mode = "split_flux"
    elif main_args.env.startswith("flux"):
        mode = "flux"
    else:
        mode = "n/a"
    # TODO internal args? Shouldn't such args be external? IE replace env arg with mode+problem?
    internal_args = {'mode':mode}

    args = Namespace(**vars(main_args), **vars(env_args), **vars(model_args), **internal_args)

    if len(rest) > 0:
        print("Unrecognized arguments: " + " ".join(rest))
        sys.exit(0)

    if args.help_env or args.help_model:
        if args.help_env:
            env_arg_parser.print_help()
        if args.help_algo:
            model_arg_parser.print_help()
        sys.exit(0)

    if args.repeat is not None:
        metadata.load_to_namespace(args.repeat, args)

    set_global_seed(args.seed)

    env = build_env(args.env, args)
    eval_env = build_env(args.env, args, test=True) # Some algos like an extra env for evaluation.
    if args.eval_env is None:
        eval_episodes = 1
    else:
        assert(args.eval_env == "custom")
        #TODO fix this ugly hack and make these parameters.
        eval_env.grid._init_type = "schedule"
        eval_env.grid._init_schedule = ["smooth_sine", "smooth_rare", "accelshock"]
        eval_episodes = len(eval_env.grid._init_schedule)
        eval_env.episode_length = 500

    action_snapshot.declare_standard_envs(args)

    if args.emi == 'batch':
        pass

    #Replacing old run_train.
    # Declare algo; pass model/model class to algo?
    # Scaling/adjusting is related to mode, and is the responsibility of the algo.

    # Things like this make me wish I was writing in a functional language.
    # I sure could go for some partial evaluation and some function composition.
    def flat_rescale_from_tanh(action):
        action = rescale(action, [-1,1], [0,1])
        return action / np.sum(action, axis=-1)[..., None]

    def softmax(action):
        exp_actions = np.exp(action)
        return exp_actions / np.sum(exp_actions, axis=-1)[..., None]

    def back_to_tanh(action):
        return rescale(action, [0,1], [-1,1])

    def identity_function(arg):
        return arg

    def identity_correction(squashed_policy, logp_pi):
        return logp_pi

    clip_obs = 5 # (in stddevs from the mean)
    epsilon = 1e-10
    def z_score_last_dim(obs):
        z_score = (obs - obs.mean(axis=-1)[..., None]) / (obs.std(axis=-1)[..., None] + epsilon)
        return np.clip(z_score, -clip_obs, clip_obs)

    def z_score_all_dims(obs):
        z_score = (obs - obs.mean()) / (obs.std() + epsilon)
        return np.clip(z_score, -clip_obs, clip_obs)

    # Need to handle different state and action spaces differently.
    if args.env == "weno_burgers":
        squash_function = None   # Use default tanh squash/correction.
        squash_correction = None
        action_adjust = softmax
        action_adjust_inverse = back_to_tanh
        obs_adjust = z_score_last_dim
    elif args.env == "split_flux_burgers":
        squash_function = identity_function
        squash_correction = identity_correction
        action_adjust = identity_function
        action_adjust_inverse = identity_function
        obs_adjust = z_score_last_dim
    elif args.env == "flux_burgers":
        squash_function = identity_function
        squash_correction = identity_correction
        action_adjust = identity_function
        action_adjust_inverse = identity_function
        obs_adjust = z_score_last_dim
    else:
        raise Exception("Need to implement parameterized scaling for: \"{}\".".format(args.env))

    policy_kwargs = {'layers':args.layers, 'squash_function':squash_function}
    if args.algo == "sac":
        policy_kwargs['squash_correction'] = squash_correction


    if args.algo == "sac":
        model = SACBatch(SACPolicy, env, eval_env=eval_env, gamma=args.gamma, policy_kwargs=policy_kwargs, learning_rate=args.learning_rate, buffer_size=args.buffer_size,
                 learning_starts=100, batch_size=args.batch_size, verbose=1, tensorboard_log="./log/weno_burgers/tensorboard",
                 action_adjust=action_adjust, action_adjust_inverse=action_adjust_inverse, obs_adjust=obs_adjust)
    elif args.algo == "ddpg":
        print("TODO: add parameterized squash function, action/state scaling to DDPG")
        sys.exit(0)
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

        model = DDPGBatch(DDPGPolicy, env, eval_env=eval_env, gamma=args.gamma,
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
    try:
        log_link_name = "last"
        if os.path.islink(log_link_name):
            os.unlink(log_link_name)
        os.symlink(args.log_dir, log_link_name, target_is_directory=True)
    except OSError:
        print("Failed to create \"last\" symlink. Maybe you're a non-admin on a Windows machine?")

    metadata.create_meta_file(args.log_dir, args)

    # Put stable-baselines logs in same directory.
    logger.configure(folder=args.log_dir, format_strs=['stdout', 'log', 'csv'])  # ,tensorboard'
    logger.set_level(logger.DEBUG)  # logger.INFO

    # Call model.learn().
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        model.learn(total_timesteps=args.total_timesteps, log_interval=args.log_freq, eval_episodes=eval_episodes, render=args.render, render_every=args.animate)
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
