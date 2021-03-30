import os
os.environ['PYTHONHASHSEED'] = "42069"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
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

#TODO: Remove dependency on this logger. The functionality we need can be easily implemented.
from stable_baselines import logger
#from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise

from rl_pde.run import train
from envs import get_env_arg_parser, build_env
from models import get_model_arg_parser
from util import metadata, action_snapshot
from util.function_dict import numpy_fn
from util.lookup import get_model_class, get_emi_class
from util.misc import rescale, set_global_seed

def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not train and show the environment parameters not listed here.")
    parser.add_argument('--help-model', default=False, action='store_true',
                        help="Do not train and show the model training parameters not listed here.")
    parser.add_argument('--model', type=str, default='sac',
                        help="Type of model to train. Options are 'sac' and nothing else.")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to train the agent.")
    parser.add_argument('--emi', type=str, default='batch',
                        help="Environment-model interface. Options are 'batch' and 'std'.")
    parser.add_argument('--obs-scale', '--obs_scale', type=str, default='z_score_last',
                        help="Adjustment function to observation. Compute Z score along the last"
                        + " dimension (the stencil) with 'z_score_last', the Z score along every"
                        + " dimension with 'z_score_all', or leave them the same with 'none'.")
    parser.add_argument('--eval-env', '--eval_env', default=None,
                        help="Evaluation env. Default is to use an identical environment to the training environment."
                        + " Pass 'custom' to use a representative sine/rarefaction/accelshock combination.")
    parser.add_argument('--log-dir', '--log_dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is"
                        + " log/env/model/timestamp.")
    parser.add_argument('--ep-length', '--ep_length', type=int, default=250,
                        help="Number of timesteps in an episode.")
    parser.add_argument('--total-episodes', '--total_episodes', type=int, default=1000,
                        help="Total number of episodes to train.")
    parser.add_argument('--log-freq', '--log_freq', type=int, default=10,
                        help="Number of episodes to wait between logging information.")
    parser.add_argument('--n-best-models', '--n_best_models', type=int, default=5,
                        help="Number of best models so far to keep track of.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--repeat', type=str, default=None,
                        help="Copy parameters from a meta.txt file to run a similar or identical experiment."
                        + " Explicitly passed parameters override parameters in the meta file."
                        + " Passing an explicit --log-dir is recommended.")
    parser.add_argument('-y', '--y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', '--n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files. Useful for scripts. Overrides the -y option.")

    main_args, rest = parser.parse_known_args()

    env_arg_parser = get_env_arg_parser()
    env_args, rest = env_arg_parser.parse_known_args(rest)

    model_arg_parser = get_model_arg_parser()
    model_args, rest = model_arg_parser.parse_known_args(rest)
     
    if main_args.env.startswith("weno"):
        mode = "weno"
    elif main_args.env.startswith("split_flux"):
        mode = "split_flux"
    elif main_args.env.startswith("flux"):
        mode = "flux"
    else:
        mode = "n/a"
    # TODO internal args? Shouldn't such args be external? IE replace env arg with mode+problem?
    # Reason not to is that it would hide the fact that different modes are really different envs.
    internal_args = {'mode':mode}

    args = Namespace(**vars(main_args), **vars(env_args), **vars(model_args), **internal_args)

    if args.help_env or args.help_model:
        if args.help_env:
            env_arg_parser.print_help()
        if args.help_model:
            model_arg_parser.print_help()
        sys.exit(0)

    if len(rest) > 0:
        raise Exception("Unrecognized arguments: " + " ".join(rest))

    if args.repeat is not None:
        metadata.load_to_namespace(args.repeat, args)

    set_global_seed(args.seed)

    env = build_env(args.env, args)

    eval_env_args = Namespace(**vars(args))
    eval_env_args.follow_solution = False
    if args.eval_env is None:
        eval_envs = [build_env(args.env, eval_env_args, test=True)]
    else:
        assert(args.eval_env == "custom")
        eval_envs = []
        smooth_sine_args = Namespace(**vars(eval_env_args))
        smooth_sine_args.init_type = "smooth_sine"
        smooth_sine_args.ep_length = 500
        eval_envs.append(build_env(args.env, smooth_sine_args, test=True))
        smooth_rare_args = Namespace(**vars(eval_env_args))
        smooth_rare_args.init_type = "smooth_rare"
        smooth_rare_args.ep_length = 500
        eval_envs.append(build_env(args.env, smooth_rare_args, test=True))
        accelshock_args = Namespace(**vars(eval_env_args))
        accelshock_args.init_type = "accelshock"
        accelshock_args.ep_length = 500
        eval_envs.append(build_env(args.env, accelshock_args, test=True))

    action_snapshot.declare_standard_envs(args)

    # Fill in default args that depend on other args.
    if args.buffer_size is None:
        args.buffer_size = 10000 if args.emi == "std" else 500000
    if args.train_freq is None:
        args.train_freq = 1 if args.emi == "std" else env.action_space.shape[0]

    if args.replay_style == 'marl':
        if args.emi != 'marl':
            args.emi = 'marl'
            print("EMI set to MARL for MARL-style replay buffer.")
        if args.batch_size % (args.nx + 1) != 0:
            old_batch_size = args.batch_size
            new_batch_size = old_batch_size + (args.nx + 1) - (old_batch_size % (args.nx + 1))
            args.batch_size = new_batch_size
            print("Batch size changed from {} to {} to align with MARL-style replay buffer."
                    .format(old_batch_size, new_batch_size))
        if args.buffer_size % (args.nx + 1) != 0:
            old_buffer_size = args.buffer_size
            new_buffer_size = old_buffer_size + (args.nx + 1) - (old_buffer_size % (args.nx + 1))
            args.buffer_size = new_buffer_size
            print("Replay buffer size changed from {} to {} to align with MARL-style buffer."
                    .format(old_buffer_size, new_buffer_size))

    if args.model == 'pg' or args.model == 'reinforce':
        if args.gamma == 0.0:
            args.return_style = "myopic"
    if args.model == 'full':
        if args.emi != 'batch-global':
            args.emi = 'batch-global'
            print("EMI set to batch-global to fit \'full\' model.")
    model_cls = get_model_class(args.model)

    if args.obs_scale is None:
        if args.mode == 'weno':
            args.obs_scale = 'z_score_last_dim'
        elif args.mode == "split_flux":
            args.obs_scale = 'z_score_last_dim'
        elif args.mode == "flux":
            args.obs_scale = 'z_score_last_dim'
        else:
            print("No state normalization coded for {}.".format(args.env))
            args.obs_scale = 'none'
    obs_adjust = numpy_fn(args.obs_scale)

    if args.action_scale is None:
        if args.mode == 'weno':
            args.action_scale = 'softmax'
        else:
            print("No action normalization coded for {}".format(args.env))
            args.action_scale = 'none'
    action_adjust = numpy_fn(args.action_scale)

    emi_cls = get_emi_class(args.emi)
    emi = emi_cls(env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)

    #DDPG stuff.
    """
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
                param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev),
                                                     desired_action_stddev=float(stddev))
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
                policy_kwargs=policy_kwargs, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                buffer_size=args.buffer_size,
                batch_size=args.batch_size, action_noise=action_noise, param_noise=param_noise,
                verbose=1)
    """

    # Set up logging.
    start_time = time.localtime()
    if args.log_dir is None:
        timestamp = time.strftime("%y%m%d_%H%M%S")
        default_log_dir = os.path.join("log", args.env, args.model, timestamp)
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
    #TODO Get rid of this dependency, only connect these logs when using an SB model wrapper.
    logger.configure(folder=args.log_dir, format_strs=['stdout', 'log', 'csv'])  # ,tensorboard'
    logger.set_level(logger.DEBUG)  # logger.INFO

    # Call train().
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        train(env, eval_envs, emi, args)
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
    emi.save_model(model_file_name)


if __name__ == "__main__":
    main()
