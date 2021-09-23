import os
os.environ['PYTHONHASHSEED'] = "42069"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" #Block GPU for now.
import argparse
from argparse import Namespace
import shutil
import signal
import sys
import time
import subprocess

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib
matplotlib.use("Agg")
import yaml

#TODO: Remove dependency on this logger. The functionality we need can be easily implemented.
from stable_baselines import logger
#from stable_baselines.ddpg.noise import AdaptiveParamNoiseSpec, OrnsteinUhlenbeckActionNoise, NormalActionNoise

from rl_pde.run import train
from rl_pde.emi import DimensionalAdapterEMI, VectorAdapterEMI
from envs import builder as env_builder
from models import builder as model_builder
from util import action_snapshot
from util.metadata import MetaFile
from util.param_manager import ArgTreeManager
from util.function_dict import numpy_fn
from util.lookup import get_model_class, get_emi_class, get_model_dims
from util.misc import rescale, set_global_seed

ON_POSIX = 'posix' in sys.builtin_module_names

def main():
    arg_manager = ArgTreeManager()
    parser = argparse.ArgumentParser(
        description="Train an RL agent in an environment. Note that this script also takes various arguments not listed here.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--help-env', default=False, action='store_true',
                        help="Do not train and show the environment parameters not listed here.")
    parser.add_argument('--help-model', default=False, action='store_true',
                        help="Do not train and show the model training parameters not listed here.")
    parser.add_argument('--model', type=str, default='full',
                        help="Type of model to train. Options are 'full', 'sac', and 'pg'.")
    parser.add_argument('--env', type=str, default="weno_burgers",
                        help="Name of the environment in which to train the agent.")
    parser.add_argument('--emi', type=str, default='batch',
                        help="Environment-model interface. Options are 'batch' and 'std'.")
    # TODO Should obs-scale and action-scale be in model parameters?
    parser.add_argument('--obs-scale', '--obs_scale', type=str, default='z_score_last',
                        help="Adjustment function to observation. Compute Z score along the last"
                        + " dimension (the stencil) with 'z_score_last', the Z score along every"
                        + " dimension with 'z_score_all', or leave them the same with 'none'.")
    parser.add_argument('--action-scale', '--action_scale', type=str, default=None,
                        help="Adjustment function to action. Default depends on environment."
                        + " 'softmax' computes softmax, 'rescale_from_tanh' scales to [0,1] then"
                        + " divides by the sum of the weights, 'none' does nothing.")
    parser.add_argument('--eval-env', '--eval_env', default='std',
                        help="Evaluation env. Pass 'std' for the representative"
                        + " sine/rarefaction/accelshock combination. Pass 'long' to use the"
                        + " training environment with twice as many timesteps. Pass 'same' to use"
                        + " an identical copy of the training environment.")
    parser.add_argument('--log-dir', '--log_dir', type=str, default=None,
                        help="Directory to place log file and other results. Default is"
                        + " log/env/model/timestamp.")
    parser.add_argument('--total-episodes', '--total_episodes', type=int, default=1000,
                        help="Total number of episodes to train.")
    parser.add_argument('--log-freq', '--log_freq', type=int, default=10,
                        help="Number of episodes to wait between logging information.")
    parser.add_argument('--n-best-models', '--n_best_models', type=int, default=5,
                        help="Number of best models so far to keep track of.")
    parser.add_argument('--seed', type=int, default=1,
                        help="Set random seed for reproducibility.")
    parser.add_argument('--repeat', type=str, default=None,
                        help="Load all of the parameters from a previous experiment to run a"
                        + " similar or identical experiment. Explicitly passed parameters override"
                        + " loaded paramters. Passing an explicit --log-dir is recommend.")
    parser.add_argument('-y', '--y', default=False, action='store_true',
                        help="Choose yes for any questions, namely overwriting existing files. Useful for scripts.")
    parser.add_argument('-n', '--n', default=False, action='store_true',
                        help="Choose no for any questions, namely overwriting existing files."
                        + " Useful for scripts. Overrides the -y option.")

    arg_manager.set_parser(parser)
    env_arg_manager = arg_manager.get_child("e", long_name="Environment Parameters")
    env_arg_manager.set_parser(env_builder.get_env_arg_parser())
    model_arg_manager = arg_manager.get_child("m", long_name="Model Parameters")
    model_arg_manager.set_parser(model_builder.get_model_arg_parser())

    args, rest = arg_manager.parse_known_args()

    if args.help_env:
        env_arg_manager.print_help()
        sys.exit(0)
    if args.help_model:
        model_arg_manager.print_help()
        sys.exit(0)

    if len(rest) > 0:
        raise Exception("Unrecognized arguments: " + " ".join(rest))

    env_builder.set_contingent_env_defaults(args, args.e, test=False)
    model_builder.set_contingent_model_defaults(args, args.m, test=False)

    if args.repeat is not None:
        _, extension = os.path.splitext(args.repeat)

        if extension == '.yaml':
            args_dict = yaml.safe_load(args.repeat)
            arg_manager.load_from_dict(args_dict)
        else:
            # Original meta.txt format.
            metadata.load_to_namespace(args.repeat, arg_manager)

    set_global_seed(args.seed)

    action_type = env_builder.env_action_type(args.env)
    dims = env_builder.env_dimensions(args.env)

    env = env_builder.build_env(args.env, args.e)

    eval_env_arg_manager = env_arg_manager.copy()
    eval_env_args = eval_env_arg_manager.args
    eval_env_args.follow_solution = False # Doesn't make sense for eval envs to do that.
    if args.eval_env == "std" or args.eval_env == "custom":
        # We could restore defaults to do this - have env_arg_manager parse the empty string to get
        # defaults, set some values manually, then use set_contingent_env_defaults.
        # But do we want that behavior? Do we want unusual parameters to affect both the training
        # and evaluation environments?

        #eval_env_args.memoize = True
        # Use standard default evaluation environments.
        if args.env == "weno_burgers":
            eval_envs = []
            eval_env_args.boundary = None
            eval_env_args.ep_length = args.e.ep_length * 2
            eval_env_args.time_max = args.e.time_max * 2
            
            eval_env_args.init_type = "smooth_sine"
            eval_envs.append(env_builder.build_env(args.env, eval_env_args, test=True))

            eval_env_args.init_type = "smooth_rare"
            eval_envs.append(env_builder.build_env(args.env, eval_env_args, test=True))

            eval_env_args.init_type = "accelshock"
            eval_envs.append(env_builder.build_env(args.env, eval_env_args, test=True))

        elif args.env == "weno_burgers_2d":
            eval_envs = []
            eval_env_args.boundary = None
            eval_env_args.ep_length = args.e.ep_length * 2
            eval_env_args.time_max = args.e.time_max * 2

            eval_env_args.init_type = "gaussian"
            eval_envs.append(env_builder.build_env(args.env, eval_env_args, test=True))

            eval_env_args.init_type = "1d-smooth_sine-x"
            eval_envs.append(env_builder.build_env(args.env, eval_env_args, test=True))

            eval_env_args.init_type = "jsz7"
            # Restore jsz7 defaults.
            vars(eval_env_args).update({"min_value":None, "max_value":None, "num_cells":None,
                "time_max":None, "ep_length":None, "timestep":None})
            env_builder.set_contingent_env_defaults(args, eval_env_args, test=True)
            eval_envs.append(env_builder.build_env(args.env, eval_env_args, test=True))

        else:
            print(f"No standard evaluation environments declared for {args.env};"
                    + " evaluation environment will be identical to training environment.")
            eval_envs = [env_builder.build_env(args.env, eval_env_args, test=True)]

    elif args.eval_env == "long":
        eval_env_args.ep_length = args.e.ep_length * 2
        eval_envs = [env_builder.build_env(args.env, eval_env_args, test=True)]
    elif args.eval_env == "same" or args.eval_env is None:
        eval_envs = [env_builder.build_env(args.env, eval_env_args, test=True)]
    else:
        raise Exception("eval env type \"{}\" not recognized.".format(args.eval_env))

    # Fill in defaults that are contingent on other arguments.
    if args.m.replay_style == 'marl':
        if args.emi != 'marl':
            args.emi = 'marl'
            print("EMI set to MARL for MARL-style replay buffer.")
    if args.model == 'full':
        if args.emi != 'batch-global':
            args.emi = 'batch-global'
            print("EMI set to batch-global to fit \'full\' model.")

    model_cls = get_model_class(args.model)

    if args.obs_scale is None:
        if action_type == 'weno':
            args.obs_scale = 'z_score_last_dim'
        elif action_type == "split_flux":
            args.obs_scale = 'z_score_last_dim'
        elif action_type == "flux":
            args.obs_scale = 'z_score_last_dim'
        else:
            print("No state normalization coded for {}.".format(args.env))
            args.obs_scale = 'none'
    obs_adjust = numpy_fn(args.obs_scale)

    if args.action_scale is None:
        if action_type == 'weno':
            args.action_scale = 'softmax'
        else:
            print("No action normalization coded for {}".format(args.env))
            args.action_scale = 'none'
    action_adjust = numpy_fn(args.action_scale)

    emi_cls = get_emi_class(args.emi)
    model_dims = get_model_dims(args.model)
    if model_dims < dims:
        if model_dims == 1:
            emi = DimensionalAdapterEMI(emi_cls, env, model_cls, args,
                    obs_adjust=obs_adjust, action_adjust=action_adjust)
        else:
            raise Exception("Cannot adapt {}-dimensional model to {}-dimensional environment."
                    .format(model_dims, dims))
    elif args.env == 'weno_euler':
        emi = VectorAdapterEMI(emi_cls, env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)
    else:
        emi = emi_cls(env, model_cls, args, obs_adjust=obs_adjust, action_adjust=action_adjust)

    #Old DDPG stuff. Probably won't need this again, but leaving it here just in case.
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
    log_link_name = "last"
    if ON_POSIX:
        try:
            if os.path.islink(log_link_name):
                os.unlink(log_link_name)
            os.symlink(args.log_dir, log_link_name, target_is_directory=True)
        except OSError:
            print("Failed to create \"last\" symlink. Continuing without it.")
    else:
        # On Windows, creating a symlink requires admin priveleges, but creating
        # a "junction" does not, even though a junction is just a symlink on directories.
        # I think there may be some support in Python3.8 for this,
        # but we need Python3.7 for Tensorflow 1.15.
        try:
            if os.path.isdir(log_link_name):
                os.rmdir(log_link_name)
            subprocess.run("mklink /J {} {}".format(log_link_name, args.log_dir), shell=True)
        except OSError:
            print("Failed to create \"last\" symlink. Continuing without it.")

    meta_file = MetaFile(args.log_dir, arg_manager)
    meta_file.write_new()

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
        meta_file.log_finish_time(status="stopped by interrupt")
        sys.exit(0)
    except Exception as e:
        meta_file.log_finish_time(status="stopped by exception: {}".format(type(e).__name__))
        raise  # Re-raise so exception is also printed.

    print("Finished!")
    meta_file.log_finish_time(status="finished cleanly")

    model_file_name = os.path.join(args.log_dir, "model_final")
    emi.save_model(model_file_name)


if __name__ == "__main__":
    main()
