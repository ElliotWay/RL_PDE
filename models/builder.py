import argparse

import tensorflow as tf

from util.argparse import positive_int, nonnegative_float, positive_float, proportion_float

def get_model_arg_parser(model_type=None):
    parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--layers', type=positive_int, nargs='+', default=[32, 32],
            help="Size of network layers.")
    parser.add_argument('--layer-norm', '--layer_norm', default=False, action='store_true',
            help="Use layer normalization between network layers.")

    # One could make a sound argument that gamma is an environment parameter.
    # In practice, it primarily affects the model.
    if model_type in [None, 'sac', 'pg', 'ddpg']:
        parser.add_argument('--gamma', type=proportion_float, default=0.0,
                help="Discount factor on future rewards.")

    if model_type in [None, 'full', 'sac', 'pg', 'fixed-1step']:
        parser.add_argument('--learning-rate', '--learning_rate', '--lr', type=positive_float,
                default=3e-4,
                help="Learning rate for all components of the model.")
    elif model_type == 'ddpg':
        parser.add_argument('--actor-lr', '--actor_lr', type=positive_float, default=1e-4,
                help="Learning rate for actor network (pi).")
        parser.add_argument('--critic-lr', '--critic_lr', type=positive_float, default=1e-3,
                help="Learning rate for critic network (Q).")
        parser.add_argument('--noise-type', '--noise_type', type=str,
                default='adaptive-param_0.2',
                help="Noise added to actions.")

    if model_type in [None, 'full', 'sac']:
        parser.add_argument('--batch-size', '--batch_size', type=positive_int, default=None,
                help="Number of samples the network trains on at once. For global backprop,"
                + " the default depends on the environment. 64 for other models.")

    if model_type in [None, 'sac']: 
        parser.add_argument('--buffer-size', '--buffer_size', type=positive_int, default=None,
                help="Size of the replay buffer. The default is 10000 for std EMI, and 500000"
                + " otherwise.")
        parser.add_argument('--train-freq', '--train_freq', type=int, default=None,
                help="Ratio between the number of steps and the number of times to train."
                + " The default for std EMI is 1, i.e. train every step. The default otherwise"
                + " is nx+1, i.e. train every time step.")
        parser.add_argument('--learning-starts', type=int, default=None,
                help="At what step the learning starts; before this, actions are take at"
                + " random. Note: a full timestep actually contains dx steps. Default is one"
                + " episode's worth.")
        parser.add_argument('--replay-style', '--replay_style', type=str, default=None,
                help="Option to use MARL style replay buffer that samples entire timesteps"
                + " instead of individual locations. Use --replay-style marl."
                + " This also forces --emi marl and increases batch size to be a multiple of the"
                + " spatial dimension.")

    if model_type in [None, 'full', 'pg']:
        parser.add_argument('--optimizer', type=str, default="adam",
                help="Gradient Descent algorithm to use for training. SAC ignores this parameter"
                + " and always uses Adam.")

    if model_type in [None, 'pg']:
        parser.add_argument('--return-style', '--return_style', type=str, default=None,
                help="(PG) Style of returns for estimating Q(s,a). Default depends on Model."
                + " 'full' uses (or tries to use) the entire return. 'myopic' uses only the"
                + " immediate reward, as if gamma was set to 0.0.")
 
    return parser


def set_contingent_model_defaults(main_args, model_args, test=False):
    if model_args.batch_size is None:
        if main_args.model == "full":
            if main_args.env in ["weno_euler", "weno_burgers_2d"]:
                model_args.batch_size = 1
            else:
                model_args.batch_size = 10
        else:
            model_args.batch_size = 64

    if main_args.model == "sac":
        if model_args.buffer_size is None:
            model_args.buffer_size = 10000 if main_args.emi == "std" else 500000
        if model_args.train_freq is None:
            model_args.train_freq = 1 if main_args.emi == "std" else main_args.e.num_cells

        if model_args.replay_style == 'marl':
            if model_args.batch_size % (main_args.e.num_cells + 1) != 0:
                old_batch_size = model_args.batch_size
                new_batch_size = old_batch_size + (main_args.e.num_cells + 1) - (old_batch_size %
                        (main_args.e.num_cells + 1))
                model_args.batch_size = new_batch_size
                print("Batch size changed from {} to {} to align with MARL-style replay buffer."
                        .format(old_batch_size, new_batch_size))
            if model_args.buffer_size % (main_args.e.num_cells + 1) != 0:
                old_buffer_size = model_args.buffer_size
                new_buffer_size = (old_buffer_size + (main_args.e.num_cells + 1)
                        - (old_buffer_size % (main_args.e.num_cells + 1)))
                model_args.buffer_size = new_buffer_size
                print("Replay buffer size changed from {} to {} to align with MARL-style buffer."
                       .format(old_buffer_size, new_buffer_size))
         
    if main_args.model == 'pg' or main_args.model == 'reinforce':
        if model_args.gamma == 0.0:
            model_args.return_style = "myopic"


def get_optimizer(model_args):
    if (model_args.optimizer is None
            or model_args.optimizer == "sgd"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=model_args.learning_rate)
    elif model_args.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=model_args.learning_rate)
    elif model_args.optimizer == "momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate=model_args.learning_rate,
                momemntum=model_args.momentum)
    elif model_args.optimizer == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=model_args.learning_rate,
                decay=0.99, momentum=model_args.momentum, epsilon=1e-5)
    else:
        raise Exception("Unknown optimizer: {}".format(model_args.optimizer))
    return optimizer
