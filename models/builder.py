import argparse

def get_model_arg_parser():
    parser = argparse.ArgumentParser(
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--layers', type=int, nargs='+', default=[32, 32],
            help="Size of network layers.")
    parser.add_argument('--layer-norm', '--layer_norm', default=False, action='store_true',
            help="Use layer normalization between network layers.")
    parser.add_argument('--gamma', type=float, default=0.0,
            help="Discount factor on future rewards.")
    parser.add_argument('--learning-rate', '--learning_rate', '--lr', type=float, default=3e-4,
            help="(some) Learning rate if the model uses only one for all networks.")
    parser.add_argument('--actor-lr', '--actor_lr', type=float, default=1e-4,
            help="(some) Learning rate for actor network (pi).")
    parser.add_argument('--critic-lr', '--critic_lr', type=float, default=1e-3,
            help="(some) Learning rate for critic network (Q).")
    parser.add_argument('--buffer-size', '--buffer_size', type=int, default=None,
            help="Size of the replay buffer. The default is 10000 for std EMI, and 500000"
            + " otherwise.")
    parser.add_argument('--batch-size', '--batch_size', type=int, default=64,
            help="Size of batch samples from replay buffer.")
    parser.add_argument('--train-freq', '--train_freq', type=int, default=None,
            help="(SAC) Ratio between the number of steps and the number of times to train."
            + " The default for std is 1, i.e. train every step. The default otherwise"
            + " is nx+1, i.e. train every time step.")
    #parser.add_argument('--noise-type', '--noise_type', type=str, default='adaptive-param_0.2',
            #help="(DDPG) Noise added to actions.")
    parser.add_argument('--learning-starts', type=int, default=None,
            help="(SAC) At what step the learning starts; before this, actions are take at"
            + " random. Note: a full timestep actually contains dx steps. Default is one episode's"
            + " worth.")
    parser.add_argument('--replay-style', '--replay_style', type=str, default=None,
            help="(SAC) Option to use MARL style replay buffer that samples entire timesteps"
            + " instead of individual locations. Use --replay-style marl."
            + " This also forces --emi marl and increases batch size to be a multiple of the"
            + " spatial dimension.")
 
    return parser
