import os
import zipfile
import numpy as np
import tensorflow as tf
import gym

# I found these too useful to avoid using them, but too complex to justify copying.
# Copy them over anyway if you need to modify the network internals further
# (or to remove external dependencies).
# One change to make to that code is to allow for float64 size weights, though that's probably
# unnecessary.
from stable_baselines.common.tf_layers import mlp, linear

from models import Model
from util.misc import serialize_ndarray, deserialize_ndarray

def gaussian_policy_net(state_tensor, action_shape, layers, activation_fn,
        scope="policy", reuse=False):
    """
    Construct a Gaussian policy network.

    The input of the network is the state, and the output is a set of means and the
    log of standard deviations representing the multivariate Gaussian distribution from which the
    action can be drawn.

    Notably, the log standard deviation output is NOT connected to the state; it is instead just a
    set of trainable weights like a separate bias vector. This means this style of network will not
    function for an algorithm such as SAC that needs to control the standard deviation for
    exploration.

    Parameters
    ----------
    state_tensor : Tensor (probably a placeholder) containing the state.
    action_shape : tuple representing shape of an action (e.g. env.action_space.shape)
    layers : iterable of ints for each layer size e.g. [32, 32]. Can be empty list for no hidden
                layers.
    activation_fn : TensorFlow activation function, e.g. tf.nn.relu

    Returns the means and log standard deviations.
    """
    flattened_state_size = np.prod(state_tensor.shape[1:])
    flattened_action_size = np.prod(action_shape)
    with tf.variable_scope(scope, reuse):
        flat_state = tf.reshape(state_tensor, (-1, flattened_state_size))
        # TODO Could use layer normalization - that might be a good idea, esp. with ReLU.
        policy_latent = mlp(flat_state, layers=layers, activ_fn=activation_fn, layer_norm=False)

        # Not sure why the sqrt(2) hyperparameter for initialization. StableBaselines uses it.
        flat_mean = linear(policy_latent, "mean", flattened_action_size, init_scale=np.sqrt(2))
        flat_log_std = tf.get_variable(name='logstd', shape=[flattened_action_size],
                                   initializer=tf.zeros_initializer())

        mean = tf.reshape(flat_mean, (-1,) + action_shape)
        log_std = tf.reshape(flat_log_std, (-1,) + action_shape)

    return mean, log_std

def tf_nll_gaussian(values, means, stds):
    """
    Negative log likelihood of a value being drawn from a Gaussian distribution.
    Used to compute pi(a|s) for an arbitrary action some time after that action has been drawn.
    Composed from TF functions.
    Assumes the first dimension is the batch dimension, and all remaining dimensions
    are dimensions of the distribution (i.e. dimensions of the action space).

    Adapted from the Stable Baselines version.
    """
    action_dims = tuple(range(len(values.shape))[1:])
    action_size = tf.cast(tf.reduce_prod(values.shape[1:]), values.dtype)
    return (0.5 * tf.reduce_sum(tf.square((values - means) / stds), axis=action_dims) 
           + 0.5 * np.log(2.0 * np.pi) * action_size 
           + tf.reduce_sum(stds, axis=-1))

def get_optimizer(args):
    if ("optimizer" not in args
            or args.optimizer is None
            or args.optimizer == "sgd"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate)
    elif args.optimizer == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    elif args.optimizer == "momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate=args.learning_rate,
                momemntum=args.momentum)
    elif args.optimizer == "rmsprop":
        optimizer = tf.train.RMSPropOptimizer(learning_rate=args.learning_rate,
                decay=0.99, momentum=args.momentum, epsilon=1e-5)
    else:
        raise Exception("Unknown optimizer: {}".format(args.optimizer))
    return optimizer

class PolicyGradientModel(Model):
    def __init__(self, env, args):

        self.env = env
        self.args = args

        obs_space = env.observation_space
        action_space = env.action_space
        if not isinstance(obs_space, gym.spaces.Box):
            raise Exception("This model can only handle continuous (i.e. Box) state spaces.")
        if not isinstance(action_space, gym.spaces.Box):
            raise Exception("This model can only handle continuous (i.e. Box) action spaces.")
        dtype = obs_space.dtype
        if not action_space.dtype is dtype:
            raise Exception("State space and action space of different dtypes?"
                    + " ({} and {})".format(dtype, action_space.dtype))

        self.session = tf.Session()
        # Need explicit graph to prevent eager execution,
        # which is apparently not possible with optimizer.apply_gradients.
        #self.graph = tf.Graph()
        #with self.graph.as_default():
            #self.session = tf.Session(graph=self.graph)

        self.state_ph = tf.placeholder(dtype=dtype,
                shape=(None,) + obs_space.shape, name="state")
        self.action_ph = tf.placeholder(dtype=dtype,
                shape=(None,) + action_space.shape, name="action")
        self.reward_ph = tf.placeholder(dtype=dtype, shape=(None,), name="reward")
        self.next_state_ph = tf.placeholder(dtype=dtype,
                shape=(None,) + obs_space.shape, name="next_state")
        self.done_ph = tf.placeholder(dtype=tf.bool, shape=(None,), name="done")
        self.returns_ph = tf.placeholder(dtype=dtype, shape=(None,), name="returns")

        # Probably use ReLU? Could use tanh since network doesn't have too many layers.
        self.policy_mean, self.policy_log_std = gaussian_policy_net(
                self.state_ph, action_space.shape, args.layers, tf.nn.relu,
                scope="policy")
        self.policy_std = tf.exp(self.policy_log_std)
        #TODO Squash policy output?
        self.policy = (self.policy_mean + self.policy_std
                * tf.random.normal(tf.shape(self.policy_mean), dtype=self.policy_mean.dtype))
        self.deterministic_policy = self.policy_mean

        # Policy Gradient
        # (log pi(a|s)) * Q(s,a)
        # (Note the double negative of "loss" and "negative" log likelihood means we maximize
        # this quantity).
        self.policy_loss = tf.reduce_mean(
                tf_nll_gaussian(self.action_ph, self.policy_mean, self.policy_std)
                * self.returns_ph)

        policy_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy")
        policy_gradients = tf.gradients(self.policy_loss, policy_params)
        grads = list(zip(policy_gradients, policy_params))

        self.optimizer = get_optimizer(self.args)
        self.train_policy = self.optimizer.apply_gradients(grads)

        tf.global_variables_initializer().run(session=self.session)

    def compute_returns(self, s, a, r, s2, done):
        if ("return_style" not in self.args
                or self.args.return_style is None
                or self.args.return_style == "full"):

            reward_acc = 0.0
            # This part is un-Pythonic.
            # Pre-allocate space for returns instead of inserting entries into list to save time
            # from mallocs. This means we need to actually use an index to iterate over these
            # lists.
            returns = np.empty_like(r)
            for index in reversed(range(len(r))):
                if done[index]:
                    reward_acc = 0.0
                reward_acc = r[index] + self.args.gamma * reward_acc
                returns[index] = reward_acc
            
            return returns
                      
        elif self.args.return_style == "myopic":
            return r
        else:
            raise Exception("Unknown return-style: {}".format(self.args.return_style))
    
    def train(self, s, a, r, s2, done):

        returns = self.compute_returns(s, a, r, s2, done)

        feed_dict = {self.state_ph:s, self.action_ph:a, self.reward_ph:r,
                self.next_state_ph:s2, self.done_ph:done,
                self.returns_ph:returns}

        policy_loss, _ = self.session.run([self.policy_loss, self.train_policy], feed_dict=feed_dict)
        
        return {"policy_loss": policy_loss}

    def predict(self, state, deterministic=True):
        feed_dict = {self.state_ph:state}

        if deterministic:
            return self.session.run(self.deterministic_policy, feed_dict=feed_dict), None
        else:
            return self.session.run(self.policy, feed_dict=feed_dict), None

    def save(self, path):
        print("Model not saved: save function not implemented.")
        return path

    def load(self, path):
        print("Model not loaded: load function not implemented.")
