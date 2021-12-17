import os
import zipfile
from collections import OrderedDict
import numpy as np
import tensorflow as tf
import gym

from models import Model
from models.builder import get_optimizer
from models.net import gaussian_policy_net
from util.serialize import save_to_zip, load_from_zip
from util.lookup import get_activation

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

class PolicyGradientModel(Model):
    def __init__(self, env, args):

        self.return_style = args.m.return_style
        self.gamma = args.m.gamma

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

        a_fn = get_activation(args.m.activation)
        self.policy_mean, self.policy_log_std = gaussian_policy_net(
                self.state_ph, action_space.shape, args.m.layers,
                activation_fn=a_fn, layer_norm=args.m.layer_norm,
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
        self.params = policy_params

        self.optimizer = get_optimizer(args.m)
        self.train_policy = self.optimizer.apply_gradients(grads)

        tf.global_variables_initializer().run(session=self.session)

        self.setup_loading()

    def setup_loading(self):
        self.load_op = {}
        self.load_ph = {}
        for param in self.params:
            placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape)
            self.load_op[param] = param.assign(placeholder)
            self.load_ph[param] = placeholder

    def compute_returns(self, s, a, r, s2, done):
        if (self.return_style is None
                or self.return_style == "full"):

            reward_acc = 0.0
            # This part is un-Pythonic.
            # Pre-allocate space for returns instead of inserting entries into list to save time
            # from mallocs. This means we need to actually use an index to iterate over these
            # lists.
            returns = np.empty_like(r)
            for index in reversed(range(len(r))):
                if done[index]:
                    reward_acc = 0.0
                reward_acc = r[index] + self.gamma * reward_acc
                returns[index] = reward_acc
            
            return returns
                      
        elif self.return_style == "myopic":
            return r
        else:
            raise Exception("Unknown return-style: {}".format(self.return_style))
    
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
        """
        Save model paramaters to be loaded later.

        Based on StableBaselines save structure.
        """
        extra_data = {
                "return_style": self.return_style,
                "gamma": self.gamma,
        }
        params = self.params
        param_values = self.session.run(params)
        param_dict = OrderedDict((param.name, value) for param, value in zip(params, param_values))

        return save_to_zip(path, data=extra_data, params=param_dict)

    def load(self, path):
        data, param_dict = load_from_zip(path)
        self.__dict__.update(data)

        feed_dict = {}
        for param in self.params:
            placeholder = self.load_ph[param]
            param_value = param_dict[param.name]
            feed_dict[placeholder] = param_value

        self.session.run(self.load_op, feed_dict=feed_dict)
        print("Model loaded from {}".format(path))
        print("I'm not 100% sure loading works correctly, in case it looks completely wrong.")
