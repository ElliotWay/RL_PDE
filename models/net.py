import numpy as np
import tensorflow as tf

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

def policy_net(state_ph, action_shape, layers, activation_fn,
        scope="policy", reuse=False):
    """
    Construct a deterministic policy network.

    The input of the network is the state, and the output is the action.
    (Or rather, batches thereof.)
    """

    flattened_state_size = np.prod(state_tensor.shape[1:])
    flattened_action_size = np.prod(action_shape)

    with tf.variable_scope(scope, reuse):
        flat_state = tf.reshape(state_tensor, (-1, flattened_state_size))
        # TODO Could use layer normalization - that might be a good idea, esp. with ReLU.
        policy_latent = mlp(flat_state, layers=layers, activ_fn=activation_fn, layer_norm=False)

        # Not sure why the sqrt(2) hyperparameter for initialization. StableBaselines uses it.
        flat_action = linear(policy_latent, "action", flattened_action_size, init_scale=np.sqrt(2))

        action = tf.reshape(flat_action, (-1,) + action_shape)

    return action
