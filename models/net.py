import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization

# Note that these are only used in models that are not global backprop.
# If you really need them, consider copying the SB files (and adding float64 support).
from stable_baselines.common.tf_layers import mlp, linear

def gaussian_policy_net(state_tensor, action_shape, layers, activation_fn, layer_norm,
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
        policy_latent = mlp(flat_state, layers=layers, activ_fn=activation_fn,
                layer_norm=layer_norm)

        # Not sure why the sqrt(2) hyperparameter for initialization. StableBaselines uses it.
        flat_mean = linear(policy_latent, "mean", flattened_action_size, init_scale=np.sqrt(2))
        flat_log_std = tf.get_variable(name='logstd', shape=[flattened_action_size],
                                   initializer=tf.zeros_initializer())

        mean = tf.reshape(flat_mean, (-1,) + action_shape)
        log_std = tf.reshape(flat_log_std, (-1,) + action_shape)

    return mean, log_std

class PolicyNet(Layer):
    def __init__(self, layers, action_shape, activation_fn=tf.nn.relu, layer_norm=False,
                    name=None, dtype=None):
        self.action_shape = action_shape
        self.activation_fn = activation_fn
        self.data_type = dtype
        self.layer_norm = layer_norm
        flattened_action_shape = np.prod(action_shape)

        self.hidden_layers = []
        for i, size in enumerate(layers):
            # LayerNorm happens between the fully connected layer and the activation function.
            # (I think. I don't have a good grasp of why that is.)
            a_fn = None if self.layer_norm else self.activation_fn
            fc_layer = Dense(size, activation=a_fn, name=("fc" + str(i)), dtype=self.data_type)
            self.hidden_layers.append(fc_layer)

        if self.layer_norm:
            self.norm_layers = []
            for i, size in enumerate(layers):
                norm_layer = LayerNormalization(name="layernorm"+str(i), dtype=self.data_type)
                self.norm_layers.append(norm_layer)

        # SB uses orth_init for this layer. Is that necessary?
        self.output_layer = Dense(flattened_action_shape, name="action", dtype=self.data_type)

        super().__init__(name=name)

    def build(self, input_shape):
        # Sub-layers should build automatically when they are called for the first time.
        super().build(input_shape)

    def call(self, state, training=None):
        # The state shape should be [None, something, something, etc.] so using np.prod on
        # state.shape[1:] works here.
        flattened_state_size = np.prod(state.shape[1:])
        flat_state = tf.reshape(state, (-1, flattened_state_size))

        output = flat_state
        if self.layer_norm:
            for fc_layer, norm_layer in zip(self.hidden_layers, self.norm_layers):
                fc_output = fc_layer(output)
                normalized_output = norm_layer(fc_output)
                output = self.activation_fn(normalized_output)
        else:
            for layer in self.hidden_layers:
                output = layer(output)
        flat_action = self.output_layer(output)

        action = tf.reshape(flat_action, (-1,) + self.action_shape)

        return action

class NoisyPolicyNet(PolicyNet):
    def __init__(self, noise_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.noise_size = noise_size

    def call(self, state, training):
        action = super().call(state, training=training)

        if training:
            action = action + self.noise_size * tf.random.normal(shape=tf.shape(action),
            #action = action + self.noise_size * tf.random.uniform(shape=tf.shape(action),
                                                                #minval=-1.0, maxval=1.0,
                                                                dtype=action.dtype)
        return action

class FunctionWrapper(Layer):
    """
    Wrap another Layer with input and output functions.
    """
    def __init__(self, layer, input_fn, output_fn):
        super().__init__()
        self.layer = layer
        self.input_fn = input_fn
        self.output_fn = output_fn

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, input_tensor):
        if self.input_fn is not None:
            modified_input = self.input_fn(input_tensor)
        else:
            modified_input = input_tensor
        output_tensor = self.layer(modified_input)
        if self.output_fn is not None:
            modified_output = self.output_fn(output_tensor)
        else:
            modified_output = output_tensor
        return modified_output

