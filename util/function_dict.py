import numpy as np
import tensorflow as tf

"""
A dictionary of unary functions.
The intent is that a function can be expressed as a string key, and that function can be fetched
from here. A string key works better as a command line argument or as an object to serialize.
Each key has two entries: a Numpy function that is fetched with numpy_fn(key); and a corresponding
Tensorflow tf.function that is fetched with tensorflow_fn(key).
Note that the Numpy function is intended to be applied on a single object (possibly with many
dimensions), where Tensorflow functions are intended to be applied on a batch of objects. The
Tensorflow functions never operate over the first dimension (e.g. mean is never over the batch
dimension).
"""

_numpy_fn_dict = {}
_tensorflow_fn_dict = {}

def numpy_fn(key):
    return _numpy_fn_dict[key]

def tensorflow_fn(key):
    return _tensorflow_fn_dict[key]

def get_available_functions():
    return _numpy_fn_dict.keys()


def rescale(values, source, target):
    source_low, source_high = source
    target_low, target_high = target

    descaled = (values - source_low) / (source_high - source_low)

    rescaled = (descaled * (target_high - target_low)) + target_low

    return rescaled

def create_rescale_tf(source, target):
    """
    This creates the tf.function instead of acting as one itself. This allows you to pass the
    source and target ranges as scalars instead of converting them to tf.constants first.
    You'll still need to apply the returned function to your tensor.
    """
    source_low, source_high = source
    target_low, target_high = target
    
    source_low = tf.constant(source_low)
    source_high = tf.constant(source_high)
    target_low = tf.constant(target_low)
    target_high = tf.constant(target_high)

    @tf.function
    def rescale_func(value_tensor):
        descaled = (value_tensor - source_low) / (source_high - source_low)
        rescaled = (descaled * (target_high - target_low)) + target_low

    return rescale_func


# Things like this make me wish I was writing in a functional language.
# I sure could go for some partial evaluation and some function composition.

def flat_rescale_from_tanh(action):
    action = rescale(action, [-1,1], [0,1])
    return action / np.sum(action, axis=-1)[..., None]
flat_rescale_from_tanh_tf = create_rescale_tf([-1,1], [0,1])

fn_key = 'rescale_from_tanh'
_numpy_fn_dict[fn_key] = flat_rescale_from_tanh
_tensorflow_fn_dict[fn_key] = flat_rescale_from_tanh_tf

def softmax(action):
    exp_actions = np.exp(action)
    return exp_actions / np.sum(exp_actions, axis=-1)[..., None]
# softmax is already built into Tensorflow.
# Note that the axis argument defaults to -1 on the tf.nn.softmax version.

fn_key = 'softmax'
_numpy_fn_dict[fn_key] = softmax
_tensorflow_fn_dict[fn_key] = tf.nn.softmax

def rescale_to_tanh(action):
    return rescale(action, [0,1], [-1,1])
rescale_to_tanh_tf = create_rescale_tf([0,1], [-1,1])

fn_key = 'rescale_to_tanh'
_numpy_fn_dict[fn_key] = rescale_to_tanh
_tensorflow_fn_dict[fn_key] = rescale_to_tanh_tf

def identity_function(arg):
    return arg
# identity already exists in Tensorflow.

fn_key = 'identity'
_numpy_fn_dict[fn_key] = identity_function
_tensorflow_fn_dict[fn_key] = tf.identity

# What was this originally for?
# This was part of modifying the output of the network in stablebaselines/sac/policies.py, which
# includeds a correction to the Gaussian likelihood of the policy after applying a squashing
# function.
# That correction is built in by default, this function was used to remove that correction if we
# weren't using a squash function.
# It's not needed in the current implementation, I think? But I'm leaving it here to remind me,
# since it's sort of related to the idea of functions as parameters.
#def identity_correction(squashed_policy, logp_pi):
    #return logp_pi

clip_obs = 5.0 # (in stddevs from the mean)
clip_obs_tf = tf.constant(clip_obs)
epsilon = 1e-10
epsilon_tf = tf.constant(epsilon)
def z_score_last_dim(obs):
    z_score = (obs - obs.mean(axis=-1)[..., None]) / (obs.std(axis=-1)[..., None] + epsilon)
    return np.clip(z_score, -clip_obs, clip_obs)
@tf.function
def z_score_last_dim_tf(obs_tensor):
    z_score = ((obs_tensor - tf.reduce_mean(obs_tensor, axis=-1)[..., None])
            / (tf.math.reduce_std(obs_tensor, axis=-1)[..., None] + epsilon_tf))
    return tf.clip_by_value(z_score, -clip_obs_tf, clip_obs)

fn_key = 'z_score_last'
_numpy_fn_dict[fn_key] = z_score_last_dim
_tensorflow_fn_dict[fn_key] = z_score_last_dim_tf

def z_score_all_dims(obs):
    z_score = (obs - obs.mean()) / (obs.std() + epsilon)
    return np.clip(z_score, -clip_obs, clip_obs)
@tf.function
def z_score_all_dims_tf(obs_tensor):
    # Need to leave the batch dimension untouched, so we map tf.reduce_mean across the
    # first dimension.
    shifted = (obs_tensor - tf.map_fn(tf.reduce_mean, obs_tensor))
    #TODO - figure out a way to broadcast this properly. I've spent too long on this considering
    # I'm not planning on using it soon.
    raise NotImplementedError

#fn_key = 'z_score_all'
#_numpy_fn_dict[fn_key] = z_score_all_dims
#_tensorflow_fn_dict[fn_key] = z_score_all_dims_tf

fn_key = 'none'
_numpy_fn_dict[fn_key] = None
_tensorflow_fn_dict[fn_key] = None
