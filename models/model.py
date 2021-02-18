import os
import numpy as np

class Model:
    """
    RL model representing the ability to train on experience.
    """
    def __init__(self, env, args):
        """ Create a model for the given env using the args namespace. """
        raise NotImplementedError
    def train(self, s, a, r, s2, done):
        """
        Train using samples. The samples should be consecutive.
        Returns a dict of information (may be empty dict).
        """
        raise NotImplementedError
    def predict(self, state, deterministic=True):
        """
        Select actions given a batch of states.
        The model should NOT train inside this method.

        Also returns the recurrent state if we're using a recurrent network, or None otherwise.
        Returns action, recurrent_state
        """
        raise NotImplementedError
    def save(self, path):
        """
        Save the model to path.
        Returns the actual path saved to (in case it is changed by e.g. adding an extension).
        """
        raise NotImplementedError
    def load(self, path):
        """ Load the model from path. """
        raise NotImplementedError

class TestModel:
    """ Fake model for testing. """
    def __init__(self, env, args):
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape

    def train(self, s, a, r, s2, done):
        print("Test model is pretending to train with {} samples.".format(len(s)))
        return {}

    def predict(self, obs, deterministic=True):
        if obs.shape == self.obs_shape:
            action_shape = self.action_shape
        else:
            assert obs.shape == (len(obs),) + self.obs_shape, \
                    ("obs shape {} does not match expected shape {}"
                            .format(obs.shape, self.obs_shape))
            action_shape = (len(obs),) + self.action_shape
        return np.random.random(action_shape), None

    def save(self, path):
        print("Test model is pretending to save to {}".format(path))
        full_path = path + ".zip"
        f = open(full_path, 'w')
        f.write("Fake model from TestModel.\n")
        f.close()
        return full_path

    def load(self, path):
        print("Test model is pretending to load from {}".format(path))

class BaselinesModel:
    """
    Model that wraps an existing model class using the Stable Baselines interface.
    Contains default implementations of predict, save, and load that forward these calls to
    the underlying model declared in self._model; our interface is based on Stable Baselines, so
    those models each have these 3 methods too.
    """
    def __init__(self):
        # Initialize the inner model in the subclass.
        self._model = None
        raise NotImplementedError

    def predict(self, state, deterministic=True):
        return self._model.predict(state, deterministic=deterministic)
    def save(self, path):
        self._model.save(path)
        # SB adds a .zip by default, unless the path already has an extension.
        # (see stable-baselines/common/base_class.py:576)
        _, ext = os.path.splitext(path)
        if ext == "":
            path += ".zip"
        return path
    def load(self, path):
        self._model.load(path)
