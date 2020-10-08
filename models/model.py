import os

class Model:
    """
    RL model representing the ability to train on experience.
    """
    def __init__(self, env, args):
        """ Create a model for the given env using the args namespace. """
        raise NotImplementedError
    def train(s, a, r, s2, done):
        """
        Train using samples.
        Should return a dict of information (may be empty dict).
        """
        raise NotImplementedError
    def predict(state, deterministic=True):
        """
        Select actions given a batch of states.
        The model should NOT train inside this method.
        """
        raise NotImplementedError
    def save(path):
        """ Save the model to path. """
        raise NotImplementedError
    def load(path):
        """ Load the model from path. """
        raise NotImplementedError

class TestModel:
    """ Fake model for testing. """
    def __init__(self, env, args):
        self.action_shape = env.action_space.shape
    def train(s, a, r, s2, done):
        print("Test model is pretending to train with {} samples.".format(len(s)))

    def predict(state, deterministic=True):
        full_shape = (len(state),) + self.action_shape
        return np.random.random(full_shape), None
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

    def predict(state, deterministic=True):
        return self._model.predict(state, deterministic=deterministic)
    def save(path):
        self._model.save(path)
        # SB adds a .zip by default, unless the path already has an extension.
        # (see stable-baselines/common/base_class.py:576)
        _, ext = os.path.splitext(path)
        if ext == "":
            path += ".zip"
        return path
    def load(path):
        self._model.load(path)
