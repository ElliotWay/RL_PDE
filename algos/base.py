import numpy as np

def load_algo(model_path):
    """
    Load the model from the path, then load it into the appropriate Algo based on
    information in the model.
    Return the Algo.
    """
    #TODO implement
    raise NotImplementedError

class Algo:
    def training_episode(self, env):
        """
        Train for an episode.
        Not obligated to train, could just collect samples.
        Doesn't HAVE to be one episode either; could be deceptive.
        Return a dict with information. That dict should have
        'avg_reward', 'timesteps'.
        """
        raise NotImplementedError
    def get_policy(self):
        """
        Return a policy object. The policy object should have a predict method with the following
        structure:
        predict(state_batch, deterministic=False) -> action_batch, None
        (The None is for compatibility with recurrent policies.)
        """
        raise NotImplementedError
    def save_model(self, path):
        #TODO what must it contain to identify algo?
        """
        Save the underlying model to a path.
        The model must contain 
        Return the name of the saved path (especially if you changed it).
        """
        raise NotImplementedError
    def load_model(self, path):
        """
        Load a model from a path.
        """
        raise NotImplementedError

class TestAlgo(Algo):
    def __init__(self, action_shape):
        self.action_shape = action_shape
    def training_episode(self, env):
        print("Test algorithm is pretending to train.")
        fake_info = {'avg_reward': np.random.random()*2-1, 'timesteps':100}
        return fake_info
    def predict(self, state, deterministic=False):
        full_shape = (len(state),) + self.action_shape
        return np.random.random(full_shape), None
    def get_policy(self):
        return self
    def save_model(self, path):
        print("Test algorithm is pretending to save to {}".format(path))
        full_path = path + ".zip"
        f = open(full_path, 'w')
        f.write("Fake model from TestAlgo.\n")
        f.close()
        return full_path
    def load_model(self, path):
        print("Test algorithm is pretending to load from {}".format(path))
