import numpy as np

from rl_pde.run import rollout
from rl_pde.policy import Policy

class EMI:
    """
    An EMI is an Environment Model Interface.

    An EMI controls how information is passed between the environment
    and the model. Traditionally in RL information is passed from the environment
    directly to the model, but in this project we're treating the state as a batch of many states
    and other changes. However, you can still get this behavior with StandardEMI, an entirely
    transparent interface.

    An EMI also optionally exposes hooks around interactions with the policy, such as normalizing
    the observation before feeding it to the model, or normalizing the action before feeding it
    to the environment. Note that the model is typically aware of the environment and may do some
    of its own adjustments; this is only necessary for adjustments that the model is not aware of
    needing.

    This is an abstract class that defines the interface of an EMI, and provides simple default
    implementations of save_model and load_model.
    Subclasses must either declare self._model or override save_model and load_model.
    """
    def __init__(self, env, model_cls, args, action_adjust=None, obs_adjust=None):
        # Initialize the model in the subclass (or do something else).
        self._model = None
        raise NotImplementedError
    def training_episode(self, env):
        """
        Train for an episode.
        Not obligated to train, could just collect samples.
        Doesn't HAVE to be one episode either; could be deceptive.
        Return a dict with information. That dict should have
        'reward', 'l2_error', and 'timesteps'. 'reward' should be the undiscounted return,
        'l2_error' should the l2_error of the final state. They may be averages, instead
        of just one value, if you're actually running multiple episodes.
        """
        raise NotImplementedError
    def get_policy(self):
        """
        Return a policy object. The policy object must have a predict method with the following
        structure:
        predict(state_batch, deterministic=False) -> action_batch, None
        (The None is for compatibility with recurrent policies.)
        """
        raise NotImplementedError
    def save_model(self, path):
        """
        Save the underlying model to a path.
        (Forwards the save call to the model, unless a subclass does
        something else.)
        Returns the name of the saved path (which is relevant as it might be changed to add a .zip
        to .pkl or whatever).
        """
        return self._model.save(path)
    def load_model(self, path):
        """
        Load a model from a path.
        """
        self._model.load(path)


class TestEMI(EMI):
    """ Fake EMI for testing. """
    def __init__(self, env, model_cls, args, action_adjust=None, obs_adjust=None):
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape
    def training_episode(self, env):
        print("Test EMI is pretending to train.")
        fake_info = {'reward': np.random.random()*2-1, 'l2_error': 0.0, 'timesteps':100}
        return fake_info
    def predict(self, obs, deterministic=False):
        if obs.shape == self.obs_shape:
            action_shape = self.action_shape
        else:
            assert obs.shape == (len(obs),) + self.obs_shape
            action_shape = (len(obs),) + self.action_shape
        return np.random.random(action_shape), None
    def get_policy(self):
        return self
    def save_model(self, path):
        print("Test EMI is pretending to save to {}".format(path))
        full_path = path + ".zip"
        f = open(full_path, 'w')
        f.write("Fake model from TestEMI.\n")
        f.close()
        return full_path
    def load_model(self, path):
        print("Test EMI is pretending to load from {}".format(path))


class PolicyWrapper(Policy):
    """
    Wraps a Model or Policy, providing only the predict() interface.

    Controls state/action adjusment as necessary.

    Use save_model_samples() before training to record observations and actions from the
    perspective of the model, i.e. after adjustments to the observation but before adjustments to
    the action.
    Use get_model_samples() to get the collected obs, actions. This also turns off sample recording;
    use save_model_samples() again if you are going to continue training.
    """
    def __init__(self, model, action_adjust=None, obs_adjust=None):
        self.model = model
        self.action_adjust = action_adjust
        self.obs_adjust = obs_adjust

        self.save_samples = False
        self.model_obs = []
        self.model_action = []

    def save_model_samples(self):
        self.model_obs = []
        self.model_action = []
        self.save_samples = True

    def get_model_samples(self):
        self.save_samples = False
        return self.model_obs, self.model_action

    def predict(self, obs, deterministic=False):
        if self.obs_adjust is not None:
            adjusted_obs = self.obs_adjust(obs)
        else:
            adjusted_obs = obs
        if self.save_samples:
            self.model_obs.append(adjusted_obs)

        action, info = self.model.predict(adjusted_obs, deterministic=deterministic)
        if self.save_samples:
            self.model_action.append(action)

        if self.action_adjust is not None:
            adjusted_action = self.action_adjust(action)
        else:
            adjusted_action = action
        return adjusted_action, info


class StandardEMI(EMI):
    """
    EMI that simply takes samples from the environment and gives them to the model.
    (It still applies a PolicyWrapper to potentially modify actions and observations.)
    """
    def __init__(self, env, model_cls, args, action_adjust=None, obs_adjust=None):
        self._model = model_cls(env, args)
        self.original_env = env
        self.policy = PolicyWrapper(self._model, action_adjust, obs_adjust)

    def training_episode(self, env):
        self.policy.save_model_samples()
        _, _, r, done, raw_s2 = rollout(env, self.policy)

        # Training requires the state/actions from the perspective of the model, not the
        # perspective of the environment.
        s, a = self.policy.get_model_samples()
        if self.policy.obs_adjust is not None:
            last_state = self.policy.obs_adjust(raw_s2[-1])
        else:
            last_state = raw_s2[-1]
        s2 = s[1:] + [last_state]

        extra_info = self._model.train(s, a, r, s2, done)

        # Take mean over any remaining dimensions, even though we're acting like r must be only 1
        # dimensional here.
        total_reward = np.mean(np.sum(r, axis=0))
        l2_error = env.compute_l2_error()
        timesteps = len(s)

        info_dict = {'reward':total_reward, 'l2_error':l2_error, 'timesteps':timesteps}
        info_dict.update(extra_info)
        return info_dict

    def get_policy(self):
        return self.policy
