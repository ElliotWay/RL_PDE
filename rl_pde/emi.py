import numpy as np

import gym

from rl_pde.run import rollout

class EMI:
    """
    An EMI is an Environment Model Interface.

    An EMI controls how information is passed between the environment
    and the model. Traditionally in RL information is passed from the environment
    directly to the model, but in this project we're treating the state as a batch of many states
    and other changes. However, you can still get this behavior with StandardEMI, an entirely
    transparent interface.

    This is an abstract class that defines the interface of an EMI, and provides simple default
    implementations of save_model and load_model.
    Subclasses must either declare self._model or override save_model and load_model.
    """
    def __init__(self, env, model_cls, args):
        # Initialize the model in the subclass (or do something else).
        self._model = None
        raise NotImplementedError
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
    """
    Fake EMI for testing.
    """
    def __init__(self, env, model_cls, args):
        self.action_shape = env.action_space.shape
    def training_episode(self, env):
        print("Test EMI is pretending to train.")
        fake_info = {'avg_reward': np.random.random()*2-1, 'timesteps':100}
        return fake_info
    def predict(self, state, deterministic=False):
        full_shape = (len(state),) + self.action_shape
        return np.random.random(full_shape), None
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

class StandardEMI(EMI):
    """
    EMI that simply takes samples from the environment and gives them to the model.
    Not really intended for use in this project; this class is here for comparison.
    """
    def __init__(env, model_cls, args):
        self._model = model_cls(env=env, args)
        self.original_env = env

    def training_episode(self, env):
        s, a, r, done, s2 = rollout(env, self.model)

        extra_info = model.train(s, a, r, s2, done)

        avg_reward = np.mean(r, axis=0)
        timesteps = len(s)

        info_dict = {'avg_reward': avg_reward, 'timesteps':timesteps}
        info_dict.update(extra_info)
        return info_dict

    def get_policy(self):
        return self._model

class UnbatchedEnvPL(gym.Env):
    """
    Fake environment that presents a state/action space from another environment with the first
    dimension (the spatial dimension) removed, and the remaining dimensions flattened.
    """
    def __init__(self, real_env):
        self.real_env = real_env

        real_action_low = real_env.action_space.low
        real_action_high = real_env.action_space.high
        real_obs_low = real_env.observation_space.low
        real_obs_high = real_env.observation_space.high

        assert (real_action_low[0] == real_action_low[-1]
                and real_action_high[0] == real_action_high[-1]
                and real_obs_low[0] == real_obs_low[-1]
                and real_obs_high[0] == real_obs_high[-1]), "Original env dim 1 was not spatial."

        new_action_low = real_action_low[0].flatten()
        new_action_high = real_action_high[0].flatten()
        new_obs_low = real_obs_low[0].flatten()
        new_obs_high = real_obs_high[0].flatten()

        self.action_space = gym.spaces.Box(low=new_action_low, hgh=new_action_low,
                                           dtype=real_env.action_space.dtype)
        self.observation_space = gym.spaces.Box(low=new_obs_low, hgh=new_obs_low,
                                                dtype=real_env.observation_space.dtype)

    def step(self, action):
        raise Exception("This is a placeholder env - you can only access the spaces.")
    def reset(self):
        raise Exception("This is a placeholder env - you can only access the spaces.")
    def render(self, **kwargs):
        raise Exception("This is a placeholder env - you can only access the spaces.")
    def seed(self, seed):
        raise Exception("This is a placeholder env - you can only access the spaces.")

class UnbatchedPolicy:
    """
    Policy to wrap policies trained for an UnbatchedEnvPL.
    Reshapes observations before feeding them to the model's predict function, then reshapes the
    actions the model's predict outputs.
    """
    def __init__(self, original_env, unbatched_env, model):
        assert isinstance(unbatched_env, UnbatchedEnvPL)
        self.original_obs_shape = original_env.observation_space.shape
        self.unbatched_obs_shape = unbatched_env.observation_space.shape
        self.original_action_shape = original_env.action_space.shape
        self.unbatched_action_shape = unbatched_env_action_space.shape
        self._model = model

    def predict(self, obs, deterministic=False):
        obs = np.array(obs)
        if obs.shape[1:] == self.original_obs_shape:
            vec_obs = True
        else:
            assert obs.shape == self.original_obs_shape
            vec_obs = False
        obs = obs.reshape((-1,) + self.unbatched_obs_shape)
        actions, _ = self._model.predict(obs, deterministic=deterministic)
        
        if vec_obs:
            actions = actions.reshape((-1,) + self.original_action_shape)
        else:
            actions = actions.reshape(self.original_action_shape)

        return actions, None

class BatchEMI(EMI):
    """
    EMI that takes samples from the environment and breaks them along the first dimension
    (the spatial dimension), then gives them to the model as separate samples.
    """
    def __init__(env, model_cls, args):
        self.original_env = env
        self.unbatched_env = UnbatchedEnvPL(env)
        self._model = model_cls(env=self.unbatched_env, args)
        self.policy = UnbatchedPolicy(self.original_env, self.unbatched_env, self._model)

    def training_episode(self, env):
        state, action, reward, done, new_state = rollout(env, self._model)

        # Convert batched trajectory into list of samples containing consecutive trajectories.
        def unbatch(arr):
            arr = np.array(arr)
            arr = np.swapaxes(arr, 0, 1)
            arr = arr.reshape((-1,) + arr.shape[2:])
            return arr
        unbatched_state = unbatch(state)
        unbatched_action = unbatch(action)
        unbatched_reward = unbatch(reward)
        unbatched_done = np.tile(done, len(state[0]))
        unbatched_new_state = unbatch(new_state)
        extra_info = self._model.train(unbatched_state, unbatched_action, unbatched_reward,
                unbatched_new_state, unbatched_done)

        avg_reward = np.mean(reward, axis=0)
        timesteps = len(state)
        info_dict = {'avg_reward': avg_reward, 'timesteps':timesteps}
        info_dict.update(extra_info)
        return info_dict

    def get_policy(self):
        return self.policy

