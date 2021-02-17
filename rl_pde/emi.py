import numpy as np

import gym

from rl_pde.run import rollout

# We need refs to environments and models to define default interactions between them.
#from models import SACModel
#from envs.burgers_envs import WENOBurgersEnv, SplitFluxBurgersEnv, FluxBurgersEnv

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
    """ Fake EMI for testing. """
    def __init__(self, env, model_cls, args, action_adjust=None, obs_adjust=None):
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape
    def training_episode(self, env):
        print("Test EMI is pretending to train.")
        fake_info = {'avg_reward': np.random.random()*2-1, 'timesteps':100}
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

class PolicyWrapper:
    """
    Wraps a model, providing only the predict() interface.

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

    def predict(self, obs, *args, **kwargs):
        if self.obs_adjust is not None:
            adjusted_obs = self.obs_adjust(obs)
        else:
            adjusted_obs = obs
        if self.save_samples:
            self.model_obs.append(adjusted_obs)

        action, info = self.model.predict(adjusted_obs, *args, **kwargs)
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

        avg_reward = np.mean(r)
        timesteps = len(s)

        info_dict = {'avg_reward': avg_reward, 'timesteps':timesteps}
        info_dict.update(extra_info)
        return info_dict

    def get_policy(self):
        return self.policy

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

        assert (np.all(real_action_low[0] == real_action_low[-1])
                and np.all(real_action_high[0] == real_action_high[-1])
                and np.all(real_obs_low[0] == real_obs_low[-1])
                and np.all(real_obs_high[0] == real_obs_high[-1])), \
                "Original env dim 1 was not spatial."

        new_action_low = real_action_low[0].flatten()
        new_action_high = real_action_high[0].flatten()
        new_obs_low = real_obs_low[0].flatten()
        new_obs_high = real_obs_high[0].flatten()

        action_space_cls = type(real_env.action_space)
        self.action_space = action_space_cls(low=new_action_low, high=new_action_high,
                                             dtype=real_env.action_space.dtype)
        obs_space_cls = type(real_env.observation_space)
        self.observation_space = obs_space_cls(low=new_obs_low, high=new_obs_high,
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
        self.unbatched_action_shape = unbatched_env.action_space.shape
        self._model = model

    def predict(self, obs, deterministic=False):
        obs = np.array(obs)
        if obs.shape[1:] == self.original_obs_shape:
            vec_obs = True
        else:
            assert obs.shape[1:] == self.original_obs_shape[1:], \
                    ("original obs shape {} does not match new shape {}"
                            .format(obs.shape, self.original_obs_shape))
            vec_obs = False
        obs = obs.reshape((len(obs),) + self.unbatched_obs_shape)
        actions, _ = self._model.predict(obs, deterministic=deterministic)
        
        if vec_obs:
            actions = actions.reshape((-1,) + self.original_action_shape)
        else:
            actions = actions.reshape((len(obs),) + self.original_action_shape[1:])

        return actions, None

class BatchEMI(EMI):
    """
    EMI that takes samples from the environment and breaks them along the first dimension
    (the spatial dimension), then gives them to the model as separate samples.
    """
    def __init__(self, env, model_cls, args, action_adjust=None, obs_adjust=None):
        self.original_env = env
        self.unbatched_env = UnbatchedEnvPL(env)
        self._model = model_cls(self.unbatched_env, args)

        # Oh yes, we've got TWO decorators for the policy here.
        unbatched_policy = UnbatchedPolicy(self.original_env, self.unbatched_env, self._model)
        self.policy = PolicyWrapper(unbatched_policy, action_adjust, obs_adjust)

    def training_episode(self, env):
        self.policy.save_model_samples()
        _, _, reward, done, raw_new_state = rollout(env, self.policy)
    
        # Training requires the state/actions from the perspective of the model, not the
        # perspective of the environment.
        state, action = self.policy.get_model_samples()
        if self.policy.obs_adjust is not None:
            last_state = self.policy.obs_adjust(raw_new_state[-1])
        else:
            last_state = raw_new_state[-1]
        new_state = state[1:] + [last_state]

        # Convert batched trajectory into list of samples while preserving consecutive
        # trajectories. Also flatten states and actions.
        def unbatch(arr):
            arr = np.array(arr)
            arr = np.swapaxes(arr, 0, 1)
            num_samples = arr.shape[0] * arr.shape[1]
            flattened_shape = np.prod(arr.shape[2:])
            arr = arr.reshape((num_samples, flattened_shape))
            return arr
        unbatched_state = unbatch(state)
        unbatched_action = unbatch(action)
        unbatched_reward = np.array(reward).transpose().flatten()
        unbatched_done = np.tile(done, len(state[0]))
        unbatched_new_state = unbatch(new_state)
        extra_info = self._model.train(unbatched_state, unbatched_action, unbatched_reward,
                unbatched_new_state, unbatched_done)

        avg_reward = np.mean(reward)
        timesteps = len(state)
        info_dict = {'avg_reward': avg_reward, 'timesteps':timesteps}
        info_dict.update(extra_info)
        return info_dict

    def get_policy(self):
        return self.policy

class HomogenousMARL_EMI(BatchEMI):
    """
    EMI that structues the environment as a MARL problem with homogenous agents.
    It takes samples from the environment, keeps the time and spatial dimension while flattening
    the rest, and passes this to the model.
    
    It is up to the model to interpret the spatial dimension as simultaneous agents.

    The environment space from the presented to the model is the shape of the individual actions,
    i.e. the flattened shape after removing time and spatial dimensions. This requires the model to
    create homogenous agents that are location-agnostic, even though the model receives information
    about location during training.
    """
    def training_episode(self, env):
        self.policy.save_model_samples()
        _, _, reward, done, raw_new_state = rollout(env, self.policy)
    
        # Training requires the state/actions from the perspective of the model, not the
        # perspective of the environment.
        state, action = self.policy.get_model_samples()
        if self.policy.obs_adjust is not None:
            last_state = self.policy.obs_adjust(raw_new_state[-1])
        else:
            last_state = raw_new_state[-1]
        new_state = state[1:] + [last_state]

        # Unlike with BatchEMI, we don't need to reorder the data. We still need to flatten states
        # and actions, however.
        def partial_flatten(arr):
            arr = np.array(arr)
            flattened_shape = np.prod(arr.shape[2:])
            new_shape = arr.shape[:2] + (flattened_shape,)
            arr = arr.reshape(new_shape)
            return arr
        state = partial_flatten(state)
        action = partial_flatten(action)
        reward = np.array(reward)
        unbatched_done = np.repeat(done, len(state[0])).reshape((-1, len(state[0])))
        new_state = partial_flatten(new_state)
        extra_info = self._model.train(state, action, reward, new_state, unbatched_done)

        avg_reward = np.mean(reward)
        timesteps = len(state)
        info_dict = {'avg_reward': avg_reward, 'timesteps':timesteps}
        info_dict.update(extra_info)
        return info_dict
