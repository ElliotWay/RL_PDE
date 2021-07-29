import numpy as np
from argparse import Namespace

import gym

from rl_pde.run import rollout
from rl_pde.policy import Policy
from rl_pde.emi.emi import EMI, PolicyWrapper

class UnbatchedEnvPL(gym.Env):
    """
    Fake environment that presents a state/action space from another environment with the first
    dimension (the spatial dimension) removed, and optionally the remaining dimensions flattened.
    """
    def __init__(self, real_env, flatten=True):
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

        new_action_low = real_action_low[0]
        new_action_high = real_action_high[0]
        new_obs_low = real_obs_low[0]
        new_obs_high = real_obs_high[0]

        if flatten:
            new_action_low = new_action_low.flatten()
            new_action_high = new_action_high.flatten()
            new_obs_low = new_obs_low.flatten()
            new_obs_high = new_obs_high.flatten()

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

class UnbatchedPolicy(Policy):
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

        avg_total_reward = np.mean(np.sum(reward, axis=0))
        l2_error = env.compute_l2_error()
        timesteps = len(state)
        info_dict = {'reward':avg_total_reward, 'l2_error':l2_error, 'timesteps':timesteps}
        info_dict.update(extra_info)
        return info_dict

    def get_policy(self):
        return self.policy
