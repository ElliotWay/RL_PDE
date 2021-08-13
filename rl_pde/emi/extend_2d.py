import numpy as np

import gym

from rl_pde.run import rollout
from rl_pde.policy import Policy
from rl_pde.agents import ExtendAgent2D
from rl_pde.emi.emi import EMI
from rl_pde.emi.emi import OneDimensionalStencil
from rl_pde.emi.batch import UnbatchedPolicy

# Not currently used, may be needed for training in 2D environments.
class Unbatched2DFakeEnv(gym.Env):
    """
    Similar to Unbatched2DEnvPL, except also adapts 2 dimensions to 1.

    The shape of actions and observations on the two dimensions must be the same.
    """
    def __init__(self, real_env, flatten=True):
        self.real_env = real_env

        dims = real_env.dimensions
        assert dims > 1

        local_action_shape = real_env.action_space[0][dims:]
        local_obs_shape = real_env.observation_space[0][dims:]
        assert all([local_action_shape == a_space[dims:] and local_obs_shape == o_space[dims:]
                for a_space, o_space in
                zip(real_env.action_space[1:], real_env.observation_space[1:])]), \
                        "Action or observation spaces must match across dimensions."

        if flatten:
            local_action_shape = np.array(local_action_shape).flatten()
            local_obs_shape = np.array(local_obs_shape).flatten()

        action_space_cls = type(real_env.action_space)
        self.action_space = action_space_cls(shape=local_action_shape,
                low=real_env.action_space[0].low.flat[0],
                high=real_env.action_space[0].high.flat[0],
                dtype=real_env.action_space[0].dtype)
        obs_space_cls = type(real_env.observation_space)
        self.observation_space = obs_space_cls(shape=local_obs_shape,
                low=real_env.observation_space[0].low.flat[0],
                high=real_env.observation_space[0].high.flat[0],
                dtype=real_env.observation_space[0].dtype)

    def step(self, action):
        raise Exception("This is a fake env - you can only access the spaces.")
    def reset(self):
        raise Exception("This is a fake env - you can only access the spaces.")
    def render(self, **kwargs):
        raise Exception("This is a fake env - you can only access the spaces.")
    def seed(self, seed):
        raise Exception("This is a fake env - you can only access the spaces.")


class DimensionalFakeEnv(gym.Env):
    """
    Env that pretends to be 1-dimensional instead of N-dimensional.

    The spatial dimension is given length 2; this fake environment only makes sense in scenarios
    where the length of the spatial dimension is irrelevant.

    The shape of actions and observations on each dimension must be the same.
    """
    def __init__(self, real_env):
        self.real_env = real_env

        dims = real_env.dimensions
        assert dims > 1

        local_action_shape = real_env.action_space[0].shape[dims:]
        local_obs_shape = real_env.observation_space[0].shape[dims:]
        assert all([local_action_shape == a_space.shape[dims:]
                and local_obs_shape == o_space.shape[dims:]
                for a_space, o_space in
                zip(real_env.action_space[1:], real_env.observation_space[1:])]), \
                        "Action or observation spaces must match across dimensions."

        new_action_shape = (2,) + local_action_shape
        action_space_cls = type(real_env.action_space[0])
        self.action_space = action_space_cls(shape=new_action_shape,
                low=real_env.action_space[0].low.flat[0],
                high=real_env.action_space[0].high.flat[0],
                dtype=real_env.action_space[0].dtype)

        new_obs_shape = (2,) + local_obs_shape
        obs_space_cls = type(real_env.observation_space[0])
        self.observation_space = obs_space_cls(shape=new_obs_shape,
                low=real_env.observation_space[0].low.flat[0],
                high=real_env.observation_space[0].high.flat[0],
                dtype=real_env.observation_space[0].dtype)

    def step(self, action):
        raise Exception("This is a fake env - you can only access the spaces.")
    def reset(self):
        raise Exception("This is a fake env - you can only access the spaces.")
    def render(self, **kwargs):
        raise Exception("This is a fake env - you can only access the spaces.")
    def seed(self, seed):
        raise Exception("This is a fake env - you can only access the spaces.")


# Not currently used, may be needed for training in 2D environments.
class Unbatched2DPolicy(Policy):
    """
    Policy to wrap policies trained for Unbatched2DFakeEnv. (Similar to UnbatchedPolicy.)

    Functions by applying 2 decorators to the policy: ExtendAgent2D for converting to/from 2D,
    and UnbatchedPolicy for converting to/from batches.
    """
    def __init__(self, original_env, unbatched_env, policy):
        assert original_env.dims > 1

        self.original_env = original_env
        self.unbatched_env = unbatched_env

        # ExtendAgent2D converts from the tuple of 2D grids to 1D slices,
        # then UnbatchedPolicy converts from 1D slices to individual stencils.
        unbatched_policy = UnbatchedPolicy(policy)
        extended_policy = ExtendAgent2D(unbatched_policy)
        self._policy = extended_policy

    def predict(self, obs, deterministic=False):
        return self._policy(obs, deterministic=deterministic)


class DimensionalAdapterEMI(EMI, OneDimensionalStencil):
    """
    EMI pseudo-decorator* that adapts a 1-dimensional EMI to 2 dimensions.

    Only works for testing, not training.
    (You could probably write this to train with the old BatchEMI approach, but not with the new
    BatchGlobalEMI approach.)

    *I'm calling this a pseudo-decorator because it needs the decorated EMI class, not the
    decorated EMI object itself. This is necessary here because the decorated EMI needs to see a
    1-dimensional environment, so we need to adjust the environment before passing it to the
    constructor of the decorated EMI.
    """
    def __init__(self, sub_emi_cls, env, model_cls, args, action_adjust=None, obs_adjust=None):

        fake1DEnv = DimensionalFakeEnv(env)
        self.sub_emi = sub_emi_cls(fake1DEnv, model_cls, args, action_adjust=action_adjust,
                obs_adjust=obs_adjust)

        self.policy = ExtendAgent2D(self.sub_emi.policy)
        self._model = self.sub_emi._model # Needed for loading.

    def training_episode(self, env):
        return self.sub_emi.training_episode(env)
        #raise Exception("DimensionalAdapterEMI: This adapter cannot be used for training.")

    def get_policy(self):
        return self.policy

    def get_1D_policy(self):
        return self.sub_emi.get_policy()
