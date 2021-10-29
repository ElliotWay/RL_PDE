import numpy as np

import gym

from rl_pde.run import rollout
from rl_pde.policy import Policy
from rl_pde.agents import ExtendAgentVector
from rl_pde.emi.emi import EMI
from rl_pde.emi.emi import OneDimensionalStencil
from rl_pde.emi.batch import UnbatchedPolicy

class VectorFakeEnv(gym.Env):
    """
    Env that pretends to have 1-dimensional scalar state instead of N-dimensional vector state.

    The spatial dimension is given length 2; this fake environment only makes sense in scenarios
    where the length of the spatial dimension is irrelevant.

    The shape of actions and observations on each dimension must be the same.
    """
    def __init__(self, real_env):
        self.real_env = real_env

        dims = real_env.dimensions
        vec_len = real_env.vec_length
        assert vec_len > 1

        local_action_shape = (real_env.action_space.shape[0],) + real_env.action_space.shape[
                                                                 (dims + 1):]  # 0th is vec_length dimension, plus dims
        local_obs_shape = (real_env.observation_space.shape[0],) + real_env.observation_space.shape[(dims + 1):]

        new_action_shape = (2,) + local_action_shape
        action_space_cls = type(real_env.action_space)
        self.action_space = action_space_cls(shape=new_action_shape,
                low=real_env.action_space.low.flat[0],
                high=real_env.action_space.high.flat[0],
                dtype=real_env.action_space.dtype)

        new_obs_shape = (2,) + local_obs_shape
        obs_space_cls = type(real_env.observation_space)
        self.observation_space = obs_space_cls(shape=new_obs_shape,
                low=real_env.observation_space.low.flat[0],
                high=real_env.observation_space.high.flat[0],
                dtype=real_env.observation_space.dtype)

    def __getattr__(self, attr):
        return getattr(self.real_env, attr)

    def reset(self, *args, **kwargs):
        return self.real_env.reset(*args, **kwargs)


class VectorAdapterEMI(EMI, OneDimensionalStencil):
    """
    EMI pseudo-decorator* that adapts a 1-dimensional scalar state EMI to multiple-dimension vector state EMI.

    *I'm calling this a pseudo-decorator because it needs the decorated EMI class, not the
    decorated EMI object itself. This is necessary here because the decorated EMI needs to see a
    1-dimensional environment, so we need to adjust the environment before passing it to the
    constructor of the decorated EMI.
    """
    def __init__(self, sub_emi_cls, env, model_cls, args, action_adjust=None, obs_adjust=None):

        fake1DEnv = VectorFakeEnv(env)
        self.sub_emi = sub_emi_cls(fake1DEnv, model_cls, args, action_adjust=action_adjust,
                obs_adjust=obs_adjust)

        self.policy = ExtendAgentVector(self.sub_emi.policy)
        self._model = self.sub_emi._model # Needed for loading.

    def training_episode(self, env):
        return self.sub_emi.training_episode(env)

    def get_policy(self):
        return self.policy

    def get_1D_policy(self):
        return self.sub_emi.get_policy()
