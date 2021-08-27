# Adaptations of EMIs for simple toy RL environments.
# The PDE EMIs are written to be fairly general, but still make assumptions about the available
# information, so we need separate EMIs for environments that only have simpler information
# available.

import numpy as np

from rl_pde.emi.emi import EMI, PolicyWrapper
from rl_pde.emi.batch import UnbatchedEnvPL, UnbatchedPolicy

class ToyBatchGlobalEMI(EMI):
    """
    Similar to BatchGlobalEMI, except for simple Toy RL environments.

    Unlike BatchGlobalEMI, ToyBatchGlobalEMI does not access the environment's grid,
    as it likely does not have one. It also does not keep a separate WENO environment with which to
    make comparisons to the RL environment.
    BatchEMI.

    Also, for the info_dict returned from training_episode(), the required 'l2_error' key always
    has the value 0.0.
    """
    def __init__(self, env, model_cls, args, action_adjust=None, obs_adjust=None):
        self._model = model_cls(env, args)

        # The external policy still sees the same interface, so we still use the same decorators as
        # with BatchEMI. Note that unlike in BatchEMI, self.policy is NOT used during training.
        unbatched_env = UnbatchedEnvPL(env, flatten=False)
        unbatched_policy = UnbatchedPolicy(env, unbatched_env, self._model)
        self.policy = PolicyWrapper(unbatched_policy, action_adjust, obs_adjust)

        self.args = args

    def training_episode(self, env):

        num_inits = self.args.batch_size
        initial_conditions = []
        for _ in range(num_inits):
            rl_state = env.reset()
            real_state = env.get_real_state()
        initial_conditions = np.array(initial_conditions)

        extra_info = self._model.train(initial_conditions)

        states = extra_info['states']
        del extra_info['states']
        actions = extra_info['actions']
        del extra_info['actions']
        rewards = extra_info['rewards']
        del extra_info['rewards']

        info_dict = {}
        # Note that information coming from the model
        # has dimensions [timestep, initial_condition, ...], so reducing across time is reducing
        # across axis 0.
        info_dict['reward'] = tuple([np.mean(np.sum(reward_part, axis=0), axis=0) for
                                        reward_part in rewards])
        info_dict['l2_error'] = 0.0
        info_dict['timesteps'] = num_inits * self.args.ep_length
        info_dict.update(extra_info)
        return info_dict

    def get_policy(self):
        return self.policy
