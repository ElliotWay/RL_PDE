import numpy as np

from rl_pde.run import rollout
from rl_pde.emi.emi import EMI, PolicyWrapper

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



